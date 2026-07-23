"""
Traitement géographique et requêtes PostGIS.

Responsabilités :
  1. Chargement des métadonnées régionales (22 régions)
  2. Calcul de proximité géographique entre régions
  3. Enrichissement des features avec données géo (altitude, lat/lon, zone)
  4. Requêtes PostGIS pour zones humides et données satellitaires
  5. Encodage des attributs géographiques pour le ML

Note : Conçu pour fonctionner avec ou sans session DB active
(compatible mode synchrone Celery FakeDB).

Appelé par :
  - src/preprocessing/feature_engineering.py

─────────────────────────────────────────────────────────────────────
CORRECTIF (voir conversation) :
  - get_zones_humides_pct utilisait ST_Intersection(r.geom, w.geom), mais
    la table regions n'a PAS de colonne "geom" (seulement centroid et
    bbox_geom) — cette requête ne pouvait jamais fonctionner. geo_wetlands
    a déjà sa propre colonne region_id : plus besoin d'intersection
    spatiale, un simple filtre + ratio superficie suffit. La clé de
    regions est aussi "code", pas "region_id".
  - get_region_ndvi passait la date en str() — asyncpg (contrairement à
    psycopg2) exige un objet date natif, pas une chaîne.
  - Ajout d'un rollback après chaque requête échouée : sous asyncpg, une
    requête cassée met la transaction en état "aborted" et fait échouer
    TOUTES les requêtes suivantes sur la même session tant qu'un ROLLBACK
    n'a pas été fait — y compris celles de feature_engineering.py qui
    partage la même session.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.preprocessing.weather_processor import ZONE_CLIMATIQUE_ENCODING


# ─────────────────────────────────────────────────────────────────
# Chargement des métadonnées régionales (cache mémoire)
# ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_regions_metadata() -> Dict[str, Any]:
    """Charge et met en cache le fichier regions_metadata.json."""
    path = Path("config/regions_metadata.json")
    if not path.exists():
        logger.warning("regions_metadata.json introuvable — métadonnées géo vides")
        return {"regions": []}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_region_index() -> Dict[str, Dict]:
    """Construit un index {region_id: metadata} pour accès O(1)."""
    meta = _load_regions_metadata()
    return {r["id"]: r for r in meta.get("regions", [])}


class GeoProcessor:
    """
    Processeur géographique — enrichissement des features avec les
    données géo-spatiales des 22 régions de Madagascar.

    Fonctionne en mode "metadata only" (pas de session DB requise),
    ce qui le rend compatible avec les tâches Celery synchrones.

    Pour les requêtes PostGIS avancées, la session DB est optionnelle.
    """

    def __init__(self, db=None):
        """
        Args:
            db : session SQLAlchemy async (optionnelle).
                 Si None → mode metadata only.
        """
        self._db = db
        self._region_index = _build_region_index()

    # ─────────────────────────────────────────────
    # Métadonnées régionales
    # ─────────────────────────────────────────────

    def get_region_meta(self, region_id: str) -> Dict[str, Any]:
        """
        Retourne les métadonnées complètes d'une région.
        Fallback vers valeurs par défaut si région inconnue.
        """
        meta = self._region_index.get(region_id)
        if meta is None:
            logger.warning("Région inconnue : {} — métadonnées par défaut", region_id)
            return self._default_region_meta(region_id)
        return meta

    def get_geo_features(self, region_id: str) -> Dict[str, Any]:
        """
        Retourne les features géographiques pour une région.

        Clés retournées (utilisées par FeatureEngineer) :
          latitude, longitude, altitude_m,
          zone_climatique_encoded, endemicite_encoded,
          densite_population_norm, pct_enfants_5ans,
          indice_vulnerabilite
        """
        meta = self.get_region_meta(region_id)

        latitude   = float(meta.get("latitude", -18.9))
        longitude  = float(meta.get("longitude", 47.5))
        altitude_m = float(meta.get("altitude_mean_m", 800))
        population = int(meta.get("population_2023", 500_000))
        area_km2   = float(meta.get("area_km2", 10_000))
        zone       = meta.get("climate_zone", "tropical_sub_humid")
        endemicite = meta.get("malaria_endemicity", "medium")

        # Encodages pour ML
        zone_encoded = ZONE_CLIMATIQUE_ENCODING.get(zone, 2)
        endemicite_encoded = self._encode_endemicite(endemicite)

        # Densité population normalisée (log scale)
        densite_norm = round(
            float(np.log1p(population / max(area_km2, 1))) / 10, 4
        ) if area_km2 > 0 else 0.0

        return {
            "region_id":              region_id,
            "latitude":               latitude,
            "longitude":              longitude,
            "altitude_m":             altitude_m,
            "zone_climatique_encoded": float(zone_encoded),
            "endemicite_encoded":     float(endemicite_encoded),
            "densite_population_norm": densite_norm,
            "pct_enfants_5ans":       0.17,  # Constante démographique Madagascar
            "population":             population,
            "area_km2":               area_km2,
            "indice_vulnerabilite":   self._compute_vulnerability_index(meta),
        }

    def get_all_regions_geo(self) -> List[Dict[str, Any]]:
        """Retourne les features géographiques pour toutes les 22 régions."""
        return [
            self.get_geo_features(rid)
            for rid in self._region_index.keys()
        ]

    # ─────────────────────────────────────────────
    # Calculs de proximité (pour features spatiales)
    # ─────────────────────────────────────────────

    def get_neighboring_regions(
        self,
        region_id: str,
        max_distance_km: float = 200.0,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les régions voisines dans un rayon donné.
        Utile pour les features de contagion spatiale.
        """
        meta = self.get_region_meta(region_id)
        lat1 = meta.get("latitude", 0)
        lon1 = meta.get("longitude", 0)

        neighbors = []
        for rid, rmeta in self._region_index.items():
            if rid == region_id:
                continue
            lat2 = rmeta.get("latitude", 0)
            lon2 = rmeta.get("longitude", 0)
            dist = self._haversine_km(lat1, lon1, lat2, lon2)
            if dist <= max_distance_km:
                neighbors.append({
                    "region_id": rid,
                    "region_name": rmeta.get("name", rid),
                    "distance_km": round(dist, 1),
                })

        return sorted(neighbors, key=lambda x: x["distance_km"])

    @staticmethod
    def _haversine_km(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Distance Haversine entre deux points GPS (km)."""
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi   = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        )
        return float(2 * R * np.arcsin(np.sqrt(a)))

    # ─────────────────────────────────────────────
    # Requêtes PostGIS (optionnelles — DB required)
    # ─────────────────────────────────────────────

    async def get_zones_humides_pct(self, region_id: str) -> float:
        """
        Calcule le % de zones humides dans la région à partir de
        geo_wetlands.superficie_ha / regions.superficie_km2.

        CORRIGÉ — l'ancienne requête faisait ST_Intersection(r.geom, w.geom)
        mais regions n'a pas de colonne "geom" (seulement centroid et
        bbox_geom) : cette requête ne pouvait jamais fonctionner. geo_wetlands
        porte déjà sa propre colonne region_id, donc un simple filtre +
        somme des superficies suffit, sans jointure spatiale.
        """
        if self._db is None or not self._is_real_db():
            return self._estimate_zones_humides(region_id)

        try:
            from sqlalchemy import text
            result = await self._db.execute(
                text("""
                    SELECT
                        COALESCE(SUM(w.superficie_ha), 0) AS wetlands_ha,
                        r.superficie_km2
                    FROM regions r
                    LEFT JOIN geo_wetlands w ON w.region_id = r.code
                    WHERE r.code = :region_id
                    GROUP BY r.superficie_km2
                """),
                {"region_id": region_id}
            )
            row = result.fetchone()
            if not row or not row.superficie_km2 or row.superficie_km2 <= 0:
                return self._estimate_zones_humides(region_id)
            superficie_ha = float(row.superficie_km2) * 100.0
            pct = float(row.wetlands_ha) / superficie_ha * 100.0
            return round(min(pct, 100.0), 2)
        except Exception as exc:
            logger.debug("DB zones_humides {} : {}", region_id, exc)
            await self._safe_rollback()
            return self._estimate_zones_humides(region_id)

    async def get_region_ndvi(
        self, region_id: str, target_date=None
    ) -> Optional[float]:
        """
        Requête DB pour le NDVI moyen de la région (depuis table geo_ndvi).
        Fallback heuristique si données non disponibles.
        """
        if self._db is None or not self._is_real_db():
            return self._estimate_ndvi(region_id)

        try:
            from datetime import date as dt
            from sqlalchemy import text
            target = target_date or dt.today()

            result = await self._db.execute(
                text("""
                    SELECT ndvi_mean
                    FROM geo_ndvi
                    WHERE region_id = :region_id
                      AND observation_date <= :target_date
                    ORDER BY observation_date DESC
                    LIMIT 1
                """),
                {"region_id": region_id, "target_date": target}
            )
            row = result.fetchone()
            return round(float(row[0]), 3) if row else self._estimate_ndvi(region_id)
        except Exception as exc:
            logger.debug("DB NDVI {} : {}", region_id, exc)
            await self._safe_rollback()
            return self._estimate_ndvi(region_id)

    # ─────────────────────────────────────────────
    # Estimations heuristiques (fallback sans DB)
    # ─────────────────────────────────────────────

    def _estimate_zones_humides(self, region_id: str) -> float:
        """
        Estimation du % de zones humides basée sur la zone climatique et
        les précipitations moyennes. Utilisée quand PostGIS est indisponible.
        """
        meta = self.get_region_meta(region_id)
        zone = meta.get("climate_zone", "tropical_sub_humid")
        alt  = float(meta.get("altitude_mean_m", 800))

        base_humide: Dict[str, float] = {
            "tropical_humid":     35.0,
            "tropical_sub_humid": 20.0,
            "tropical_highland":  15.0,
            "tropical_dry":       8.0,
            "arid":               2.0,
            "semi_arid":          4.0,
            "tropical_sub_arid":  6.0,
        }

        base = base_humide.get(zone, 15.0)
        # Réduction avec l'altitude
        alt_factor = max(0.3, 1.0 - (alt - 500) / 3000)
        return round(base * alt_factor, 1)

    def _estimate_ndvi(self, region_id: str) -> float:
        """
        Estimation NDVI basée sur zone climatique et saison.
        """
        from datetime import date
        meta  = self.get_region_meta(region_id)
        zone  = meta.get("climate_zone", "tropical_sub_humid")
        mois  = date.today().month

        ndvi_base: Dict[str, float] = {
            "tropical_humid":     0.65,
            "tropical_sub_humid": 0.45,
            "tropical_highland":  0.40,
            "tropical_dry":       0.25,
            "arid":               0.10,
            "semi_arid":          0.15,
            "tropical_sub_arid":  0.20,
        }

        base = ndvi_base.get(zone, 0.35)
        # Bonus en saison des pluies (Nov-Avr)
        if mois in (11, 12, 1, 2, 3, 4):
            base = min(0.85, base * 1.2)

        return round(base, 3)

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _is_real_db(self) -> bool:
        """Vérifie si la session DB est une vraie session async SQLAlchemy."""
        db_type = type(self._db).__name__
        return "Session" in db_type or "AsyncSession" in db_type

    async def _safe_rollback(self) -> None:
        """
        Annule la transaction en cours après une requête échouée.

        Important : self._db est la MÊME session partagée avec
        FeatureEngineer (et HealthProcessor le cas échéant). Sans ce
        rollback, une requête cassée ici empoisonne aussi les requêtes
        suivantes faites ailleurs sur cette session.
        """
        if self._db is None or not self._is_real_db():
            return
        try:
            await self._db.rollback()
        except Exception as exc:
            logger.debug("Rollback échoué (session probablement déjà fermée) : {}", exc)

    @staticmethod
    def _encode_endemicite(endemicite: str) -> int:
        """Encode le niveau d'endémicité en entier ordinal."""
        mapping = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
        return mapping.get(endemicite, 1)

    @staticmethod
    def _compute_vulnerability_index(meta: Dict) -> float:
        """
        Calcule un indice de vulnérabilité [0,1] à partir des métadonnées.
        Combinaison zone climatique + endémicité + densité sanitaire.
        """
        zone = meta.get("climate_zone", "tropical_sub_humid")
        endemicite = meta.get("malaria_endemicity", "medium")
        health_facilities = meta.get("health_facilities", 100)
        population = meta.get("population_2023", 500_000)

        zone_vuln = {
            "tropical_humid": 0.7, "tropical_sub_humid": 0.5,
            "tropical_highland": 0.3, "tropical_dry": 0.5,
            "arid": 0.8, "semi_arid": 0.7, "tropical_sub_arid": 0.6,
        }.get(zone, 0.5)

        endem_vuln = {"low": 0.2, "medium": 0.4, "high": 0.7, "very_high": 0.9}.get(
            endemicite, 0.4
        )

        # Ratio structures de santé / population (plus bas = plus vulnérable)
        sante_ratio = min(1.0, health_facilities / max(1, population / 5000))
        sante_vuln  = 1.0 - sante_ratio

        # Moyenne pondérée
        index = 0.35 * zone_vuln + 0.40 * endem_vuln + 0.25 * sante_vuln
        return round(float(np.clip(index, 0, 1)), 3)

    @staticmethod
    def _default_region_meta(region_id: str) -> Dict[str, Any]:
        """Métadonnées par défaut pour une région inconnue."""
        return {
            "id": region_id,
            "name": region_id,
            "latitude": -20.0,
            "longitude": 47.0,
            "altitude_mean_m": 800,
            "area_km2": 20_000,
            "population_2023": 500_000,
            "climate_zone": "tropical_sub_humid",
            "malaria_endemicity": "medium",
            "health_facilities": 80,
        }