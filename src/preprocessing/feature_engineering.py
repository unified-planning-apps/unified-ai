"""
Orchestrateur central du feature engineering.

Contrats GARANTIS avec les modèles ML (ordre et noms de clés) :

  build_malaria_features(region_id) → dict avec EXACTEMENT les 26 clés de
    MALARIA_FEATURE_NAMES (src/models/malaria_predictor.py)

  build_nutrition_features(region_id) → dict avec EXACTEMENT les 32 clés de
    NUTRITION_FEATURE_NAMES (src/models/nutrition_predictor.py)

Compatibilité :
  - Mode async (FastAPI routers) : session SQLAlchemy async réelle
  - Mode sync (Celery scheduler) : FakeDB() stub — toutes les sources
    d'enrichissement DB sont gérées avec fallback gracieux

Appelé par :
  - src/api/routers/malaria.py
  - src/api/routers/nutrition.py
  - src/api/routers/predictions.py
  - src/data_collection/scheduler.py (_calculer_prediction_combinee)
  - ml/training_scripts/train_malaria.py
  - ml/training_scripts/train_nutrition.py
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.models.malaria_predictor import MALARIA_FEATURE_NAMES
from src.models.nutrition_predictor import NUTRITION_FEATURE_NAMES
from src.preprocessing.geo_processor import GeoProcessor
from src.preprocessing.health_processor import HealthProcessor
from src.preprocessing.weather_processor import WeatherProcessor


class FeatureEngineer:
    """
    Orchestrateur du feature engineering.

    Construit les vecteurs de features pour les deux modèles ML
    en agrégeant toutes les sources de données disponibles :
      - Données météo (WeatherFetcher + WeatherProcessor)
      - Données épidémiologiques (MalariaFetcher + HealthProcessor)
      - Données nutritionnelles (NutritionFetcher + HealthProcessor)
      - Données géographiques (GeoProcessor + régions_metadata.json)

    Garantit que les dicts retournés contiennent EXACTEMENT les features
    attendues par les modèles — les clés manquantes sont imputées à 0.0.
    """

    def __init__(self, db=None):
        """
        Args:
            db : session SQLAlchemy async (optionnelle).
                 Si None ou FakeDB → mode dégradé avec données synthétiques.
        """
        self._db       = db
        self._weather  = WeatherProcessor()
        self._health   = HealthProcessor()
        self._geo      = GeoProcessor(db)

    # ─────────────────────────────────────────────────────────────
    # API publique — contrats avec les routers et le scheduler
    # ─────────────────────────────────────────────────────────────

    async def build_malaria_features(
        self,
        region_id: str,
        target_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Construit le vecteur de features pour MalariaPredictor.

        Retourne un dict avec EXACTEMENT les 26 clés de MALARIA_FEATURE_NAMES.
        Toute clé manquante est imputée à 0.0.

        Sources :
          - Météo rolling 7j/14j/30j (WeatherFetcher → cache Redis → DB)
          - Lags épidémiologiques S-1 à S-4 (DHIS2 → DB)
          - NDVI + zones humides (Sentinel Hub → PostGIS → estimation)
          - Données géographiques (regions_metadata.json)
          - Encodage temporel cyclique (sin/cos)
        """
        target_date = target_date or date.today()
        logger.debug("Build malaria features — région={} date={}", region_id, target_date)

        # ── 1. Features géographiques (jamais échoue) ─────────────
        geo = self._geo.get_geo_features(region_id)

        # ── 2. Features météo rolling ─────────────────────────────
        weather_rolling = await self._get_weather_rolling(region_id, target_date)

        # ── 3. Encodage temporel cyclique ─────────────────────────
        temporal = self._weather.encode_temporal(target_date)

        # ── 4. NDVI + zones humides ───────────────────────────────
        ndvi         = await self._geo.get_region_ndvi(region_id, target_date)
        zones_humides = await self._geo.get_zones_humides_pct(region_id)
        ndvi = ndvi if ndvi is not None else self._geo._estimate_ndvi(region_id)

        # ── 5. Lags épidémiologiques ──────────────────────────────
        epi_lags = await self._get_malaria_lags(region_id, target_date)

        # ── 6. Assemblage final — ordre STRICT MALARIA_FEATURE_NAMES ─
        raw = {
            # Climatiques
            "temperature_moy_c":       weather_rolling.get("temperature_moy_c", 24.0),
            "temperature_min_c":       weather_rolling.get("temperature_min_c", 18.0),
            "temperature_max_c":       weather_rolling.get("temperature_max_c", 30.0),
            "precipitations_7j_mm":    weather_rolling.get("precipitations_7j_mm", 0.0),
            "precipitations_14j_mm":   weather_rolling.get("precipitations_14j_mm", 0.0),
            "precipitations_30j_mm":   weather_rolling.get("precipitations_30j_mm", 0.0),
            "humidite_moy_pct":        weather_rolling.get("humidite_moy_pct", 70.0),
            "vent_kmh":                weather_rolling.get("vent_kmh", 10.0),
            "pression_hpa":            weather_rolling.get("pression_hpa", 1013.0),
            "couverture_nuageuse_pct": weather_rolling.get("couverture_nuageuse_pct", 50.0),
            # Environnementaux
            "ndvi":             ndvi,
            "zones_humides_pct": zones_humides,
            "altitude_m":       geo.get("altitude_m", 800.0),
            # Temporels
            "semaine_sin":    temporal["semaine_sin"],
            "semaine_cos":    temporal["semaine_cos"],
            "mois_sin":       temporal["mois_sin"],
            "mois_cos":       temporal["mois_cos"],
            "saison_encoded": temporal["saison_encoded"],
            # Épidémiologiques
            "cas_lag_1sem":            epi_lags.get("cas_lag_1sem", 0.0),
            "cas_lag_2sem":            epi_lags.get("cas_lag_2sem", 0.0),
            "cas_lag_3sem":            epi_lags.get("cas_lag_3sem", 0.0),
            "cas_lag_4sem":            epi_lags.get("cas_lag_4sem", 0.0),
            "taux_positivite_tdr_pct": epi_lags.get("taux_positivite_tdr_pct", 0.0),
            # Géographiques
            "latitude":           geo.get("latitude", -20.0),
            "longitude":          geo.get("longitude", 47.0),
            "endemicite_encoded": geo.get("endemicite_encoded", 1.0),
        }

        # Ajout region_id (non feature ML — utilisé pour logs et recommandations)
        raw["region_id"] = region_id

        # ── 7. Garantie de complétude ─────────────────────────────
        return self._ensure_complete(raw, MALARIA_FEATURE_NAMES, region_id, "malaria")

    async def build_nutrition_features(
        self,
        region_id: str,
        target_date: Optional[date] = None,
        malaria_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Construit le vecteur de features pour NutritionPredictor.

        Retourne un dict avec EXACTEMENT les 32 clés de NUTRITION_FEATURE_NAMES.

        Sources :
          - Lags GAM/SAM (enquêtes SMART → DB)
          - FCS, HDDS, rCSI (WFP VAM → DB)
          - Prix alimentaires normalisés (WFP → DB)
          - Score paludisme (feature croisée MalariaPredictor)
          - Soudure et vulnérabilité (profil régional)
          - Météo et géographie
        """
        target_date = target_date or date.today()
        logger.debug("Build nutrition features — région={} date={}", region_id, target_date)

        # ── 1. Géographie (jamais échoue) ─────────────────────────
        geo = self._geo.get_geo_features(region_id)

        # ── 2. Lags nutritionnels ─────────────────────────────────
        nut_lags = await self._get_nutrition_lags(region_id, target_date)

        # ── 3. Données FCS / prix (WFP) ───────────────────────────
        fcs_data  = await self._get_fcs_data(region_id)
        prix_data = await self._get_prix_data(region_id)

        # ── 4. Score paludisme (feature croisée) ──────────────────
        if malaria_score is None:
            malaria_score = await self._get_cached_malaria_score(region_id)

        # ── 5. Features soudure et vulnérabilité ──────────────────
        soudure = self._health.compute_soudure_features(region_id, target_date)
        vuln    = self._health.compute_vulnerability_index(region_id)

        # ── 6. Météo (précipitations 30j, temp, ndvi) ─────────────
        weather_rolling = await self._get_weather_rolling(region_id, target_date)
        ndvi = await self._geo.get_region_ndvi(region_id, target_date)
        ndvi = ndvi if ndvi is not None else self._geo._estimate_ndvi(region_id)

        # ── 7. Encodage temporel ──────────────────────────────────
        temporal = self._weather.encode_temporal(target_date)

        # ── 8. Normalisation prix ─────────────────────────────────
        prix_norm = self._health.normalize_prix(prix_data)

        # ── 9. Assemblage final — ordre STRICT NUTRITION_FEATURE_NAMES ─
        raw = {
            # Historique nutritionnel
            "gam_lag_1m":        nut_lags.get("gam_lag_1m", 0.0),
            "gam_lag_3m":        nut_lags.get("gam_lag_3m", 0.0),
            "gam_lag_6m":        nut_lags.get("gam_lag_6m", 0.0),
            "sam_lag_1m":        nut_lags.get("sam_lag_1m", 0.0),
            "variation_gam_3m":  nut_lags.get("variation_gam_3m", 0.0),
            # Sécurité alimentaire
            "score_fcs":  fcs_data.get("score_fcs", 35.0),
            "hdds":       fcs_data.get("hdds", 5.0),
            "rcsi":       fcs_data.get("rcsi", 0.0),
            # Prix normalisés
            "prix_riz_normalise":    prix_norm.get("prix_riz_normalise", 1.0),
            "prix_manioc_normalise": prix_norm.get("prix_manioc_normalise", 1.0),
            "variation_prix_pct_1m": float(prix_data.get("variation_prix_pct_1m") or 0.0),
            # Disponibilité
            "dispo_cereales":          float(fcs_data.get("disponibilite_cereales", 2.0)),
            "dispo_legumineuses":      float(fcs_data.get("disponibilite_legumineuses", 2.0)),
            "dispo_proteines_animales":float(fcs_data.get("disponibilite_proteines_animales", 1.0)),
            "dispo_legumes":           float(fcs_data.get("disponibilite_legumes", 2.0)),
            "dispo_fruits":            float(fcs_data.get("disponibilite_fruits", 2.0)),
            # Feature croisée paludisme
            "score_paludisme": float(malaria_score),
            # Climatique
            "precipitations_30j_mm": weather_rolling.get("precipitations_30j_mm", 0.0),
            "temperature_moy_c":     weather_rolling.get("temperature_moy_c", 24.0),
            "ndvi":                  ndvi,
            # Socio-économique
            "indice_vulnerabilite":          vuln,
            "en_periode_soudure":            soudure["en_periode_soudure"],
            "semaines_avant_soudure_norm":   soudure["semaines_avant_soudure_norm"],
            # Temporel
            "mois_sin":       temporal["mois_sin"],
            "mois_cos":       temporal["mois_cos"],
            "saison_encoded": temporal["saison_encoded"],
            # Géographique
            "latitude":               geo.get("latitude", -20.0),
            "longitude":              geo.get("longitude", 47.0),
            "altitude_m":             geo.get("altitude_m", 800.0),
            "zone_climatique_encoded":geo.get("zone_climatique_encoded", 2.0),
            # Démographique
            "densite_population_norm": geo.get("densite_population_norm", 0.0),
            "pct_enfants_5ans":        geo.get("pct_enfants_5ans", 0.17),
        }

        raw["region_id"] = region_id

        # ── 10. Garantie complétude ───────────────────────────────
        return self._ensure_complete(raw, NUTRITION_FEATURE_NAMES, region_id, "nutrition")

    # ─────────────────────────────────────────────────────────────
    # Méthode utilitaire — construction batch pour entraînement ML
    # ─────────────────────────────────────────────────────────────

    async def build_training_dataset(
        self,
        region_ids: List[str],
        date_debut: date,
        date_fin: date,
        modele: str = "malaria",
    ) -> tuple:
        """
        Construit le dataset d'entraînement pour tous les scripts ML.

        Returns:
            (X: np.ndarray, y: np.ndarray, feature_names: list, dates: list)
        """
        import pandas as pd

        rows_X = []
        rows_y = []
        rows_meta = []

        date_courante = date_debut
        while date_courante <= date_fin:
            for region_id in region_ids:
                try:
                    if modele == "malaria":
                        features = await self.build_malaria_features(
                            region_id, date_courante
                        )
                        feat_names = MALARIA_FEATURE_NAMES
                    else:
                        features = await self.build_nutrition_features(
                            region_id, date_courante
                        )
                        feat_names = NUTRITION_FEATURE_NAMES

                    X_row = [float(features.get(n, 0.0)) for n in feat_names]
                    y_row = await self._get_label(region_id, date_courante, modele)

                    if y_row is not None:
                        rows_X.append(X_row)
                        rows_y.append(y_row)
                        rows_meta.append({
                            "region_id": region_id,
                            "date": str(date_courante),
                        })

                except Exception as exc:
                    logger.debug(
                        "Skip training sample {} {} : {}", region_id, date_courante, exc
                    )

            date_courante += timedelta(weeks=1)  # Pas hebdomadaire

        if not rows_X:
            logger.warning("Dataset {} vide — 0 samples", modele)
            return np.array([]), np.array([]), feat_names, []

        X = np.array(rows_X, dtype=np.float32)
        y = np.array(rows_y, dtype=np.float32)
        logger.info("Dataset {} : {} samples × {} features", modele, X.shape[0], X.shape[1])
        return X, y, feat_names, rows_meta

    # ─────────────────────────────────────────────────────────────
    # Helpers privés — sources de données
    # ─────────────────────────────────────────────────────────────

    async def _get_weather_rolling(
        self, region_id: str, target_date: date
    ) -> Dict[str, float]:
        """
        Récupère les données météo rolling depuis :
          1. Cache Redis (si disponible)
          2. DB PostgreSQL (weather_observations)
          3. NASA POWER API (fallback externe)
          4. Valeurs par défaut (dernier recours)
        """
        # Tentative 1 : DB
        historical = await self._fetch_weather_from_db(region_id, target_date)

        if historical:
            cleaned = self._weather.clean_series(historical)
            return self._weather.compute_rolling_features(cleaned, windows=[7, 14, 30])

        # Tentative 2 : NASA POWER API
        try:
            from src.data_collection.weather_fetcher import WeatherFetcher
            fetcher = WeatherFetcher()
            date_fin   = target_date
            date_debut = target_date - timedelta(days=35)
            historical = await fetcher.get_history_nasa(
                region_id, date_debut, date_fin
            )
            await fetcher.close()

            if historical:
                cleaned = self._weather.clean_series(historical)
                return self._weather.compute_rolling_features(cleaned, windows=[7, 14, 30])
        except Exception as exc:
            logger.debug("NASA POWER météo {} : {}", region_id, exc)

        # Fallback : valeurs par défaut climatologiques
        logger.debug("Fallback météo défaut pour {}", region_id)
        return self._weather._default_rolling_features([7, 14, 30])

    async def _get_malaria_lags(
        self, region_id: str, target_date: date
    ) -> Dict[str, float]:
        """
        Récupère les lags épidémiologiques paludisme :
          1. DB (malaria_cases) → 5 dernières semaines
          2. DHIS2 (fallback externe)
          3. Valeurs nulles (dernier recours)
        """
        # Tentative DB
        records = await self._fetch_malaria_from_db(region_id, target_date, weeks=6)

        if records:
            cleaned = self._health.clean_malaria_records(records)
            return self._health.compute_malaria_lags(cleaned, n_lags=4)

        # Tentative DHIS2
        try:
            from src.data_collection.malaria_fetcher import MalariaFetcher
            fetcher = MalariaFetcher()
            date_fin   = target_date
            date_debut = target_date - timedelta(weeks=5)
            records = await fetcher.get_cas_dhis2(region_id, date_debut, date_fin)
            await fetcher.close()
            if records:
                return self._health.compute_malaria_lags(records, n_lags=4)
        except Exception as exc:
            logger.debug("DHIS2 lags {} : {}", region_id, exc)

        # Fallback zéros
        return {f"cas_lag_{i}sem": 0.0 for i in range(1, 5)} | \
               {"taux_positivite_tdr_pct": 0.0}

    async def _get_nutrition_lags(
        self, region_id: str, target_date: date
    ) -> Dict[str, float]:
        """Récupère les lags GAM/SAM depuis DB → estimation."""
        records = await self._fetch_nutrition_from_db(region_id, months=7)

        if records:
            return self._health.compute_nutrition_lags(records, [1, 3, 6])

        # Estimation depuis profil régional
        from src.data_collection.nutrition_fetcher import NutritionFetcher
        try:
            fetcher = NutritionFetcher()
            statut  = fetcher._estimer_statut_nutritionnel(region_id)
            gam     = statut.get("gam_pct", 5.0)
            sam     = statut.get("sam_pct", 1.5)
            return {
                "gam_lag_1m": gam, "gam_lag_3m": gam * 0.95,
                "gam_lag_6m": gam * 0.90, "sam_lag_1m": sam,
                "variation_gam_3m": gam * 0.05,
            }
        except Exception:
            return {
                "gam_lag_1m": 5.0, "gam_lag_3m": 5.0,
                "gam_lag_6m": 5.0, "sam_lag_1m": 1.5,
                "variation_gam_3m": 0.0,
            }

    async def _get_fcs_data(self, region_id: str) -> Dict[str, Any]:
        """Récupère FCS, HDDS, rCSI, disponibilités depuis DB → WFP → profil."""
        db_data = await self._fetch_fcs_from_db(region_id)
        if db_data:
            return db_data

        try:
            from src.data_collection.nutrition_fetcher import NutritionFetcher
            fetcher = NutritionFetcher()
            dispo   = await fetcher.get_disponibilite_complete(region_id)
            await fetcher.close()
            return dispo
        except Exception as exc:
            logger.debug("FCS fallback {} : {}", region_id, exc)
            # Valeurs par défaut
            from src.data_collection.nutrition_fetcher import (
                REGION_FOOD_PROFILE, DEFAULT_FOOD_PROFILE
            )
            profile = REGION_FOOD_PROFILE.get(region_id, DEFAULT_FOOD_PROFILE)
            vuln    = profile.get("indice_vulnerabilite", 0.5)
            mois    = date.today().month
            en_soudure = mois in profile.get("mois_soudure", [11, 12])
            fcs = max(10, 50 * (1 - vuln) * (0.65 if en_soudure else 1.0))
            return {
                "score_fcs": round(fcs, 1), "hdds": round(6 * (1 - vuln * 0.5), 1),
                "rcsi": int(vuln * 20),
                "disponibilite_cereales": 2, "disponibilite_legumineuses": 2,
                "disponibilite_proteines_animales": 1,
                "disponibilite_legumes": 2, "disponibilite_fruits": 2,
            }

    async def _get_prix_data(self, region_id: str) -> Dict[str, Any]:
        """Récupère les prix alimentaires depuis DB → WFP → synthétique."""
        db_data = await self._fetch_prix_from_db(region_id)
        if db_data:
            return db_data

        try:
            from src.data_collection.nutrition_fetcher import NutritionFetcher
            fetcher = NutritionFetcher()
            prix    = await fetcher.get_prix_denrees(region_id)
            await fetcher.close()
            return prix
        except Exception as exc:
            logger.debug("Prix fallback {} : {}", region_id, exc)
            return {
                "prix_riz_kg": 1800.0, "prix_manioc_kg": 400.0,
                "variation_prix_pct_1m": 0.0,
            }

    async def _get_cached_malaria_score(self, region_id: str) -> float:
        """
        Récupère le score paludisme depuis le cache Redis pour la feature croisée.
        Retourne 0.3 (valeur de base) si non disponible.
        """
        try:
            import redis
            import json as _json
            from config.settings import settings
            r = redis.Redis.from_url(settings.redis.url, decode_responses=True)
            key = f"unicef:mdg:malaria:risque:{region_id}:14"
            cached = r.get(key)
            if cached:
                data = _json.loads(cached)
                return float(data.get("score_risque", 0.3))
        except Exception:
            pass

        # Estimation simple depuis l'endémicité
        geo = self._geo.get_geo_features(region_id)
        endemicite = geo.get("endemicite_encoded", 1.0)
        return round(float(endemicite / 3.0), 3)

    # ─────────────────────────────────────────────────────────────
    # Requêtes DB (avec fallback gracieux si DB indisponible)
    # ─────────────────────────────────────────────────────────────

    async def _fetch_weather_from_db(
        self, region_id: str, target_date: date, days: int = 35
    ) -> List[Dict]:
        """Charge les données météo depuis weather_observations."""
        if not self._is_real_db():
            return []
        try:
            from sqlalchemy import text
            date_debut = target_date - timedelta(days=days)
            result = await self._db.execute(
                text("""
                    SELECT
                        date_trunc('day', horodatage)::date AS date,
                        AVG(temperature_c) AS temperature_moy_c,
                        MIN(temperature_c) AS temperature_min_c,
                        MAX(temperature_c) AS temperature_max_c,
                        SUM(precipitations_mm) AS precipitations_mm,
                        AVG(humidite_pct) AS humidite_moy_pct,
                        AVG(vent_kmh) AS vent_kmh,
                        AVG(pression_hpa) AS pression_hpa,
                        AVG(couverture_nuageuse_pct) AS couverture_nuageuse_pct
                    FROM weather_observations
                    WHERE region_id = :region_id
                      AND horodatage BETWEEN :date_debut AND :date_fin
                    GROUP BY 1
                    ORDER BY 1
                """),
                {
                    "region_id": region_id,
                    "date_debut": str(date_debut),
                    "date_fin":   str(target_date),
                }
            )
            return [dict(row._mapping) for row in result.fetchall()]
        except Exception as exc:
            logger.debug("DB weather {} : {}", region_id, exc)
            return []

    async def _fetch_malaria_from_db(
        self, region_id: str, target_date: date, weeks: int = 6
    ) -> List[Dict]:
        """Charge les dernières semaines de cas depuis malaria_cases."""
        if not self._is_real_db():
            return []
        try:
            from sqlalchemy import text
            date_debut = target_date - timedelta(weeks=weeks)
            result = await self._db.execute(
                text("""
                    SELECT
                        region_id, annee, semaine_epidemio, date_rapport,
                        cas_confirmes, cas_confirmes_mixte, deces, hospitalisations,
                        taux_incidence_pour_mille, taux_positivite_tdr_pct,
                        population_a_risque, source, fiabilite_donnees
                    FROM malaria_cases
                    WHERE region_id = :region_id
                      AND date_rapport BETWEEN :date_debut AND :date_fin
                    ORDER BY annee, semaine_epidemio
                """),
                {"region_id": region_id, "date_debut": str(date_debut),
                 "date_fin": str(target_date)}
            )
            return [dict(row._mapping) for row in result.fetchall()]
        except Exception as exc:
            logger.debug("DB malaria {} : {}", region_id, exc)
            return []

    async def _fetch_nutrition_from_db(
        self, region_id: str, months: int = 7
    ) -> List[Dict]:
        """Charge l'historique GAM depuis nutrition_status."""
        if not self._is_real_db():
            return []
        try:
            from sqlalchemy import text
            date_debut = date.today() - timedelta(days=months * 30)
            result = await self._db.execute(
                text("""
                    SELECT
                        region_id, date_observation, gam_pct, sam_pct, mam_pct,
                        score_fcs, hdds, rcsi, source
                    FROM nutrition_status
                    WHERE region_id = :region_id
                      AND date_observation >= :date_debut
                    ORDER BY date_observation
                """),
                {"region_id": region_id, "date_debut": str(date_debut)}
            )
            return [dict(row._mapping) for row in result.fetchall()]
        except Exception as exc:
            logger.debug("DB nutrition {} : {}", region_id, exc)
            return []

    async def _fetch_fcs_from_db(self, region_id: str) -> Optional[Dict]:
        """Charge les dernières données FCS/HDDS/rCSI depuis la DB."""
        if not self._is_real_db():
            return None
        try:
            from sqlalchemy import text
            result = await self._db.execute(
                text("""
                    SELECT score_fcs, hdds, rcsi,
                           disponibilite_cereales, disponibilite_legumineuses,
                           disponibilite_proteines_animales,
                           disponibilite_legumes, disponibilite_fruits
                    FROM nutrition_food_security
                    WHERE region_id = :region_id
                    ORDER BY date_observation DESC
                    LIMIT 1
                """),
                {"region_id": region_id}
            )
            row = result.fetchone()
            return dict(row._mapping) if row else None
        except Exception as exc:
            logger.debug("DB FCS {} : {}", region_id, exc)
            return None

    async def _fetch_prix_from_db(self, region_id: str) -> Optional[Dict]:
        """Charge les derniers prix alimentaires depuis la DB."""
        if not self._is_real_db():
            return None
        try:
            from sqlalchemy import text
            result = await self._db.execute(
                text("""
                    SELECT prix_riz_kg, prix_manioc_kg, prix_mais_kg,
                           prix_haricots_kg, prix_huile_litre,
                           variation_prix_pct_1m
                    FROM food_prices
                    WHERE region_id = :region_id
                    ORDER BY date_observation DESC
                    LIMIT 1
                """),
                {"region_id": region_id}
            )
            row = result.fetchone()
            return dict(row._mapping) if row else None
        except Exception as exc:
            logger.debug("DB prix {} : {}", region_id, exc)
            return None

    async def _get_label(
        self, region_id: str, target_date: date, modele: str
    ) -> Optional[float]:
        """
        Récupère le label de supervision pour l'entraînement ML.
        Malaria : taux incidence normalisé [0,1]
        Nutrition : score GAM normalisé [0,1]
        """
        if modele == "malaria":
            records = await self._fetch_malaria_from_db(region_id, target_date, weeks=1)
            if records:
                last = records[-1]
                incidence = float(last.get("taux_incidence_pour_mille", 0))
                return float(np.clip(incidence / 10.0, 0, 1))  # /10 → normalisé
        else:
            records = await self._fetch_nutrition_from_db(region_id, months=1)
            if records:
                last = records[-1]
                gam = float(last.get("gam_pct", 0))
                return float(np.clip(gam / 20.0, 0, 1))  # /20 → normalisé
        return None

    # ─────────────────────────────────────────────────────────────
    # Garantie de complétude — contrat avec les modèles ML
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_complete(
        features: Dict[str, Any],
        expected_names: List[str],
        region_id: str,
        modele: str,
    ) -> Dict[str, Any]:
        """
        Garantit que le dict contient toutes les clés attendues par le modèle.
        Les clés manquantes sont imputées à 0.0 avec un warning.
        """
        missing = [n for n in expected_names if n not in features]
        if missing:
            logger.warning(
                "Features manquantes pour modele={} region={}: {} → imputation 0.0",
                modele, region_id, missing
            )
            for name in missing:
                features[name] = 0.0

        # Garantit que toutes les valeurs sont float (pas None)
        for name in expected_names:
            val = features.get(name)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                features[name] = 0.0
            else:
                features[name] = float(val)

        return features

    def _is_real_db(self) -> bool:
        """Détecte si la session DB est une vraie session SQLAlchemy."""
        if self._db is None:
            return False
        db_type = type(self._db).__name__
        return "Session" in db_type or "AsyncSession" in db_type