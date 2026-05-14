"""
Utilitaires géospatiaux pour Madagascar.

Fonctions exportées :
  haversine_distance(lat1, lon1, lat2, lon2)   → distance en km
  get_region_from_coords(lat, lon)              → code région MDG-xxx
  coords_in_madagascar(lat, lon)               → bool
  get_region_metadata(region_code)             → dict (centre, bbox, altitude_moy…)
  get_neighboring_regions(region_code)         → list[str]
  build_region_centroid_map()                  → dict code → (lat, lon)
  classify_zone_altitude(altitude_m)           → ZoneAltitude
  get_climate_zone(region_code)               → ZoneClimatique
  distance_to_nearest_wetland(lat, lon)        → float (km) — approximation
  encode_cyclic(value, max_value)              → tuple(sin, cos)
  region_risk_color(score)                     → str (hex couleur alerte)
  aggregate_region_scores(scores_dict)         → dict avec stats

Dépendances :
  - math (stdlib uniquement — pas de shapely/geopandas en production légère)
  - La précision géo est suffisante pour des granularités régionales (≈ 10 km)
  - Pour des analyses fines (fokontany), utiliser geo_processor.py (PostGIS)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.utils.constants import COULEURS_ALERTE, REGIONS_MADAGASCAR
from src.utils.logger import get_logger

log = get_logger("geo_helpers")


# ─────────────────────────────────────────────────────────────────
# Enums géographiques
# ─────────────────────────────────────────────────────────────────

class ZoneAltitude(str, Enum):
    """
    Classification altitudinale de Madagascar.
    Critique pour la transmission paludisme (altitude > 1500 m → très faible risque).
    """
    COTE_BASSE   = "côte_basse"    # 0–200 m
    TRANSITION   = "transition"    # 200–800 m
    HAUTS_PLAT   = "hauts_plateaux"  # 800–1500 m
    MONTAGNE     = "montagne"      # > 1500 m


class ZoneClimatique(str, Enum):
    """
    Zones climatiques selon la géographie de Madagascar.
    Détermine les patterns de précipitation et de saisonnalité.
    """
    COTE_EST      = "côte_est"       # Alizés, pluies toute l'année
    COTE_OUEST    = "côte_ouest"     # Saison sèche marquée
    NORD          = "nord"           # Tropical humide
    SUD_ARIDE     = "sud_aride"      # Semi-aride / aride (Androy, Anosy)
    HAUTS_PLAT    = "hauts_plateaux" # Tempéré, gel possible en hiver
    CENTRE_OUEST  = "centre_ouest"   # Transition savane


# ─────────────────────────────────────────────────────────────────
# Metadata des 22 régions Madagascar
# ─────────────────────────────────────────────────────────────────

@dataclass
class RegionMetadata:
    """Données de référence d'une région administrative de Madagascar."""
    code: str
    nom_fr: str
    nom_mg: str                              # Nom en malgache
    centroid: Tuple[float, float]            # (latitude, longitude)
    bbox: Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)
    altitude_moyenne_m: float
    zone_altitude: ZoneAltitude
    zone_climatique: ZoneClimatique
    population_estimee: int                  # Estimation 2024
    superficie_km2: float
    voisins: List[str] = field(default_factory=list)  # Codes régions limitrophes
    # Coordonnées approximatives des principales zones humides (mares, rizières)
    zones_humides_coords: List[Tuple[float, float]] = field(default_factory=list)


# Dictionnaire principal des 22 régions
# Sources : INSTAT Madagascar, GADM, WorldPop 2023
_REGIONS_DB: Dict[str, RegionMetadata] = {

    "MDG-ANA": RegionMetadata(
        code="MDG-ANA", nom_fr="Analamanga", nom_mg="Analamanga",
        centroid=(-18.9137, 47.5361),
        bbox=(-19.80, -17.80, 46.80, 48.10),
        altitude_moyenne_m=1350, zone_altitude=ZoneAltitude.HAUTS_PLAT,
        zone_climatique=ZoneClimatique.HAUTS_PLAT,
        population_estimee=4_200_000, superficie_km2=16_911,
        voisins=["MDG-VAK", "MDG-ITM", "MDG-BMT", "MDG-ALA", "MDG-ATI"],
        zones_humides_coords=[(-18.96, 47.45), (-18.75, 47.60)],
    ),

    "MDG-VAK": RegionMetadata(
        code="MDG-VAK", nom_fr="Vakinankaratra", nom_mg="Vakinankaratra",
        centroid=(-19.8667, 46.9500),
        bbox=(-21.00, -18.80, 46.20, 47.80),
        altitude_moyenne_m=1450, zone_altitude=ZoneAltitude.HAUTS_PLAT,
        zone_climatique=ZoneClimatique.HAUTS_PLAT,
        population_estimee=2_100_000, superficie_km2=19_117,
        voisins=["MDG-ANA", "MDG-ITM", "MDG-MAT", "MDG-ATI", "MDG-MEN2"],
    ),

    "MDG-ITM": RegionMetadata(
        code="MDG-ITM", nom_fr="Itasy", nom_mg="Itasy",
        centroid=(-19.1167, 46.7333),
        bbox=(-19.80, -18.40, 46.20, 47.20),
        altitude_moyenne_m=1250, zone_altitude=ZoneAltitude.HAUTS_PLAT,
        zone_climatique=ZoneClimatique.HAUTS_PLAT,
        population_estimee=780_000, superficie_km2=6_993,
        voisins=["MDG-ANA", "MDG-VAK", "MDG-BMT"],
        zones_humides_coords=[(-19.02, 46.78)],  # Lac Itasy
    ),

    "MDG-BMT": RegionMetadata(
        code="MDG-BMT", nom_fr="Bongolava", nom_mg="Bongolava",
        centroid=(-18.6500, 45.7500),
        bbox=(-19.80, -17.20, 44.80, 46.80),
        altitude_moyenne_m=800, zone_altitude=ZoneAltitude.TRANSITION,
        zone_climatique=ZoneClimatique.CENTRE_OUEST,
        population_estimee=440_000, superficie_km2=16_688,
        voisins=["MDG-ANA", "MDG-ITM", "MDG-SOF", "MDG-BOE", "MDG-MEN"],
    ),

    "MDG-MAT": RegionMetadata(
        code="MDG-MAT", nom_fr="Matsiatra Ambony", nom_mg="Matsiatra Ambony",
        centroid=(-21.4500, 47.0833),
        bbox=(-22.40, -20.40, 46.20, 47.80),
        altitude_moyenne_m=1200, zone_altitude=ZoneAltitude.HAUTS_PLAT,
        zone_climatique=ZoneClimatique.HAUTS_PLAT,
        population_estimee=1_200_000, superficie_km2=20_460,
        voisins=["MDG-VAK", "MDG-ATI", "MDG-VAT", "MDG-IHO", "MDG-ASO"],
    ),

    "MDG-ATI": RegionMetadata(
        code="MDG-ATI", nom_fr="Amoron'i Mania", nom_mg="Amoron'i Mania",
        centroid=(-20.6000, 47.3500),
        bbox=(-21.40, -19.80, 46.80, 48.20),
        altitude_moyenne_m=1100, zone_altitude=ZoneAltitude.HAUTS_PLAT,
        zone_climatique=ZoneClimatique.HAUTS_PLAT,
        population_estimee=830_000, superficie_km2=16_896,
        voisins=["MDG-ANA", "MDG-VAK", "MDG-MAT", "MDG-VAT", "MDG-ATS"],
    ),

    "MDG-VAT": RegionMetadata(
        code="MDG-VAT", nom_fr="Vatovavy", nom_mg="Vatovavy",
        centroid=(-21.4667, 47.9833),
        bbox=(-22.20, -20.60, 47.20, 48.60),
        altitude_moyenne_m=400, zone_altitude=ZoneAltitude.TRANSITION,
        zone_climatique=ZoneClimatique.COTE_EST,
        population_estimee=1_050_000, superficie_km2=12_473,
        voisins=["MDG-ATI", "MDG-MAT", "MDG-FIT", "MDG-ATS"],
    ),

    "MDG-FIT": RegionMetadata(
        code="MDG-FIT", nom_fr="Fitovinany", nom_mg="Fitovinany",
        centroid=(-22.3000, 47.9500),
        bbox=(-23.00, -21.60, 47.20, 48.60),
        altitude_moyenne_m=300, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.COTE_EST,
        population_estimee=900_000, superficie_km2=10_687,
        voisins=["MDG-VAT", "MDG-MAT", "MDG-ANO", "MDG-IHO"],
    ),

    "MDG-ANO": RegionMetadata(
        code="MDG-ANO", nom_fr="Atsimo-Atsinanana", nom_mg="Atsimo-Atsinanana",
        centroid=(-23.3833, 47.6000),
        bbox=(-24.40, -22.40, 46.80, 48.40),
        altitude_moyenne_m=250, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.COTE_EST,
        population_estimee=830_000, superficie_km2=18_863,
        voisins=["MDG-FIT", "MDG-IHO", "MDG-AAN"],
    ),

    "MDG-ATS": RegionMetadata(
        code="MDG-ATS", nom_fr="Atsinanana", nom_mg="Atsinanana",
        centroid=(-18.1500, 49.4000),
        bbox=(-20.00, -16.40, 48.60, 50.20),
        altitude_moyenne_m=200, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.COTE_EST,
        population_estimee=1_200_000, superficie_km2=21_934,
        voisins=["MDG-ANA", "MDG-ATI", "MDG-VAT", "MDG-ALA", "MDG-ANA2"],
    ),

    "MDG-ANA2": RegionMetadata(
        code="MDG-ANA2", nom_fr="Analanjirofo", nom_mg="Analanjirofo",
        centroid=(-16.1667, 49.7667),
        bbox=(-17.80, -14.60, 49.00, 50.60),
        altitude_moyenne_m=300, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.COTE_EST,
        population_estimee=1_000_000, superficie_km2=21_930,
        voisins=["MDG-ATS", "MDG-ALA", "MDG-SAV", "MDG-DIA"],
    ),

    "MDG-ALA": RegionMetadata(
        code="MDG-ALA", nom_fr="Alaotra-Mangoro", nom_mg="Alaotra-Mangoro",
        centroid=(-17.9833, 48.4167),
        bbox=(-19.60, -16.40, 47.60, 49.40),
        altitude_moyenne_m=750, zone_altitude=ZoneAltitude.TRANSITION,
        zone_climatique=ZoneClimatique.HAUTS_PLAT,
        population_estimee=1_180_000, superficie_km2=31_948,
        voisins=["MDG-ANA", "MDG-ATS", "MDG-ANA2", "MDG-SOF", "MDG-BOE"],
        zones_humides_coords=[(-17.50, 48.50)],  # Lac Alaotra
    ),

    "MDG-BOE": RegionMetadata(
        code="MDG-BOE", nom_fr="Boeny", nom_mg="Boeny",
        centroid=(-16.1000, 46.3500),
        bbox=(-17.60, -14.60, 44.80, 48.00),
        altitude_moyenne_m=200, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.COTE_OUEST,
        population_estimee=840_000, superficie_km2=31_046,
        voisins=["MDG-BMT", "MDG-ALA", "MDG-SOF", "MDG-MEN"],
    ),

    "MDG-SOF": RegionMetadata(
        code="MDG-SOF", nom_fr="Sofia", nom_mg="Sofia",
        centroid=(-14.8000, 47.6000),
        bbox=(-16.40, -13.20, 46.60, 49.20),
        altitude_moyenne_m=400, zone_altitude=ZoneAltitude.TRANSITION,
        zone_climatique=ZoneClimatique.NORD,
        population_estimee=1_200_000, superficie_km2=50_100,
        voisins=["MDG-BMT", "MDG-BOE", "MDG-ALA", "MDG-ANA2", "MDG-DIA"],
    ),

    "MDG-MEN": RegionMetadata(
        code="MDG-MEN", nom_fr="Melaky", nom_mg="Melaky",
        centroid=(-16.7833, 44.6833),
        bbox=(-18.60, -15.00, 43.40, 46.00),
        altitude_moyenne_m=150, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.COTE_OUEST,
        population_estimee=350_000, superficie_km2=38_852,
        voisins=["MDG-BMT", "MDG-BOE", "MDG-MEN2"],
    ),

    "MDG-MEN2": RegionMetadata(
        code="MDG-MEN2", nom_fr="Menabe", nom_mg="Menabe",
        centroid=(-19.8333, 44.9167),
        bbox=(-21.60, -18.00, 43.80, 46.40),
        altitude_moyenne_m=180, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.COTE_OUEST,
        population_estimee=650_000, superficie_km2=48_967,
        voisins=["MDG-VAK", "MDG-MAT", "MDG-MEN", "MDG-ASO", "MDG-IHO"],
    ),

    "MDG-DIA": RegionMetadata(
        code="MDG-DIA", nom_fr="Diana", nom_mg="Diana",
        centroid=(-12.3500, 49.2833),
        bbox=(-13.60, -11.20, 48.20, 50.40),
        altitude_moyenne_m=350, zone_altitude=ZoneAltitude.TRANSITION,
        zone_climatique=ZoneClimatique.NORD,
        population_estimee=760_000, superficie_km2=13_237,
        voisins=["MDG-SOF", "MDG-ANA2", "MDG-SAV"],
    ),

    "MDG-SAV": RegionMetadata(
        code="MDG-SAV", nom_fr="Sava", nom_mg="Sava",
        centroid=(-14.2667, 50.1500),
        bbox=(-15.60, -12.80, 49.20, 50.80),
        altitude_moyenne_m=300, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.COTE_EST,
        population_estimee=1_100_000, superficie_km2=25_518,
        voisins=["MDG-DIA", "MDG-SOF", "MDG-ANA2"],
    ),

    "MDG-IHO": RegionMetadata(
        code="MDG-IHO", nom_fr="Ihorombe", nom_mg="Ihorombe",
        centroid=(-22.4500, 46.1000),
        bbox=(-23.60, -21.40, 44.80, 47.60),
        altitude_moyenne_m=900, zone_altitude=ZoneAltitude.TRANSITION,
        zone_climatique=ZoneClimatique.HAUTS_PLAT,
        population_estimee=300_000, superficie_km2=26_436,
        voisins=["MDG-MAT", "MDG-FIT", "MDG-ANO", "MDG-MEN2", "MDG-ASO", "MDG-AAN"],
    ),

    "MDG-ASO": RegionMetadata(
        code="MDG-ASO", nom_fr="Atsimo-Andrefana", nom_mg="Atsimo-Andrefana",
        centroid=(-23.3500, 44.1667),
        bbox=(-25.60, -21.00, 42.40, 46.20),
        altitude_moyenne_m=200, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.SUD_ARIDE,
        population_estimee=1_100_000, superficie_km2=66_236,
        voisins=["MDG-MAT", "MDG-MEN2", "MDG-IHO", "MDG-AAN", "MDG_AND"],
    ),

    "MDG_AND": RegionMetadata(
        code="MDG_AND", nom_fr="Androy", nom_mg="Androy",
        centroid=(-25.0333, 45.4167),
        bbox=(-25.60, -24.20, 44.00, 47.00),
        altitude_moyenne_m=150, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.SUD_ARIDE,
        population_estimee=830_000, superficie_km2=19_317,
        voisins=["MDG-ASO", "MDG-AAN"],
    ),

    "MDG-AAN": RegionMetadata(
        code="MDG-AAN", nom_fr="Anosy", nom_mg="Anosy",
        centroid=(-24.7500, 46.8333),
        bbox=(-25.60, -23.60, 45.60, 48.20),
        altitude_moyenne_m=300, zone_altitude=ZoneAltitude.COTE_BASSE,
        zone_climatique=ZoneClimatique.SUD_ARIDE,
        population_estimee=680_000, superficie_km2=25_731,
        voisins=["MDG-IHO", "MDG-ANO", "MDG-ASO", "MDG_AND"],
    ),
}

# ─────────────────────────────────────────────────────────────────
# Bounding box Madagascar (pour validation rapide des coordonnées)
# ─────────────────────────────────────────────────────────────────

_MDG_LAT_MIN = -25.61
_MDG_LAT_MAX = -11.95
_MDG_LON_MIN =  43.22
_MDG_LON_MAX =  50.48

# Rayon terrestre moyen (km) — utilisé par haversine
_EARTH_RADIUS_KM = 6_371.0


# ═════════════════════════════════════════════════════════════════
# 1. CALCULS DE DISTANCE
# ═════════════════════════════════════════════════════════════════

def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calcule la distance orthodromique (great-circle) entre deux points GPS.

    Args:
        lat1, lon1 : Coordonnées du point A (degrés décimaux)
        lat2, lon2 : Coordonnées du point B (degrés décimaux)

    Returns:
        Distance en kilomètres (float).

    Précision : < 0.5% pour distances < 1 000 km (suffisant pour Madagascar).

    Example:
        >>> haversine_distance(-18.91, 47.53, -12.35, 49.28)
        749.2  # km Antananarivo → Antsiranana
    """
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)

    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return round(_EARTH_RADIUS_KM * c, 2)


def distance_between_regions(code_a: str, code_b: str) -> float:
    """
    Distance entre les centroïdes de deux régions (km).

    Returns:
        float distance ou -1.0 si un code est invalide.
    """
    meta_a = _REGIONS_DB.get(code_a)
    meta_b = _REGIONS_DB.get(code_b)

    if not meta_a or not meta_b:
        log.warning(
            "distance_between_regions: code inconnu — {} ou {}", code_a, code_b
        )
        return -1.0

    return haversine_distance(
        meta_a.centroid[0], meta_a.centroid[1],
        meta_b.centroid[0], meta_b.centroid[1],
    )


# ═════════════════════════════════════════════════════════════════
# 2. RÉSOLUTION GÉOGRAPHIQUE
# ═════════════════════════════════════════════════════════════════

def coords_in_madagascar(lat: float, lon: float) -> bool:
    """
    Vérifie si des coordonnées sont à l'intérieur du bounding box de Madagascar.

    Note :
        Vérification rapide par bounding box (pas de polygone précis).
        Pour une précision point-dans-polygone, utiliser PostGIS via geo_processor.py.

    Returns:
        True si les coordonnées semblent être dans Madagascar.
    """
    return (
        _MDG_LAT_MIN <= lat <= _MDG_LAT_MAX
        and _MDG_LON_MIN <= lon <= _MDG_LON_MAX
    )


def get_region_from_coords(lat: float, lon: float) -> Optional[str]:
    """
    Identifie la région la plus probable à partir de coordonnées GPS.

    Algorithme :
        1. Vérification que les coords sont dans Madagascar
        2. Vérification si dans la bbox d'une région
        3. Si ambiguïté (zones de chevauchement bbox), retourne la région
           dont le centroïde est le plus proche.

    Args:
        lat, lon : Coordonnées GPS (degrés décimaux)

    Returns:
        Code région MDG-xxx ou None si hors Madagascar.
    """
    if not coords_in_madagascar(lat, lon):
        log.warning(
            "Coordonnées hors Madagascar : lat={} lon={}", lat, lon
        )
        return None

    candidates: List[Tuple[str, float]] = []

    for code, meta in _REGIONS_DB.items():
        lat_min, lat_max, lon_min, lon_max = meta.bbox
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            dist = haversine_distance(lat, lon, meta.centroid[0], meta.centroid[1])
            candidates.append((code, dist))

    if not candidates:
        # Fallback : région dont le centroïde est le plus proche
        log.debug(
            "Aucune bbox exacte pour lat={} lon={} — fallback distance centroïde", lat, lon
        )
        all_dists = [
            (code, haversine_distance(lat, lon, m.centroid[0], m.centroid[1]))
            for code, m in _REGIONS_DB.items()
        ]
        return min(all_dists, key=lambda x: x[1])[0]

    # Parmi les candidats bbox, retourner le plus proche (centroïde)
    return min(candidates, key=lambda x: x[1])[0]


# ═════════════════════════════════════════════════════════════════
# 3. ACCÈS AUX METADATA
# ═════════════════════════════════════════════════════════════════

def get_region_metadata(region_code: str) -> Optional[RegionMetadata]:
    """
    Retourne les métadonnées complètes d'une région.

    Args:
        region_code : Code MDG-xxx (voir REGIONS_MADAGASCAR dans constants.py)

    Returns:
        RegionMetadata ou None si code inconnu.
    """
    meta = _REGIONS_DB.get(region_code)
    if meta is None:
        log.warning("Région inconnue : {}", region_code)
    return meta


def get_neighboring_regions(region_code: str) -> List[str]:
    """
    Retourne la liste des codes de régions limitrophes.

    Utile pour :
        - Diffusion spatiale du paludisme (propagation vers voisins)
        - Lissage spatial des prédictions
        - Alertes régionales croisées

    Returns:
        Liste de codes ou [] si région inconnue.
    """
    meta = _REGIONS_DB.get(region_code)
    if meta is None:
        log.warning("get_neighboring_regions : région inconnue {}", region_code)
        return []
    return list(meta.voisins)


def build_region_centroid_map() -> Dict[str, Tuple[float, float]]:
    """
    Retourne un dictionnaire {code_région: (lat, lon)} pour toutes les régions.

    Usage :
        - Construction matrices de distance entre régions
        - Initialisation cartes choroplèthes (Folium)
        - Features de distance dans les modèles ML
    """
    return {code: meta.centroid for code, meta in _REGIONS_DB.items()}


def get_all_region_codes() -> List[str]:
    """Retourne la liste ordonnée des 22 codes de régions."""
    return list(_REGIONS_DB.keys())


def get_regions_by_climate_zone(zone: ZoneClimatique) -> List[str]:
    """
    Filtre les régions par zone climatique.

    Usage typique :
        - Ajustement des paramètres du modèle par cluster climatique
        - Rapports régionaux groupés
    """
    return [
        code for code, meta in _REGIONS_DB.items()
        if meta.zone_climatique == zone
    ]


def get_regions_by_altitude_zone(zone: ZoneAltitude) -> List[str]:
    """Filtre les régions par zone altitudinale."""
    return [
        code for code, meta in _REGIONS_DB.items()
        if meta.zone_altitude == zone
    ]


# ═════════════════════════════════════════════════════════════════
# 4. CLASSIFICATION GÉOGRAPHIQUE
# ═════════════════════════════════════════════════════════════════

def classify_zone_altitude(altitude_m: float) -> ZoneAltitude:
    """
    Classifie une altitude en zone de transmission paludisme.

    Référence :
        OMS / Institut Pasteur Madagascar — au-dessus de 1500 m,
        la transmission du P. falciparum est quasi nulle.

    Args:
        altitude_m : Altitude en mètres (float ≥ 0)

    Returns:
        ZoneAltitude correspondante.
    """
    if altitude_m < 0:
        log.warning("Altitude négative reçue : {} m — traité comme 0", altitude_m)
        altitude_m = 0.0

    if altitude_m <= 200:
        return ZoneAltitude.COTE_BASSE
    elif altitude_m <= 800:
        return ZoneAltitude.TRANSITION
    elif altitude_m <= 1500:
        return ZoneAltitude.HAUTS_PLAT
    return ZoneAltitude.MONTAGNE


def get_climate_zone(region_code: str) -> Optional[ZoneClimatique]:
    """
    Retourne la zone climatique d'une région.

    Returns:
        ZoneClimatique ou None si code invalide.
    """
    meta = _REGIONS_DB.get(region_code)
    return meta.zone_climatique if meta else None


def get_malaria_transmission_factor(region_code: str) -> float:
    """
    Facteur de correction du risque paludisme selon l'altitude et le climat.

    Retourne un multiplicateur entre 0.0 (risque nul) et 1.0 (risque maximal).
    Utilisé en feature engineering pour pondérer les prédictions brutes du modèle.

    Logique :
        - Hauts Plateaux > 1500 m : 0.15 (transmission résiduelle saisonnière)
        - Hauts Plateaux 800–1500 m : 0.50
        - Zones de transition : 0.80
        - Côtes et zones humides : 1.00
        - Sud aride : 0.45 (faible humidité limite le vecteur)
    """
    meta = _REGIONS_DB.get(region_code)
    if not meta:
        return 1.0  # Valeur neutre si inconnu

    # Facteur altitude
    altitude_factor = {
        ZoneAltitude.MONTAGNE:  0.10,
        ZoneAltitude.HAUTS_PLAT: 0.45,
        ZoneAltitude.TRANSITION: 0.80,
        ZoneAltitude.COTE_BASSE: 1.00,
    }.get(meta.zone_altitude, 1.0)

    # Ajustement climatique
    climate_modifier = {
        ZoneClimatique.SUD_ARIDE:    0.55,  # Humidité limitante
        ZoneClimatique.HAUTS_PLAT:   0.90,  # Déjà pris en compte altitude
        ZoneClimatique.COTE_EST:     1.05,  # Pluies fréquentes → gîtes larvaires
        ZoneClimatique.NORD:         1.05,
        ZoneClimatique.COTE_OUEST:   1.00,
        ZoneClimatique.CENTRE_OUEST: 0.95,
    }.get(meta.zone_climatique, 1.0)

    return min(1.0, round(altitude_factor * climate_modifier, 3))


# ═════════════════════════════════════════════════════════════════
# 5. ZONES HUMIDES (GÎTES LARVAIRES)
# ═════════════════════════════════════════════════════════════════

def distance_to_nearest_wetland(lat: float, lon: float, region_code: str) -> float:
    """
    Calcule la distance approximative au point humide le plus proche dans la région.

    Note :
        Approximation basée sur les zones humides cataloguées dans _REGIONS_DB.
        Pour une analyse fine, utiliser des données NDVI ou satellite (geo_processor.py).

    Args:
        lat, lon     : Coordonnées du point d'intérêt
        region_code  : Code de la région pour filtrer les zones humides connues

    Returns:
        Distance en km au point humide le plus proche.
        Retourne 999.0 si aucune zone humide répertoriée dans la région.
    """
    meta = _REGIONS_DB.get(region_code)
    if not meta or not meta.zones_humides_coords:
        return 999.0

    distances = [
        haversine_distance(lat, lon, wlat, wlon)
        for wlat, wlon in meta.zones_humides_coords
    ]
    return min(distances)


# ═════════════════════════════════════════════════════════════════
# 6. ENCODAGE CYCLIQUE (FEATURE ENGINEERING)
# ═════════════════════════════════════════════════════════════════

def encode_cyclic(value: float, max_value: float) -> Tuple[float, float]:
    """
    Encode une variable cyclique (mois, heure, jour de l'an) en sin/cos.

    Sans cet encodage, le modèle ne peut pas savoir que janvier (1) et
    décembre (12) sont proches. L'encodage circulaire préserve cette continuité.

    Args:
        value     : Valeur à encoder (ex: mois = 1–12, heure = 0–23)
        max_value : Période du cycle (ex: 12 pour mois, 24 pour heures)

    Returns:
        Tuple (sin_val, cos_val) ∈ [-1, 1]

    Example:
        >>> encode_cyclic(1, 12)   # Janvier
        (0.5, 0.866)
        >>> encode_cyclic(12, 12)  # Décembre — proche de janvier
        (-0.5, 0.866)
    """
    angle = 2 * math.pi * value / max_value
    return round(math.sin(angle), 6), round(math.cos(angle), 6)


def encode_month_cyclic(month: int) -> Tuple[float, float]:
    """Raccourci : encode un mois (1–12) en sin/cos."""
    return encode_cyclic(month, 12)


def encode_day_of_year_cyclic(day_of_year: int) -> Tuple[float, float]:
    """Raccourci : encode un jour de l'année (1–365) en sin/cos."""
    return encode_cyclic(day_of_year, 365)


def encode_week_of_year_cyclic(week: int) -> Tuple[float, float]:
    """Raccourci : encode une semaine de l'année (1–52) en sin/cos."""
    return encode_cyclic(week, 52)


# ═════════════════════════════════════════════════════════════════
# 7. COULEURS ALERTE & VISUALISATION
# ═════════════════════════════════════════════════════════════════

def region_risk_color(score: float) -> str:
    """
    Retourne la couleur d'alerte hexadécimale associée à un score de risque.

    Mapping aligné avec COULEURS_ALERTE dans constants.py :
        0.00–0.25 → vert  (#388E3C)
        0.25–0.50 → jaune (#F9A825)
        0.50–0.75 → orange (#F57C00)
        0.75–1.00 → rouge (#D32F2F)

    Args:
        score : Float ∈ [0.0, 1.0]

    Returns:
        Couleur hex string.
    """
    score = max(0.0, min(1.0, score))  # Clamp

    if score < 0.25:
        return COULEURS_ALERTE["vert"]
    elif score < 0.50:
        return COULEURS_ALERTE["jaune"]
    elif score < 0.75:
        return COULEURS_ALERTE["orange"]
    return COULEURS_ALERTE["rouge"]


# ═════════════════════════════════════════════════════════════════
# 8. AGRÉGATION SPATIALE DES SCORES
# ═════════════════════════════════════════════════════════════════

def aggregate_region_scores(
    scores: Dict[str, float],
    include_neighbors_weight: float = 0.0,
) -> Dict[str, Dict]:
    """
    Agrège et enrichit les scores de risque par région avec des statistiques spatiales.

    Args:
        scores                   : {code_région: score_float ∈ [0,1]}
        include_neighbors_weight : Si > 0, pondère le score avec la moyenne des
                                   voisins (lissage spatial). Valeur typique : 0.2

    Returns:
        Dict enrichi par région :
        {
          "MDG-ANA": {
            "score":          float,
            "score_lisse":    float,   # Score après lissage spatial
            "couleur":        str,     # Hex couleur alerte
            "voisins_scores": list,    # Scores des régions voisines
            "score_moyen_voisins": float,
          },
          ...
        }
    """
    if not scores:
        return {}

    result: Dict[str, Dict] = {}

    for code, score in scores.items():
        score = max(0.0, min(1.0, float(score)))
        voisins = get_neighboring_regions(code)

        # Scores des voisins (uniquement ceux présents dans le dict)
        voisins_scores = [
            scores[v] for v in voisins if v in scores
        ]

        score_moyen_voisins = (
            sum(voisins_scores) / len(voisins_scores)
            if voisins_scores else score
        )

        # Lissage spatial optionnel
        if include_neighbors_weight > 0 and voisins_scores:
            score_lisse = (
                score * (1 - include_neighbors_weight)
                + score_moyen_voisins * include_neighbors_weight
            )
        else:
            score_lisse = score

        score_lisse = round(min(1.0, max(0.0, score_lisse)), 4)

        result[code] = {
            "score":               round(score, 4),
            "score_lisse":         score_lisse,
            "couleur":             region_risk_color(score_lisse),
            "voisins_scores":      [round(s, 4) for s in voisins_scores],
            "score_moyen_voisins": round(score_moyen_voisins, 4),
        }

    # Statistiques globales (niveau national)
    all_scores = [v["score"] for v in result.values()]
    result["__national__"] = {
        "score_moyen":  round(sum(all_scores) / len(all_scores), 4),
        "score_max":    round(max(all_scores), 4),
        "score_min":    round(min(all_scores), 4),
        "n_regions":    len(all_scores),
        "n_alerte":     sum(1 for s in all_scores if s >= 0.50),
        "n_critique":   sum(1 for s in all_scores if s >= 0.75),
    }

    return result


# ═════════════════════════════════════════════════════════════════
# 9. UTILITAIRES DIVERS
# ═════════════════════════════════════════════════════════════════

def build_distance_matrix() -> Dict[str, Dict[str, float]]:
    """
    Construit la matrice de distances (km) entre tous les centroïdes des 22 régions.

    Usage :
        - Feature ML : distance aux foyers épidémiques connus
        - Optimisation logistique (acheminement fournitures médicales)
        - Clustering géographique

    Returns:
        Dict imbriqué {code_a: {code_b: distance_km}}

    Note :
        Calcul en O(n²) — résultat statique, peut être mis en cache Redis.
    """
    codes = list(_REGIONS_DB.keys())
    matrix: Dict[str, Dict[str, float]] = {c: {} for c in codes}

    for i, code_a in enumerate(codes):
        for code_b in codes[i:]:
            dist = distance_between_regions(code_a, code_b)
            matrix[code_a][code_b] = dist
            matrix[code_b][code_a] = dist  # Symétrique

    return matrix


def get_region_population_weight(region_code: str, total_pop: Optional[int] = None) -> float:
    """
    Retourne le poids démographique d'une région (fraction de la population nationale).

    Utile pour pondérer les prédictions lors des agrégations nationales.

    Args:
        region_code : Code de la région
        total_pop   : Population totale de référence (défaut : somme des 22 régions)

    Returns:
        Float ∈ [0, 1] ou 0.0 si région inconnue.
    """
    meta = _REGIONS_DB.get(region_code)
    if not meta:
        return 0.0

    if total_pop is None:
        total_pop = sum(m.population_estimee for m in _REGIONS_DB.values())

    return round(meta.population_estimee / total_pop, 6) if total_pop > 0 else 0.0