"""
Nettoyage, normalisation et transformation des données météorologiques.

Responsabilités :
  1. Détection et imputation des valeurs manquantes / aberrantes
  2. Agrégation temporelle (horaire → journalier → hebdomadaire)
  3. Calcul des fenêtres glissantes (rolling 7j, 14j, 30j)
  4. Encodage cyclique des variables temporelles (sin/cos)
  5. Calcul indices dérivés (SPI, Heat Index, anomalies)
  6. Normalisation des variables pour le ML

Appelé par :
  - src/preprocessing/feature_engineering.py (build_malaria/nutrition_features)
  - ml/training_scripts/train_malaria.py (pipeline d'entraînement)
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────────────
# Limites physiques pour détection outliers (Madagascar context)
# ─────────────────────────────────────────────────────────────────
WEATHER_BOUNDS: Dict[str, Tuple[float, float]] = {
    "temperature_moy_c":    (-5.0,  45.0),
    "temperature_min_c":    (-10.0, 40.0),
    "temperature_max_c":    (0.0,   50.0),
    "precipitations_mm":    (0.0,   500.0),  # Max cyclone intense
    "humidite_moy_pct":     (0.0,   100.0),
    "vent_kmh":             (0.0,   300.0),  # Max cyclone cat 5
    "pression_hpa":         (850.0, 1050.0),
    "couverture_nuageuse_pct": (0.0, 100.0),
    "rayonnement_solaire_mj":  (0.0, 40.0),
    "humidite_sol_fraction":   (0.0, 1.0),
    "ndvi":                 (-1.0,  1.0),
}

# Encodage saison : 0=sèche, 1=transition, 2=pluies
SAISON_ENCODING: Dict[int, int] = {
    1: 2, 2: 2, 3: 2, 4: 2,   # Saison des pluies
    5: 1, 10: 1,                # Transition
    6: 0, 7: 0, 8: 0, 9: 0,   # Saison sèche
    11: 2, 12: 2,               # Début saison des pluies
}

# Encodage zones climatiques
ZONE_CLIMATIQUE_ENCODING: Dict[str, int] = {
    "tropical_highland":  0,
    "tropical_humid":     1,
    "tropical_sub_humid": 2,
    "tropical_dry":       3,
    "arid":               4,
    "semi_arid":          5,
    "tropical_sub_arid":  6,
}


class WeatherProcessor:
    """
    Processeur de données météorologiques.
    Transforme les données brutes des fetchers en features propres pour le ML.
    """

    def __init__(self):
        self._imputation_stats: Dict[str, int] = {}  # Compteur imputations

    # ─────────────────────────────────────────────
    # Nettoyage — valeurs manquantes et outliers
    # ─────────────────────────────────────────────

    def clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nettoie un enregistrement météo journalier :
          - Remplace les valeurs NASA POWER sentinel (-999) par NaN
          - Clamp les outliers physiquement impossibles
          - Imputation basique des NaN par valeur climatologique
        """
        cleaned = dict(record)

        for var, (lower, upper) in WEATHER_BOUNDS.items():
            val = cleaned.get(var)
            if val is None:
                continue

            val = float(val)

            # Valeurs sentinel NASA POWER / DHIS2
            if val in (-999, -9999, 9999, 99999):
                cleaned[var] = None
                self._imputation_stats[var] = self._imputation_stats.get(var, 0) + 1
                continue

            # Clamp physique
            if val < lower or val > upper:
                logger.debug(
                    "Outlier clampé {}: {:.2f} → [{}, {}]",
                    var, val, lower, upper
                )
                cleaned[var] = float(np.clip(val, lower, upper))

        return cleaned

    def clean_series(self, records: List[Dict]) -> List[Dict]:
        """Nettoie une série temporelle et impute les valeurs manquantes."""
        if not records:
            return []

        cleaned = [self.clean_record(r) for r in records]

        # Imputation par interpolation linéaire sur les NaN
        for var in WEATHER_BOUNDS.keys():
            vals = [r.get(var) for r in cleaned]
            imputed = self._interpolate_series(vals)
            for i, r in enumerate(cleaned):
                r[var] = imputed[i]

        n_imputations = sum(self._imputation_stats.values())
        if n_imputations > 0:
            logger.debug(
                "Imputation météo : {} valeurs manquantes traitées",
                n_imputations
            )

        return cleaned

    @staticmethod
    def _interpolate_series(values: List[Optional[float]]) -> List[float]:
        """
        Interpolation linéaire pour les NaN dans une série.
        Remplace les NaN en début/fin par la première/dernière valeur connue.
        """
        if not values:
            return []

        arr = np.array([v if v is not None else np.nan for v in values], dtype=float)

        # Si tout est NaN → retourner des zéros
        if np.all(np.isnan(arr)):
            return [0.0] * len(arr)

        # Interpolation linéaire
        nans_idx = np.where(np.isnan(arr))[0]
        if len(nans_idx) == 0:
            return arr.tolist()

        valid_idx = np.where(~np.isnan(arr))[0]
        arr_interp = np.interp(np.arange(len(arr)), valid_idx, arr[valid_idx])
        return arr_interp.tolist()

    # ─────────────────────────────────────────────
    # Agrégation temporelle
    # ─────────────────────────────────────────────

    def aggregate_daily(self, hourly_records: List[Dict]) -> List[Dict]:
        """
        Agrège des données horaires en journalières.
        Calcule min/max/moy pour température, somme pour précipitations.
        """
        from collections import defaultdict

        daily: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

        for r in hourly_records:
            dt = r.get("horodatage", r.get("date", ""))
            if isinstance(dt, str):
                day_key = dt[:10]  # "YYYY-MM-DD"
            elif isinstance(dt, datetime):
                day_key = dt.strftime("%Y-%m-%d")
            else:
                continue

            daily[day_key]["temperature_c"].append(r.get("temperature_c", 0))
            daily[day_key]["precipitations_mm"].append(r.get("precipitations_mm", 0))
            daily[day_key]["humidite_pct"].append(r.get("humidite_pct", 70))
            daily[day_key]["vent_kmh"].append(r.get("vent_kmh", 0))
            daily[day_key]["pression_hpa"].append(r.get("pression_hpa", 1013))
            daily[day_key]["couverture_nuageuse_pct"].append(
                r.get("couverture_nuageuse_pct", 0)
            )

        result = []
        for day_key in sorted(daily.keys()):
            d = daily[day_key]
            temps = d.get("temperature_c", [20])
            result.append({
                "date":                  day_key,
                "temperature_moy_c":     round(float(np.mean(temps)), 2),
                "temperature_min_c":     round(float(np.min(temps)), 2),
                "temperature_max_c":     round(float(np.max(temps)), 2),
                "precipitations_mm":     round(float(np.sum(d.get("precipitations_mm", [0]))), 2),
                "humidite_moy_pct":      round(float(np.mean(d.get("humidite_pct", [70]))), 1),
                "vent_kmh":              round(float(np.mean(d.get("vent_kmh", [0]))), 1),
                "pression_hpa":          round(float(np.mean(d.get("pression_hpa", [1013]))), 1),
                "couverture_nuageuse_pct": round(
                    float(np.mean(d.get("couverture_nuageuse_pct", [0]))), 1
                ),
            })

        return result

    # ─────────────────────────────────────────────
    # Features de fenêtres glissantes
    # ─────────────────────────────────────────────

    def compute_rolling_features(
        self,
        records: List[Dict],
        windows: List[int] = [7, 14, 30],
    ) -> Dict[str, float]:
        """
        Calcule les statistiques sur fenêtres glissantes pour les
        N derniers jours d'une série météo ordonnée chronologiquement.

        Retourne un dict plat avec les features nommées
        (ex: precipitations_7j_mm, precipitations_14j_mm...).

        Ce dict est directement utilisé par FeatureEngineer.
        """
        if not records:
            return self._default_rolling_features(windows)

        # Tri chronologique
        sorted_records = sorted(records, key=lambda r: r.get("date", ""))

        features: Dict[str, float] = {}

        for window in windows:
            window_data = sorted_records[-window:] if len(sorted_records) >= window \
                else sorted_records

            if not window_data:
                features[f"precipitations_{window}j_mm"] = 0.0
                continue

            pluies = [float(r.get("precipitations_mm", 0)) for r in window_data]
            temps  = [float(r.get("temperature_moy_c", 20)) for r in window_data]
            humid  = [float(r.get("humidite_moy_pct", 70)) for r in window_data]

            features[f"precipitations_{window}j_mm"] = round(float(np.sum(pluies)), 2)
            features[f"temp_moy_{window}j_c"]        = round(float(np.mean(temps)), 2)
            features[f"humidite_moy_{window}j_pct"]  = round(float(np.mean(humid)), 1)
            features[f"jours_pluie_{window}j"]       = int(sum(1 for p in pluies if p > 1.0))

        # Features du dernier jour disponible
        last = sorted_records[-1]
        features["temperature_moy_c"]      = float(last.get("temperature_moy_c", 20))
        features["temperature_min_c"]      = float(last.get("temperature_min_c", 15))
        features["temperature_max_c"]      = float(last.get("temperature_max_c", 30))
        features["humidite_moy_pct"]       = float(last.get("humidite_moy_pct", 70))
        features["vent_kmh"]               = float(last.get("vent_kmh", 10))
        features["pression_hpa"]           = float(last.get("pression_hpa", 1013))
        features["couverture_nuageuse_pct"]= float(last.get("couverture_nuageuse_pct", 50))
        features["humidite_sol_fraction"]  = float(last.get("humidite_sol_fraction", 0.3))
        features["ndvi"]                   = float(last.get("ndvi", 0.3))

        return features

    @staticmethod
    def _default_rolling_features(windows: List[int]) -> Dict[str, float]:
        """Valeurs par défaut si pas de données météo disponibles."""
        features: Dict[str, float] = {}
        for w in windows:
            features[f"precipitations_{w}j_mm"] = 0.0
            features[f"temp_moy_{w}j_c"]        = 24.0
            features[f"humidite_moy_{w}j_pct"]  = 72.0
            features[f"jours_pluie_{w}j"]       = 0.0
        features.update({
            "temperature_moy_c": 24.0, "temperature_min_c": 18.0,
            "temperature_max_c": 30.0, "humidite_moy_pct": 72.0,
            "vent_kmh": 12.0, "pression_hpa": 1013.0,
            "couverture_nuageuse_pct": 50.0, "humidite_sol_fraction": 0.3,
            "ndvi": 0.3,
        })
        return features

    # ─────────────────────────────────────────────
    # Encodages temporels cycliques
    # ─────────────────────────────────────────────

    @staticmethod
    def encode_temporal(target_date: date) -> Dict[str, float]:
        """
        Encode cycliquement la date pour capturer la saisonnalité :
          - semaine_sin / semaine_cos (cycle 52 semaines)
          - mois_sin   / mois_cos    (cycle 12 mois)
          - saison_encoded            (0=sèche, 1=transition, 2=pluies)

        L'encodage sin/cos préserve la continuité : semaine 52 ≈ semaine 1.
        """
        iso = target_date.isocalendar()
        semaine = iso[1]
        mois    = target_date.month

        return {
            "semaine_sin":    round(math.sin(2 * math.pi * semaine / 52), 4),
            "semaine_cos":    round(math.cos(2 * math.pi * semaine / 52), 4),
            "mois_sin":       round(math.sin(2 * math.pi * mois / 12), 4),
            "mois_cos":       round(math.cos(2 * math.pi * mois / 12), 4),
            "saison_encoded": float(SAISON_ENCODING.get(mois, 1)),
        }

    @staticmethod
    def encode_zone_climatique(zone: str) -> int:
        """Encode la zone climatique en entier (ordinal)."""
        return ZONE_CLIMATIQUE_ENCODING.get(zone, 2)

    # ─────────────────────────────────────────────
    # Indices climatiques dérivés
    # ─────────────────────────────────────────────

    @staticmethod
    def compute_heat_index(temp_c: float, humidite_pct: float) -> float:
        """Rothfusz Heat Index — température ressentie (°C)."""
        if temp_c < 27:
            return temp_c
        T = temp_c * 9 / 5 + 32
        H = humidite_pct
        HI = (
            -42.379 + 2.04901523 * T + 10.14333127 * H
            - 0.22475541 * T * H - 0.00683783 * T ** 2
            - 0.05481717 * H ** 2 + 0.00122874 * T ** 2 * H
            + 0.00085282 * T * H ** 2 - 0.00000199 * T ** 2 * H ** 2
        )
        return round((HI - 32) * 5 / 9, 1)

    @staticmethod
    def compute_spi(precipitations: List[float], scale: int = 30) -> float:
        """
        Standardized Precipitation Index.
        SPI < -1 : sécheresse modérée | SPI < -2 : sévère | SPI > 1 : humide.
        """
        if len(precipitations) < scale:
            return 0.0
        window = np.array(precipitations[-scale:])
        mean = float(np.mean(window))
        std  = float(np.std(window)) + 1e-9
        return round((float(np.sum(window)) / scale - mean) / std, 3)

    @staticmethod
    def compute_anomaly(
        current_value: float,
        historical_series: List[float],
    ) -> float:
        """Écart à la normale (en écarts-types) — Z-score."""
        if not historical_series:
            return 0.0
        mean = float(np.mean(historical_series))
        std  = float(np.std(historical_series)) + 1e-9
        return round((current_value - mean) / std, 3)

    @staticmethod
    def compute_zones_humides(
        precipitations_30j: float,
        ndvi: float,
        altitude_m: float,
    ) -> float:
        """
        Estime le pourcentage de zones humides favorables aux moustiques (0-100).
        Heuristique basée sur précipitations, végétation et altitude.
        """
        # Facteur précipitations (0-40)
        pluie_factor = min(40, precipitations_30j / 5)
        # Facteur NDVI (0-30 → végétation dense = plus de zones humides)
        ndvi_factor = max(0, ndvi) * 30
        # Facteur altitude (réduction au-dessus de 1200m)
        alt_penalty = max(0, (altitude_m - 1200) / 100) * 5
        score = pluie_factor + ndvi_factor - alt_penalty
        return round(float(np.clip(score, 0, 100)), 1)

    # ─────────────────────────────────────────────
    # Normalisation pour ML
    # ─────────────────────────────────────────────

    @staticmethod
    def normalize_price(
        price: Optional[float],
        reference_price: float,
    ) -> float:
        """Normalise un prix par rapport au prix de référence national."""
        if price is None or reference_price == 0:
            return 1.0
        return round(float(price) / reference_price, 3)

    @staticmethod
    def normalize_density(
        population: int,
        area_km2: float,
    ) -> float:
        """Densité population normalisée (log scale) pour le ML."""
        if area_km2 <= 0 or population <= 0:
            return 0.0
        density = population / area_km2
        return round(float(np.log1p(density)) / 10, 4)