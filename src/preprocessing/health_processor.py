"""
Traitement des données épidémiologiques (paludisme) et nutritionnelles.

Responsabilités :
  1. Nettoyage et validation des données DHIS2 / WHO GHO
  2. Calcul des lags épidémiologiques (cas semaines précédentes)
  3. Lissage des séries temporelles (moyenne mobile, Savitzky-Golay)
  4. Détection de ruptures de série (changement de méthode de collecte)
  5. Normalisation GAM/SAM, FCS, HDDS pour le ML
  6. Calcul des indicateurs dérivés (taux incidence, létalité, taux positivité)

Appelé par :
  - src/preprocessing/feature_engineering.py
  - ml/training_scripts/train_malaria.py
  - ml/training_scripts/train_nutrition.py
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────────────
# Seuils de validation — données épidémiologiques
# ─────────────────────────────────────────────────────────────────
EPIDEMIO_BOUNDS: Dict[str, Tuple[float, float]] = {
    "cas_confirmes":          (0, 500_000),
    "cas_confirmes_mixte":           (0, 1_000_000),
    "deces":                  (0, 10_000),
    "hospitalisations":       (0, 100_000),
    "taux_positivite_tdr_pct":(0, 100),
    "taux_incidence_pour_mille": (0, 500),
}

NUTRITION_BOUNDS: Dict[str, Tuple[float, float]] = {
    "gam_pct":         (0, 60),
    "sam_pct":         (0, 40),
    "mam_pct":         (0, 50),
    "stunting_pct":    (0, 80),
    "underweight_pct": (0, 80),
    "score_fcs":       (0, 112),
    "hdds":            (0, 12),
    "rcsi":            (0, 56),
    "prix_riz_kg":     (0, 50_000),   # MGA
    "prix_manioc_kg":  (0, 10_000),
}

# Prix de référence nationaux Madagascar (MGA/kg, 2024)
PRIX_REFERENCE_MGA: Dict[str, float] = {
    "riz":     1800.0,
    "manioc":  400.0,
    "mais":    800.0,
    "haricots":3000.0,
    "huile":   6000.0,  # par litre
}


class HealthProcessor:
    """
    Processeur de données de santé (épidémiologie paludisme + nutrition).
    """

    # ─────────────────────────────────────────────
    # Nettoyage épidémiologique
    # ─────────────────────────────────────────────


    @staticmethod
    def _to_date(value):
        """
        Convertit une valeur en objet date.

        Accepte :
            - datetime.date
            - "YYYY-MM-DD"
            - "YYYY-MM-DD HH:MM:SS"

        Retourne date.min si invalide.
        """
        if value is None:
            return date.min

        if isinstance(value, date):
            return value

        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except Exception:
                return date.min

        return date.min
    def clean_malaria_records(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Nettoie une série de cas paludisme :
          - Clampe les outliers physiques
          - Impute les semaines manquantes par interpolation
          - Recalcule les taux dérivés si incohérents
        """
        if not records:
            return []

        cleaned = []
        for r in records:
            c = dict(r)
            for var, (lo, hi) in EPIDEMIO_BOUNDS.items():
                val = c.get(var)
                if val is None:
                    continue
                c[var] = int(np.clip(float(val), lo, hi))

            # Recalcul cohérence : cas_confirmes ≤ cas_confirmes_mixte + 20%
            cas_conf = c.get("cas_confirmes", 0)
            cas_susp = c.get("cas_confirmes_mixte", 0)
            if cas_conf > cas_susp * 1.2 and cas_susp > 0:
                logger.debug(
                    "Incohérence cas S{}-{}: confirmes={} > suspects={}",
                    c.get("semaine_epidemio"), c.get("annee"),
                    cas_conf, cas_susp,
                )
                c["fiabilite_donnees"] = "incomplète"

            # Recalcul taux positivité si TDR disponible
            tdr_total = c.get("tests_malaria", 0)
            tdr_pos   = c.get("tdr_positifs", 0)
            if tdr_total > 0 and tdr_pos >= 0:
                c["taux_positivite_tdr_pct"] = round(
                    min(100, tdr_pos / tdr_total * 100), 2
                )

            cleaned.append(c)

        return cleaned

    def fill_missing_weeks(
        self,
        records: List[Dict[str, Any]],
        date_debut: date,
        date_fin: date,
        region_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Détecte et impute les semaines manquantes dans la série
        (ex : semaine non reportée = 0 cas ou interpolée).
        """
        # Index par semaine ISO
        index: Dict[Tuple[int, int], Dict] = {}
        for r in records:
            key = (r.get("annee", 0), r.get("semaine_epidemio", 0))
            index[key] = r

        result = []
        current = date_debut
        while current <= date_fin:
            iso    = current.isocalendar()
            annee  = iso[0]
            semaine = iso[1]
            key    = (annee, semaine)

            if key in index:
                result.append(index[key])
            else:
                # Semaine manquante → record vide
                result.append({
                    "region_id":          region_id,
                    "annee":              annee,
                    "semaine_epidemio":   semaine,
                    "date_rapport":       str(current),
                    "cas_confirmes":      0,
                    "cas_confirmes_mixte":       0,
                    "deces":              0,
                    "hospitalisations":   0,
                    "taux_incidence_pour_mille": 0.0,
                    "taux_positivite_tdr_pct":   0.0,
                    "population_a_risque": 0,
                    "source":             "imputé (semaine manquante)",
                    "fiabilite_donnees":  "incomplète",
                })

            current += timedelta(weeks=1)

        return result

    # ─────────────────────────────────────────────
    # Lags épidémiologiques
    # ─────────────────────────────────────────────

    def compute_malaria_lags(
        self,
        records: List[Dict[str, Any]],
        n_lags: int = 4,
    ) -> Dict[str, float]:
        """
        Calcule les features de lag épidémiologique.
        Le paludisme présente une auto-corrélation forte sur 1-4 semaines.

        Retourne un dict avec :
          cas_lag_1sem, cas_lag_2sem, cas_lag_3sem, cas_lag_4sem
          (normalisés par population_a_risque pour comparabilité inter-régions)
        """
        if not records:
            return {f"cas_lag_{i}sem": 0.0 for i in range(1, n_lags + 1)}

        # Tri chronologique
        sorted_records = sorted(
            records,
            key=lambda r: (r.get("annee", 0), r.get("semaine_epidemio", 0))
        )

        features: Dict[str, float] = {}
        n = len(sorted_records)

        for lag in range(1, n_lags + 1):
            idx = n - lag - 1  # Dernier enregistrement = semaine courante
            if idx >= 0:
                rec = sorted_records[idx]
                cas = float(rec.get("cas_confirmes", 0))
                pop = float(rec.get("population_a_risque", 500_000))
                # Normalisation pour 100k habitants (comparabilité)
                features[f"cas_lag_{lag}sem"] = round(
                    cas / pop * 100_000 if pop > 0 else 0.0, 4
                )
            else:
                features[f"cas_lag_{lag}sem"] = 0.0

        # Taux de positivité TDR de la dernière semaine disponible
        last = sorted_records[-1] if sorted_records else {}
        features["taux_positivite_tdr_pct"] = float(
            last.get("taux_positivite_tdr_pct", 0)
        )

        return features

    def smooth_time_series(
        self,
        values: List[float],
        window: int = 3,
        method: str = "rolling_mean",
    ) -> List[float]:
        """
        Lisse une série temporelle épidémiologique.
          method : 'rolling_mean' | 'savgol'
        """
        if len(values) < window:
            return values

        arr = np.array(values, dtype=float)

        if method == "rolling_mean":
            kernel = np.ones(window) / window
            smoothed = np.convolve(arr, kernel, mode="same")
            # Correction des bords
            smoothed[:window // 2] = arr[:window // 2]
            smoothed[-(window // 2):] = arr[-(window // 2):]
            return smoothed.tolist()

        elif method == "savgol":
            try:
                from scipy.signal import savgol_filter
                poly_order = min(3, window - 1)
                return savgol_filter(arr, window_length=window, polyorder=poly_order).tolist()
            except ImportError:
                return self.smooth_time_series(values, window, "rolling_mean")

        return values

    def detect_series_break(
        self,
        values: List[float],
        zscore_threshold: float = 3.0,
    ) -> List[int]:
        """
        Détecte les ruptures de série (changement structurel).
        Retourne les indices des points anomaliques.
        """
        if len(values) < 8:
            return []

        arr = np.array(values)
        mean = np.mean(arr)
        std  = np.std(arr) + 1e-9

        return [
            i for i, v in enumerate(arr)
            if abs(v - mean) / std > zscore_threshold
        ]

    # ─────────────────────────────────────────────
    # Nettoyage nutritionnel
    # ─────────────────────────────────────────────

    def clean_nutrition_record(
        self, record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Nettoie un enregistrement nutritionnel :
          - Valide les bornes OMS pour GAM/SAM/MAM
          - Assure la cohérence GAM = SAM + MAM
          - Normalise les prix alimentaires
        """
        cleaned = dict(record)

        for var, (lo, hi) in NUTRITION_BOUNDS.items():
            val = cleaned.get(var)
            if val is None:
                continue
            cleaned[var] = float(np.clip(float(val), lo, hi))

        # Cohérence GAM ≥ SAM + MAM
        gam = cleaned.get("gam_pct", 0)
        sam = cleaned.get("sam_pct", 0)
        mam = cleaned.get("mam_pct", 0)

        if sam + mam > gam * 1.05:  # 5% de tolérance
            logger.debug(
                "Incohérence GAM: sam({}) + mam({}) > gam({})",
                sam, mam, gam
            )
            # Répartition proportionnelle
            total = sam + mam
            if total > 0:
                cleaned["sam_pct"] = round(gam * (sam / total), 2)
                cleaned["mam_pct"] = round(gam * (mam / total), 2)

        return cleaned

    def compute_nutrition_lags(
        self,
        records: List[Dict[str, Any]],
        reference_date: date,
        lags_months: List[int] = [1, 3, 6],
    ) -> Dict[str, float]:
        """
        Calcule les lags nutritionnels.

        Parameters
        ----------
        records :
            Historique nutrition.

        reference_date :
            Date de construction des features.

        lags_months :
            Liste des lags en mois.

        Returns
        -------
        dict
        """

        if not records:
            return {
                "gam_lag_1m": 0.0,
                "gam_lag_3m": 0.0,
                "gam_lag_6m": 0.0,
                "sam_lag_1m": 0.0,
                "variation_gam_3m": 0.0,
            }

        sorted_records = sorted(
            records,
            key=lambda r: self._to_date(r.get("date_observation"))
        )

        features = {}

        for lag in lags_months:

            target = reference_date - timedelta(days=lag * 30)

            closest = min(
                sorted_records,
                key=lambda r: abs(
                    (
                        self._to_date(r.get("date_observation"))
                        - target
                    ).days
                ),
            )

            features[f"gam_lag_{lag}m"] = float(
                closest.get("gam_pct") or 0
            )

            if lag == 1:
                features["sam_lag_1m"] = float(
                    closest.get("sam_pct") or 0
                )

        gam_current = float(
            sorted_records[-1].get("gam_pct") or 0
        )

        features["variation_gam_3m"] = round(
            gam_current - features["gam_lag_3m"],
            2,
        )

        return features
    # ─────────────────────────────────────────────
    # Normalisation features nutrition pour ML
    # ─────────────────────────────────────────────

    @staticmethod
    def normalize_prix(
        prix_dict: Dict[str, Optional[float]]
    ) -> Dict[str, float]:
        """
        Normalise les prix alimentaires par rapport aux références nationales.
        Prix normalisé = prix_observé / prix_référence
        1.0 = prix normal | > 1.5 = choc inflationniste
        """
        return {
            "prix_riz_normalise": round(
                (prix_dict.get("prix_riz_kg") or PRIX_REFERENCE_MGA["riz"])
                / PRIX_REFERENCE_MGA["riz"], 3
            ),
            "prix_manioc_normalise": round(
                (prix_dict.get("prix_manioc_kg") or PRIX_REFERENCE_MGA["manioc"])
                / PRIX_REFERENCE_MGA["manioc"], 3
            ),
        }

    @staticmethod
    def compute_soudure_features(
        region_id: str,
        current_date: Optional[date] = None,
    ) -> Dict[str, float]:
        """
        Calcule les features liées à la période de soudure.
        Retourne en_periode_soudure et semaines_avant_soudure_norm.
        """
        from src.data_collection.nutrition_fetcher import REGION_FOOD_PROFILE, DEFAULT_FOOD_PROFILE

        current_date = current_date or date.today()
        mois = current_date.month

        profile = REGION_FOOD_PROFILE.get(region_id, DEFAULT_FOOD_PROFILE)
        mois_soudure = profile.get("mois_soudure", [11, 12])

        en_soudure = mois in mois_soudure

        # Semaines avant prochaine soudure (normalisé 0-1 pour ML)
        if not en_soudure:
            prochains = [m for m in mois_soudure if m > mois]
            if not prochains:
                prochains = [m + 12 for m in mois_soudure]
            mois_debut = min(prochains)
            semaines_avant = max(0, (mois_debut - mois) * 4)
        else:
            semaines_avant = 0

        return {
            "en_periode_soudure":         float(int(en_soudure)),
            "semaines_avant_soudure_norm": round(
                min(1.0, semaines_avant / 24), 3  # Normalisé sur 24 semaines max
            ),
        }

    @staticmethod
    def compute_vulnerability_index(region_id: str) -> float:
        """
        Retourne l'indice de vulnérabilité régionale [0,1].
        Chargé depuis le profil REGION_FOOD_PROFILE.
        """
        from src.data_collection.nutrition_fetcher import REGION_FOOD_PROFILE, DEFAULT_FOOD_PROFILE
        profile = REGION_FOOD_PROFILE.get(region_id, DEFAULT_FOOD_PROFILE)
        return float(profile.get("indice_vulnerabilite", 0.5))

    # ─────────────────────────────────────────────
    # Indicateurs épidémio dérivés
    # ─────────────────────────────────────────────

    @staticmethod
    def compute_case_fatality_rate(
        deces: int, cas_confirmes: int
    ) -> float:
        """Taux de létalité (Case Fatality Rate) en %."""
        if cas_confirmes == 0:
            return 0.0
        return round(deces / cas_confirmes * 100, 3)

    @staticmethod
    def compute_incidence_rate(
        cas_confirmes: int,
        population: int,
        per: int = 1000,
    ) -> float:
        """Taux d'incidence pour `per` habitants."""
        if population == 0:
            return 0.0
        return round(cas_confirmes / population * per, 4)

    @staticmethod
    def encode_endemicite(endemicite: str) -> int:
        """Encode le niveau d'endémicité en entier ordinal."""
        mapping = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
        return mapping.get(endemicite, 1)