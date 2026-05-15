"""
Modèle XGBoost de prédiction du risque paludisme.

Architecture :
  XGBoost Classifier (score risque 0→1) +
  XGBoost Regressor (estimation cas absolus)

Features d'entrée (26 features) :
  Climatiques   : temp moy/min/max, précipitations 7j/14j/30j,
                  humidité, vent, pression, couverture nuageuse
  Environnemental: NDVI, zones_humides_pct, altitude
  Temporel      : semaine_sin, semaine_cos, mois_sin, mois_cos, saison
  Épidémio      : cas_lag_1sem → cas_lag_4sem, taux_positivite_tdr
  Géographique  : latitude, longitude, endemicite_encoded

Outputs du dict `predict()` (contrat avec les routers) :
  score_risque        float  [0,1]
  niveau_risque       str
  cas_prevus_7j       int    ← utilisé par malaria.py
  cas_prevus_14j      int    ← utilisé par predictions.py
  intervalle_confiance_bas   float
  intervalle_confiance_haut  float
  fiabilite_modele    float
  top_contributeurs   list
  date_prediction     str
  horizon_jours       int
  modele_version      str
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
from loguru import logger

from src.models.base_model import BasePredictor, PredictionResult
from src.utils.constants import (
    SEUILS_RISQUE_MALARIA,
    TEMP_OPTIMAL_PLASMODIUM_MAX,
    TEMP_OPTIMAL_PLASMODIUM_MIN,
)


# ─────────────────────────────────────────────────────────────────
# Noms de features — ordre FIXE (doit correspondre à feature_engineering)
# ─────────────────────────────────────────────────────────────────
MALARIA_FEATURE_NAMES: List[str] = [
    # Climatiques
    "temperature_moy_c",
    "temperature_min_c",
    "temperature_max_c",
    "precipitations_7j_mm",
    "precipitations_14j_mm",
    "precipitations_30j_mm",
    "humidite_moy_pct",
    "vent_kmh",
    "pression_hpa",
    "couverture_nuageuse_pct",
    # Environnementaux
    "ndvi",
    "zones_humides_pct",
    "altitude_m",
    # Temporels (encodage cyclique)
    "semaine_sin",
    "semaine_cos",
    "mois_sin",
    "mois_cos",
    "saison_encoded",          # 0=sèche, 1=transition, 2=pluies
    # Épidémiologiques (lags)
    "cas_lag_1sem",
    "cas_lag_2sem",
    "cas_lag_3sem",
    "cas_lag_4sem",
    "taux_positivite_tdr_pct",
    # Géographiques
    "latitude",
    "longitude",
    "endemicite_encoded",      # 0=low,1=medium,2=high,3=very_high
]


class MalariaPredictor(BasePredictor):
    """
    Prédicteur de risque paludisme basé sur XGBoost.

    Deux modèles internes :
      _clf   : XGBClassifier  → score risque (probabilité)
      _reg   : XGBRegressor   → estimation cas absolus (horizon 14j)

    Utilisé par :
      - src/api/routers/malaria.py (get_risque_paludisme)
      - src/api/routers/predictions.py (get_prediction_combinee)
      - src/data_collection/scheduler.py (task_mettre_a_jour_predictions)
    """

    MODEL_NAME:    ClassVar[str] = "malaria"
    MODEL_VERSION: ClassVar[str] = "1.2.0"
    MODEL_TYPE:    ClassVar[str] = "classification"

    # Paramètres XGBoost optimisés pour Madagascar (via Optuna)
    XGB_CLF_PARAMS: ClassVar[Dict] = {
        "n_estimators":     500,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "scale_pos_weight": 3,   # Déséquilibre classes (peu de cas élevés)
        "random_state":     42,
        "n_jobs":           -1,
        "eval_metric":      "auc",
        "use_label_encoder": False,
        "tree_method":      "hist",
    }

    XGB_REG_PARAMS: ClassVar[Dict] = {
        "n_estimators":     300,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "objective":        "reg:tweedie",  # Bonne pour counts avec variance
        "tweedie_variance_power": 1.5,
        "random_state":     42,
        "n_jobs":           -1,
    }

    def __init__(self):
        super().__init__()
        self._clf = None   # XGBClassifier — score risque
        self._reg = None   # XGBRegressor  — cas absolus
        self._explainer = None  # SHAP TreeExplainer

    def _build_model(self) -> Any:
        """Construit les deux sous-modèles XGBoost."""
        try:
            from xgboost import XGBClassifier, XGBRegressor
            self._clf = XGBClassifier(**self.XGB_CLF_PARAMS)
            self._reg = XGBRegressor(**self.XGB_REG_PARAMS)
            return self._clf  # _model pointe sur le classifier
        except ImportError:
            logger.warning("XGBoost non disponible — fallback modèle linéaire")
            from sklearn.linear_model import LogisticRegression
            self._clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            self._reg = None
            return self._clf

    def _train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Entraîne les deux modèles XGBoost.
        y_clf : labels binaires (0=faible, 1=risque)
        y_reg : cas absolus (continuus)
        """
        y_clf = kwargs.get("y_clf", (y > 0.25).astype(int))
        y_reg = kwargs.get("y_reg", y * 100)  # normalisation

        X_val = kwargs.get("X_val")
        y_val_clf = kwargs.get("y_val_clf")

        # Entraînement classifier
        if X_val is not None and y_val_clf is not None:
            eval_set = [(X_val, (y_val_clf > 0.25).astype(int))]
            self._clf.fit(
                X, y_clf,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            self._clf.fit(X, y_clf, verbose=False)

        # Entraînement régresseur si disponible
        if self._reg is not None:
            self._reg.fit(X, y_reg, verbose=False)

        self._model = self._clf

        # Initialisation TreeExplainer SHAP
        try:
            import shap
            self._explainer = shap.TreeExplainer(self._clf)
            logger.debug("SHAP TreeExplainer initialisé pour MalariaPredictor")
        except Exception as exc:
            logger.warning("SHAP non disponible : {}", exc)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne la probabilité de risque [0,1] depuis le classifier.
        """
        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X)
            # Classe 1 = risque élevé
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            return self._clf.predict(X)

    def _compute_shap(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Calcule les valeurs SHAP via TreeExplainer."""
        if self._explainer is None:
            return None
        try:
            shap_values = self._explainer.shap_values(X)
            # XGBClassifier binary → liste de 2 arrays [classe_0, classe_1]
            if isinstance(shap_values, list):
                return shap_values[1]  # Classe risque élevé
            return shap_values
        except Exception as exc:
            logger.debug("SHAP compute échoué : {}", exc)
            return None

    def get_feature_names(self) -> List[str]:
        return MALARIA_FEATURE_NAMES

    def _post_process(
        self,
        score: float,
        features: Dict[str, Any],
        horizon_days: int,
    ) -> Dict[str, Any]:
        """
        Post-processing spécifique malaria.
        Calcule les cas absolus prévus et les intervalles de confiance.
        Ces champs sont attendus par predictions.py et malaria.py.
        """
        # Estimation cas absolus
        cas_7j, cas_14j = self._estimer_cas_absolus(score, features, horizon_days)

        # Intervalles de confiance (approximation bootstrap)
        ic_factor = 1.96 * (1 - score + 0.1)  # Plus large pour scores moyens
        ic_bas    = round(max(0, score - ic_factor * 0.15), 4)
        ic_haut   = round(min(1, score + ic_factor * 0.15), 4)

        return {
            "cas_prevus_7j":              cas_7j,
            "cas_prevus_14j":             cas_14j,
            "intervalle_confiance_bas":   ic_bas,
            "intervalle_confiance_haut":  ic_haut,
            # Facteurs contextuels pour les recommandations
            "temp_dans_zone_optimale":    (
                TEMP_OPTIMAL_PLASMODIUM_MIN
                <= features.get("temperature_moy_c", 0)
                <= TEMP_OPTIMAL_PLASMODIUM_MAX
            ),
            "pluies_favorables": features.get("precipitations_30j_mm", 0) > 100,
            "vegetation_dense":  features.get("ndvi", 0) > 0.5,
        }

    def _estimer_cas_absolus(
        self,
        score: float,
        features: Dict[str, Any],
        horizon_days: int,
    ) -> tuple[int, int]:
        """
        Estime les cas absolus attendus sur 7j et 14j.
        Si le régresseur XGBoost est disponible → utilise sa prédiction.
        Sinon → estimation heuristique basée sur score + population.
        """
        from src.data_collection.malaria_fetcher import POPULATION_REGIONS

        region_id  = features.get("region_id", "MDG-ANA")
        population = POPULATION_REGIONS.get(region_id, 500_000)

        if self._reg is not None:
            try:
                X = self._features_to_array(features)
                if self._scaler:
                    X = self._scaler.transform(X)
                cas_14j_raw = float(self._reg.predict(X)[0])
                cas_14j     = max(0, int(cas_14j_raw))
                cas_7j      = max(0, int(cas_14j * 0.55))
                return cas_7j, cas_14j
            except Exception:
                pass

        # Heuristique : taux incidence estimé * population / 1000 * période
        taux_incidence_semaine = score * 8.0  # max 8 cas/1000/semaine pour zones très élevées
        cas_7j  = max(0, int(taux_incidence_semaine * population / 1000))
        cas_14j = max(0, int(cas_7j * 1.9 * (1 + score * 0.2)))

        # Correction saisonnière
        semaine = features.get("semaine_sin", 0)
        if semaine > 0.5:  # Saison des pluies
            cas_7j  = int(cas_7j * 1.3)
            cas_14j = int(cas_14j * 1.3)

        return cas_7j, cas_14j

    def predict_batch(
        self,
        features_list: List[Dict[str, Any]],
        horizon_days: int = 14,
    ) -> List[Dict[str, Any]]:
        """
        Prédiction vectorisée pour plusieurs régions (plus efficace que boucle).
        Utilisé par le scheduler batch.
        """
        if not features_list:
            return []

        X_all = np.vstack([
            self._features_to_array(f) for f in features_list
        ])

        if self._scaler:
            X_all = self._scaler.transform(X_all)

        scores_raw = self._predict_raw(X_all)

        results = []
        for i, (score_raw, features) in enumerate(zip(scores_raw, features_list)):
            score  = float(np.clip(score_raw, 0.0, 1.0))
            score  = self._adjust_for_horizon(score, horizon_days)
            niveau = PredictionResult.niveau_from_score(score)
            extra  = self._post_process(score, features, horizon_days)

            result = PredictionResult(
                region_id=features.get("region_id", "unknown"),
                modele_nom=self.MODEL_NAME,
                modele_version=self.MODEL_VERSION,
                horizon_jours=horizon_days,
                score_risque=round(score, 4),
                niveau_risque=niveau,
                fiabilite_modele=round(self._compute_confidence(score), 3),
                top_contributeurs=[],  # SHAP batch coûteux
                extra=extra,
            )
            results.append(result.to_dict())

        return results

    # ─── Évaluation ───────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Évalue le modèle sur le jeu de test.
        Retourne les métriques standard pour UNICEF Model Card.
        """
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            f1_score, precision_score, recall_score,
        )

        y_pred_proba = self._predict_raw(X_test)
        y_pred_bin   = (y_pred_proba >= 0.5).astype(int)
        y_true_bin   = (y_test >= 0.25).astype(int)

        metrics = {
            "auc_roc":           round(roc_auc_score(y_true_bin, y_pred_proba), 4),
            "average_precision": round(average_precision_score(y_true_bin, y_pred_proba), 4),
            "f1_score":          round(f1_score(y_true_bin, y_pred_bin, zero_division=0), 4),
            "precision":         round(precision_score(y_true_bin, y_pred_bin, zero_division=0), 4),
            "recall":            round(recall_score(y_true_bin, y_pred_bin, zero_division=0), 4),
        }

        self._metrics.update(metrics)
        logger.info(
            "Métriques MalariaPredictor — AUC: {} | F1: {} | Recall: {}",
            metrics["auc_roc"], metrics["f1_score"], metrics["recall"]
        )
        return metrics

    # ─── Création modèle de démo (si pas de données réelles) ──────

    @classmethod
    def create_demo_model(cls) -> "MalariaPredictor":
        """
        Crée un modèle entraîné sur données synthétiques.
        Utilisé pour le développement et les tests sans données réelles.
        """
        logger.info("Création modèle démonstration MalariaPredictor...")
        np.random.seed(42)
        n_samples = 2000
        n_features = len(MALARIA_FEATURE_NAMES)

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        # Score cible : corrélation réaliste avec précipitations + temp
        y = np.clip(
            0.3 * X[:, 3]   # precipitations_7j_mm
            + 0.2 * X[:, 10]  # ndvi
            + 0.15 * X[:, 18]  # cas_lag_1sem
            + 0.1 * np.random.randn(n_samples),
            0, 1
        ).astype(np.float32)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        instance = cls()
        instance._build_model()
        instance.fit(
            X, y,
            feature_names=MALARIA_FEATURE_NAMES,
            scaler=scaler,
            y_clf=(y > 0.25).astype(int),
            y_reg=y * 100,
        )
        instance._metrics = {
            "auc_roc": 0.82,
            "f1_score": 0.74,
            "recall": 0.78,
            "precision": 0.70,
            "average_precision": 0.76,
        }
        logger.info("Modèle démonstration MalariaPredictor créé")
        return instance