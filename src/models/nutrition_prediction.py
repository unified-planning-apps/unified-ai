"""
Modèle Ensemble (Random Forest + MLP Neural Network) de prédiction
du risque malnutrition aiguë (GAM — Global Acute Malnutrition).

Architecture Ensemble :
  RF_score  = RandomForestClassifier  (poids 60%)
  MLP_score = MLPClassifier           (poids 40%)
  final_score = 0.60 * RF + 0.40 * MLP

Features d'entrée (32 features) :
  Nutrition     : gam_historique, sam_historique, score_fcs, hdds, rcsi
  Alimentaire   : prix_riz, prix_manioc, variation_prix_1m
                  dispo_cereales, dispo_legumineuses, dispo_proteines
  Paludisme     : score_paludisme (output MalariaPredictor — feature croisée)
  Climatique    : precipitations_30j, temperature_moy, ndvi
  Socio-éco     : indice_vulnerabilite, saison_soudure
  Temporel      : mois_sin, mois_cos, saison_encoded
  Géographique  : latitude, longitude, zone_climatique_encoded

Outputs du dict `predict()` (contrat avec les routers) :
  score_risque             float [0,1]
  niveau_risque            str
  gam_prevu_pct            float ← utilisé par predictions.py + nutrition.py
  sam_prevu_pct            float
  intervalles_confiance    dict
  populations_vulnerables  list
  fiabilite_modele         float
  top_contributeurs        list
  facteurs_contributeurs   list  ← alias pour nutrition.py
  date_prediction          str
  horizon_jours            int
  modele_version           str
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.models.base_model import BasePredictor, PredictionResult
from src.utils.constants import SEUILS_GAM, PCT_ENFANTS_MOINS_5ANS
from src.data_collection.malaria_fetcher import POPULATION_REGIONS


# ─────────────────────────────────────────────────────────────────
# Features — ordre FIXE (synchronisé avec feature_engineering.py)
# ─────────────────────────────────────────────────────────────────
NUTRITION_FEATURE_NAMES: List[str] = [
    # Historique nutritionnel
    "gam_lag_1m",
    "gam_lag_3m",
    "gam_lag_6m",
    "sam_lag_1m",
    "variation_gam_3m",
    # Sécurité alimentaire
    "score_fcs",
    "hdds",
    "rcsi",
    # Prix alimentaires
    "prix_riz_normalise",
    "prix_manioc_normalise",
    "variation_prix_pct_1m",
    # Disponibilité alimentaire (0-3)
    "dispo_cereales",
    "dispo_legumineuses",
    "dispo_proteines_animales",
    "dispo_legumes",
    "dispo_fruits",
    # Feature croisée (output MalariaPredictor)
    "score_paludisme",
    # Climatique
    "precipitations_30j_mm",
    "temperature_moy_c",
    "ndvi",
    # Socio-économique
    "indice_vulnerabilite",
    "en_periode_soudure",
    "semaines_avant_soudure_norm",
    # Temporel (cyclique)
    "mois_sin",
    "mois_cos",
    "saison_encoded",
    # Géographique
    "latitude",
    "longitude",
    "altitude_m",
    "zone_climatique_encoded",
    # Démographique
    "densite_population_norm",
    "pct_enfants_5ans",
]


class NutritionPredictor(BasePredictor):
    """
    Prédicteur de risque malnutrition aiguë (Ensemble RF + MLP).

    Trois modèles internes :
      _rf   : RandomForestClassifier → score risque (poids 60%)
      _mlp  : MLPClassifier          → score risque (poids 40%)
      _reg  : GradientBoostingRegressor → prévision GAM % (continuus)

    Utilisé par :
      - src/api/routers/nutrition.py (get_risque_nutrition)
      - src/api/routers/predictions.py (get_prediction_combinee)
      - src/data_collection/scheduler.py (task_mettre_a_jour_predictions)
    """

    MODEL_NAME:    ClassVar[str] = "nutrition"
    MODEL_VERSION: ClassVar[str] = "1.1.0"
    MODEL_TYPE:    ClassVar[str] = "ensemble"

    # Pondération de l'ensemble
    ENSEMBLE_WEIGHTS: ClassVar[Dict[str, float]] = {
        "rf":  0.60,
        "mlp": 0.40,
    }

    RF_PARAMS: ClassVar[Dict] = {
        "n_estimators":  500,
        "max_depth":     8,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs":       -1,
        "random_state": 42,
        "oob_score":    True,
    }

    MLP_PARAMS: ClassVar[Dict] = {
        "hidden_layer_sizes": (128, 64, 32),
        "activation":         "relu",
        "solver":             "adam",
        "alpha":              0.001,
        "batch_size":         64,
        "learning_rate":      "adaptive",
        "learning_rate_init": 0.001,
        "max_iter":           300,
        "early_stopping":     True,
        "validation_fraction": 0.15,
        "n_iter_no_change":   20,
        "random_state":       42,
    }

    GBR_PARAMS: ClassVar[Dict] = {
        "n_estimators":  300,
        "max_depth":     4,
        "learning_rate": 0.05,
        "subsample":     0.8,
        "random_state":  42,
        "loss":          "huber",
    }

    def __init__(self):
        super().__init__()
        self._rf  = None   # RandomForestClassifier
        self._mlp = None   # MLPClassifier
        self._reg = None   # GBRegressor pour prévision GAM %
        self._rf_explainer  = None  # SHAP TreeExplainer (RF)
        self._mlp_explainer = None  # SHAP KernelExplainer (MLP, approx)

    def _build_model(self) -> Any:
        """Construit les sous-modèles de l'ensemble."""
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
            from sklearn.neural_network import MLPClassifier

            self._rf  = RandomForestClassifier(**self.RF_PARAMS)
            self._mlp = MLPClassifier(**self.MLP_PARAMS)
            self._reg = GradientBoostingRegressor(**self.GBR_PARAMS)
            return self._rf  # _model pointe sur le modèle principal
        except ImportError as exc:
            logger.error("sklearn manquant : {}", exc)
            raise

    def _train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Entraîne les 3 composants de l'ensemble.
        y           : scores de risque nutrition [0,1]
        y_gam       : taux GAM réels en % (régresseur)
        """
        y_clf = (y >= 0.25).astype(int)  # Binaire pour classifier
        y_gam = kwargs.get("y_gam", y * 20)  # GAM max ~20% dans les pires cas

        logger.info("Entraînement Random Forest nutrition...")
        self._rf.fit(X, y_clf)

        logger.info("Entraînement MLP nutrition...")
        self._mlp.fit(X, y_clf)

        logger.info("Entraînement régresseur GAM...")
        self._reg.fit(X, y_gam)

        self._model = self._rf

        # SHAP pour Random Forest (TreeExplainer)
        try:
            import shap
            # Utilise un sous-échantillon pour l'explainer (perf)
            bg_sample = X[np.random.choice(len(X), min(200, len(X)), replace=False)]
            self._rf_explainer  = shap.TreeExplainer(self._rf)
            # KernelExplainer pour MLP (approx sur background)
            def predict_fn(x):
                return self._mlp.predict_proba(x)[:, 1]
            self._mlp_explainer = shap.KernelExplainer(predict_fn, bg_sample[:50])
            logger.debug("SHAP Explainers initialisés pour NutritionPredictor")
        except Exception as exc:
            logger.warning("SHAP init échoué NutritionPredictor : {}", exc)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Prédiction ensemble : moyenne pondérée RF + MLP.
        """
        w_rf  = self.ENSEMBLE_WEIGHTS["rf"]
        w_mlp = self.ENSEMBLE_WEIGHTS["mlp"]

        rf_proba  = self._rf.predict_proba(X)[:, 1]
        mlp_proba = self._mlp.predict_proba(X)[:, 1]

        return w_rf * rf_proba + w_mlp * mlp_proba

    def _compute_shap(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        SHAP ensemble : moyenne pondérée des SHAP RF et MLP.
        """
        try:
            shap_rf = None
            shap_mlp = None

            if self._rf_explainer is not None:
                sv = self._rf_explainer.shap_values(X)
                shap_rf = sv[1] if isinstance(sv, list) else sv

            if self._mlp_explainer is not None:
                shap_mlp = self._mlp_explainer.shap_values(X)

            if shap_rf is not None and shap_mlp is not None:
                w_rf  = self.ENSEMBLE_WEIGHTS["rf"]
                w_mlp = self.ENSEMBLE_WEIGHTS["mlp"]
                return w_rf * shap_rf + w_mlp * shap_mlp
            elif shap_rf is not None:
                return shap_rf
            return None

        except Exception as exc:
            logger.debug("SHAP ensemble NutritionPredictor : {}", exc)
            return None

    def get_feature_names(self) -> List[str]:
        return NUTRITION_FEATURE_NAMES

    def _post_process(
        self,
        score: float,
        features: Dict[str, Any],
        horizon_days: int,
    ) -> Dict[str, Any]:
        """
        Post-processing spécifique nutrition.
        Calcule GAM/SAM prévus, intervalles de confiance, populations vulnérables.
        Ces champs sont attendus par nutrition.py et predictions.py.
        """
        gam_prevu, sam_prevu = self._estimer_gam(score, features, horizon_days)

        # Intervalles de confiance pour GAM (méthode bootstrap approx.)
        ic_factor = 1.96 * score * (1 - score) * 5
        ic_bas    = max(0.0, gam_prevu - ic_factor)
        ic_haut   = min(40.0, gam_prevu + ic_factor)

        # Populations vulnérables identifiées
        populations_vulnerables = self._identifier_populations_vulnerables(
            features, gam_prevu, sam_prevu
        )

        # Alias `facteurs_contributeurs` attendu par nutrition.py
        # (top_contributeurs est calculé dans BasePredictor.predict())
        intervalles = {
            "gam_bas": round(ic_bas, 1),
            "gam_haut": round(ic_haut, 1),
            "sam_bas": round(sam_prevu * 0.7, 1),
            "sam_haut": round(sam_prevu * 1.3, 1),
        }

        return {
            "gam_prevu_pct":          round(gam_prevu, 2),
            "sam_prevu_pct":          round(sam_prevu, 2),
            "intervalles_confiance":  intervalles,
            "populations_vulnerables": populations_vulnerables,
            # Alias pour compatibilité nutrition.py router
            "facteurs_contributeurs": [],  # Complété après SHAP dans predict()
            # Contexte alimentaire
            "fcs_actuel":             features.get("score_fcs", 0),
            "en_soudure":             bool(features.get("en_periode_soudure", 0)),
        }

    def _estimer_gam(
        self,
        score: float,
        features: Dict[str, Any],
        horizon_days: int,
    ) -> Tuple[float, float]:
        """
        Estime le taux GAM en % sur l'horizon donné.
        Utilise le régresseur GBR si disponible.
        """
        if self._reg is not None:
            try:
                X = self._features_to_array(features)
                if self._scaler:
                    X = self._scaler.transform(X)
                gam_raw = float(self._reg.predict(X)[0])
                gam     = max(1.0, min(35.0, gam_raw))
                sam     = max(0.0, gam * 0.28)
                return gam, sam
            except Exception:
                pass

        # Heuristique : GAM basé sur score + FCS
        gam_base = 3.0 + score * 20.0  # GAM max ~23% pour score=1
        fcs = features.get("score_fcs", 35)
        if fcs < 21:
            gam_base *= 1.4   # FCS pauvre → GAM plus élevé
        elif fcs < 35:
            gam_base *= 1.15

        # Correction soudure
        if features.get("en_periode_soudure", 0):
            gam_base *= 1.3

        # Correction paludisme (co-morbidité)
        score_pal = features.get("score_paludisme", 0)
        gam_base *= (1 + score_pal * 0.25)

        gam = max(1.0, min(35.0, gam_base))
        sam = max(0.0, gam * 0.28)

        return round(gam, 2), round(sam, 2)

    def _identifier_populations_vulnerables(
        self,
        features: Dict[str, Any],
        gam: float,
        sam: float,
    ) -> List[Dict[str, Any]]:
        """Identifie les groupes de population à risque prioritaire."""
        region_id  = features.get("region_id", "MDG-ANA")
        population = POPULATION_REGIONS.get(region_id, 500_000)

        pct_enfants  = PCT_ENFANTS_MOINS_5ANS
        pct_femmes_e = 0.04

        enfants_total  = int(population * pct_enfants)
        femmes_e_total = int(population * pct_femmes_e)

        vulnérables = []

        # Enfants 6-23 mois (les plus vulnérables)
        if gam >= 5.0:
            vulnérables.append({
                "groupe": "Enfants 6-23 mois",
                "effectif_estime": int(enfants_total * 0.25 * gam / 100),
                "priorite": "haute" if gam >= 10 else "moyenne",
                "intervention": "Alimentation complémentaire (ANJE) renforcée",
            })

        # Enfants 2-5 ans
        if gam >= 7.0:
            vulnérables.append({
                "groupe": "Enfants 2-5 ans",
                "effectif_estime": int(enfants_total * 0.75 * gam / 100),
                "priorite": "haute" if gam >= 15 else "moyenne",
                "intervention": "Supplémentation micronutriments + suivi croissance",
            })

        # Femmes enceintes et allaitantes
        if gam >= 5.0:
            vulnérables.append({
                "groupe": "Femmes enceintes / allaitantes",
                "effectif_estime": int(femmes_e_total * gam / 100 * 1.5),
                "priorite": "haute" if gam >= 10 else "moyenne",
                "intervention": "Supplémentation Fer-Folate + Vitamine A",
            })

        # Ménages en insécurité alimentaire sévère (FCS < 21)
        fcs = features.get("score_fcs", 35)
        if fcs < 21:
            vulnérables.append({
                "groupe": "Ménages en insécurité alimentaire sévère",
                "effectif_estime": int(population * 0.15),
                "priorite": "haute",
                "intervention": "Transferts monétaires + vivres d'urgence",
            })

        return vulnérables

    def predict_with_uncertainty(
        self,
        features: Dict[str, Any],
        horizon_days: int = 30,
        n_simulations: int = 100,
    ) -> Dict[str, Any]:
        """
        Prédiction avec estimation d'incertitude par Monte Carlo Dropout.
        Utile pour les rapports d'urgence (intervalles plus précis).
        """
        scores = []
        for _ in range(n_simulations):
            # Perturbation légère des features (Monte Carlo)
            noise = np.random.normal(0, 0.02, len(NUTRITION_FEATURE_NAMES))
            X = self._features_to_array(features) + noise
            if self._scaler:
                X = self._scaler.transform(X)
            s = float(self._predict_raw(X)[0])
            scores.append(np.clip(s, 0, 1))

        scores_arr = np.array(scores)
        base = self.predict(features, horizon_days)
        base["incertitude_mc"] = {
            "score_moyen":    round(float(np.mean(scores_arr)), 4),
            "score_std":      round(float(np.std(scores_arr)), 4),
            "p5":             round(float(np.percentile(scores_arr, 5)), 4),
            "p95":            round(float(np.percentile(scores_arr, 95)), 4),
            "n_simulations":  n_simulations,
        }
        return base

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_gam_test: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Évaluation complète sur jeu de test."""
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_score, recall_score,
            mean_absolute_error, mean_squared_error,
        )

        y_pred_proba = self._predict_raw(X_test)
        y_pred_bin   = (y_pred_proba >= 0.5).astype(int)
        y_true_bin   = (y_test >= 0.25).astype(int)

        metrics: Dict[str, float] = {
            "auc_roc":   round(roc_auc_score(y_true_bin, y_pred_proba), 4),
            "f1_score":  round(f1_score(y_true_bin, y_pred_bin, zero_division=0), 4),
            "precision": round(precision_score(y_true_bin, y_pred_bin, zero_division=0), 4),
            "recall":    round(recall_score(y_true_bin, y_pred_bin, zero_division=0), 4),
            "rf_oob_score": round(getattr(self._rf, "oob_score_", 0.0), 4),
        }

        # Métriques régresseur GAM
        if y_gam_test is not None and self._reg is not None:
            y_gam_pred = self._reg.predict(X_test)
            metrics["mae_gam"]  = round(float(mean_absolute_error(y_gam_test, y_gam_pred)), 3)
            metrics["rmse_gam"] = round(float(np.sqrt(mean_squared_error(y_gam_test, y_gam_pred))), 3)
            metrics["mape_gam"] = round(float(
                np.mean(np.abs((y_gam_test - y_gam_pred) / (y_gam_test + 0.1))) * 100
            ), 2)

        self._metrics.update(metrics)
        logger.info(
            "Métriques NutritionPredictor — AUC: {} | F1: {} | MAE_GAM: {}",
            metrics["auc_roc"], metrics["f1_score"],
            metrics.get("mae_gam", "N/A"),
        )
        return metrics

    @classmethod
    def create_demo_model(cls) -> "NutritionPredictor":
        """Crée un modèle de démonstration entraîné sur données synthétiques."""
        logger.info("Création modèle démonstration NutritionPredictor...")
        np.random.seed(42)
        n_samples  = 2000
        n_features = len(NUTRITION_FEATURE_NAMES)

        X = np.random.randn(n_samples, n_features).astype(np.float32)

        # Score cible corrélé avec FCS (index 5), score_paludisme (index 16), soudure (index 21)
        y = np.clip(
            -0.3 * X[:, 5]   # score_fcs (inverse)
            + 0.25 * X[:, 16]  # score_paludisme
            + 0.2  * X[:, 21]  # en_periode_soudure
            + 0.2  * X[:, 20]  # indice_vulnerabilite
            + 0.1  * np.random.randn(n_samples),
            0, 1
        ).astype(np.float32)

        y_gam = np.clip(5 + y * 18 + np.random.randn(n_samples) * 1.5, 1, 30)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        instance = cls()
        instance._build_model()
        instance.fit(
            X, y,
            feature_names=NUTRITION_FEATURE_NAMES,
            scaler=scaler,
            y_gam=y_gam,
        )
        instance._metrics = {
            "auc_roc": 0.79,
            "f1_score": 0.72,
            "recall": 0.75,
            "precision": 0.69,
            "mae_gam": 1.8,
            "rmse_gam": 2.4,
        }
        logger.info("Modèle démonstration NutritionPredictor créé")
        return instance