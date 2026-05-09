"""
Interface abstraite commune à tous les modèles ML du projet.

Contrats imposés (utilisés par l'API, le scheduler et les tests) :
  - predict(features, horizon_days) → dict  (clés standardisées)
  - get_health_info()               → dict
  - load_latest()                   → classmethod
  - save(path)                      → None
  - ModelRegistry                   → gestion versioning + MLflow

Clés de sortie obligatoires du dict `predict()` :
  score_risque        float  [0,1]
  niveau_risque       str    faible|moyen|élevé|très élevé
  fiabilite_modele    float  [0,1]
  top_contributeurs   list   [{nom, valeur, shap_value, contribution_pct}]
  date_prediction     str    ISO datetime
  horizon_jours       int
  modele_version      str
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from config.settings import settings

T = TypeVar("T", bound="BasePredictor")


# ─────────────────────────────────────────────────────────────────
# Dataclass de résultat standardisé
# ─────────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """
    Structure de résultat normalisée retournée par tous les modèles.
    Garantit la compatibilité avec les routers FastAPI et le scheduler Celery.
    """
    # Identité
    region_id: str
    modele_nom: str
    modele_version: str
    date_prediction: datetime = Field(default_factory=datetime.utcnow)
    horizon_jours: int

    # Scores — OBLIGATOIRES pour les routers
    score_risque: float = Field(..., ge=0, le=1)
    niveau_risque: str  # faible | moyen | élevé | très élevé
    fiabilite_modele: float = Field(default=0.8, ge=0, le=1)

    # Explicabilité — OBLIGATOIRE pour UNICEF
    top_contributeurs: List[Dict[str, Any]] = Field(default_factory=list)

    # Champs optionnels spécifiques par modèle
    extra: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise en dict compatible JSON (router + cache Redis)."""
        d = self.model_dump()
        d["date_prediction"] = d["date_prediction"].isoformat()
        # Aplatit extra au niveau racine pour compatibilité routers existants
        d.update(d.pop("extra", {}))
        return d

    @staticmethod
    def niveau_from_score(score: float) -> str:
        """Convertit un score [0,1] en niveau textuel."""
        if score >= 0.75:
            return "très élevé"
        elif score >= 0.50:
            return "élevé"
        elif score >= 0.25:
            return "moyen"
        return "faible"


# ─────────────────────────────────────────────────────────────────
# Interface abstraite BasePredictor
# ─────────────────────────────────────────────────────────────────

class BasePredictor(abc.ABC):
    """
    Classe de base abstraite pour tous les modèles ML.

    Sous-classes obligées d'implémenter :
      _build_model()       — construction de l'architecture
      _train(X, y)         — entraînement
      _predict_raw(X)      — inférence brute (avant post-processing)
      _compute_shap(X)     — calcul valeurs SHAP
      get_feature_names()  — noms des features dans l'ordre du vecteur

    La méthode publique `predict(features, horizon_days)` est fournie
    par cette classe et garantit le format de sortie standardisé.
    """

    # Métadonnées de classe — surchargées par chaque sous-classe
    MODEL_NAME:    ClassVar[str] = "base"
    MODEL_VERSION: ClassVar[str] = "1.0.0"
    MODEL_TYPE:    ClassVar[str] = "classification"  # classification | regression

    # Seuils de risque (surchargeable par sous-classe)
    RISK_THRESHOLDS: ClassVar[Dict[str, float]] = {
        "faible":     0.25,
        "moyen":      0.50,
        "élevé":      0.75,
        "très élevé": 1.00,
    }

    def __init__(self):
        self._model = None          # Modèle sklearn/xgboost/keras sous-jacent
        self._scaler = None         # StandardScaler ou MinMaxScaler
        self._feature_names: List[str] = []
        self._trained_at: Optional[datetime] = None
        self._metrics: Dict[str, float] = {}
        self._drift_score: float = 0.0
        self._prediction_count: int = 0
        self._last_prediction_at: Optional[datetime] = None
        self._model_dir = Path(settings.ml.model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

    # ─── Méthodes abstraites ──────────────────────────────────────

    @abc.abstractmethod
    def _build_model(self) -> Any:
        """Instancie et retourne le modèle ML sous-jacent."""

    @abc.abstractmethod
    def _train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Entraîne le modèle sur les données X, y."""

    @abc.abstractmethod
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Retourne les scores bruts [0,1] pour chaque sample."""

    @abc.abstractmethod
    def _compute_shap(self, X: np.ndarray) -> np.ndarray:
        """Retourne les valeurs SHAP shape (n_samples, n_features)."""

    @abc.abstractmethod
    def get_feature_names(self) -> List[str]:
        """Retourne la liste ordonnée des noms de features."""

    # ─── Interface publique (contrats utilisés par l'API) ─────────

    def predict(
        self,
        features: Dict[str, Any],
        horizon_days: int = 14,
    ) -> Dict[str, Any]:
        """
        Méthode principale d'inférence.
        Appelée par :
          - src/api/routers/predictions.py
          - src/api/routers/malaria.py
          - src/api/routers/nutrition.py
          - src/data_collection/scheduler.py

        Args:
            features   : dict de features (clés = noms, valeurs = float/int)
            horizon_days : horizon de prédiction en jours

        Returns:
            dict avec clés garanties :
              score_risque, niveau_risque, fiabilite_modele,
              top_contributeurs, date_prediction, horizon_jours,
              modele_version + champs spécifiques par modèle
        """
        if self._model is None:
            raise RuntimeError(
                f"Modèle {self.MODEL_NAME} non entraîné. "
                "Appelez .fit() ou .load_latest() d'abord."
            )

        # 1. Vectorisation
        X = self._features_to_array(features)

        # 2. Scaling
        if self._scaler is not None:
            X = self._scaler.transform(X)

        # 3. Inférence brute
        raw_scores = self._predict_raw(X)
        score = float(np.clip(raw_scores[0], 0.0, 1.0))

        # 4. Ajustement horizon (score décroît légèrement avec l'horizon)
        score = self._adjust_for_horizon(score, horizon_days)

        # 5. Niveau de risque
        niveau = PredictionResult.niveau_from_score(score)

        # 6. SHAP top contributeurs
        top_contributeurs = self._get_top_contributeurs(X, features)

        # 7. Confiance (basée sur la proximité aux seuils)
        confiance = self._compute_confidence(score)

        # 8. Post-processing spécifique modèle
        extra = self._post_process(
            score=score,
            features=features,
            horizon_days=horizon_days,
        )

        # 9. Compteur de prédictions
        self._prediction_count += 1
        self._last_prediction_at = datetime.utcnow()

        result = PredictionResult(
            region_id=features.get("region_id", "unknown"),
            modele_nom=self.MODEL_NAME,
            modele_version=self.MODEL_VERSION,
            horizon_jours=horizon_days,
            score_risque=round(score, 4),
            niveau_risque=niveau,
            fiabilite_modele=round(confiance, 3),
            top_contributeurs=top_contributeurs,
            extra=extra,
        )

        return result.to_dict()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scaler=None,
        **kwargs,
    ) -> "BasePredictor":
        """
        Entraîne le modèle. Interface unifiée pour tous les sous-modèles.
        """
        logger.info(
            "Entraînement {} — {} samples, {} features",
            self.MODEL_NAME, X.shape[0], X.shape[1],
        )

        if self._model is None:
            self._model = self._build_model()

        if scaler is not None:
            self._scaler = scaler
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        self._feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        self._train(X_scaled, y, **kwargs)
        self._trained_at = datetime.utcnow()

        logger.info("Entraînement {} terminé", self.MODEL_NAME)
        return self

    def get_health_info(self) -> Dict[str, Any]:
        """
        Retourne les métriques de santé du modèle.
        Appelé par GET /api/v1/predictions/sante-modeles
        """
        return {
            "modele": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "date_entrainement": (
                self._trained_at.isoformat()
                if self._trained_at else None
            ),
            "metriques": self._metrics,
            "drift_score": round(self._drift_score, 4),
            "statut": self._compute_statut(),
            "nb_predictions_7j": self._prediction_count,
            "derniere_prediction": (
                self._last_prediction_at.isoformat()
                if self._last_prediction_at else None
            ),
            "nb_features": len(self._feature_names),
            "feature_names": self._feature_names[:10],  # top 10
        }

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Met à jour les métriques de performance (appelé après évaluation)."""
        self._metrics.update(metrics)

    def update_drift_score(self, psi: float) -> None:
        """Met à jour le score de dérive PSI (appelé par drift detection)."""
        self._drift_score = psi

    # ─── Persistance ─────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Sérialise le modèle en .pkl + métadonnées JSON.
        Appelé par les scripts de training et le retraining Celery.
        """
        if path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = self._model_dir / f"{self.MODEL_NAME}_{timestamp}.pkl"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "trained_at": self._trained_at,
            "metrics": self._metrics,
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Fichier de métadonnées (lisible sans pickle)
        meta_path = path.with_suffix(".json")
        meta = {
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "trained_at": self._trained_at.isoformat() if self._trained_at else None,
            "metrics": self._metrics,
            "nb_features": len(self._feature_names),
            "feature_names": self._feature_names,
            "file": str(path),
            "checksum": self._checksum(path),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Modèle {} sauvegardé → {}", self.MODEL_NAME, path)
        return path

    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        """Charge un modèle depuis un fichier .pkl."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier modèle introuvable : {path}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance = cls()
        instance._model         = payload.get("model")
        instance._scaler        = payload.get("scaler")
        instance._feature_names = payload.get("feature_names", [])
        instance._trained_at    = payload.get("trained_at")
        instance._metrics       = payload.get("metrics", {})

        logger.info(
            "Modèle {} chargé depuis {} (entraîné le {})",
            cls.MODEL_NAME,
            path.name,
            instance._trained_at,
        )
        return instance

    @classmethod
    def load_latest(cls: Type[T]) -> Optional[T]:
        """
        Charge la version la plus récente du modèle dans le répertoire models/.
        Utilisé par main.py au démarrage et par le scheduler Celery.
        Retourne None si aucun modèle trouvé (mode dégradé).
        """
        model_dir = Path(settings.ml.model_dir)
        pattern   = f"{cls.MODEL_NAME}_*.pkl"
        candidates = sorted(model_dir.glob(pattern), reverse=True)

        if not candidates:
            logger.warning(
                "Aucun modèle {} trouvé dans {}",
                cls.MODEL_NAME, model_dir
            )
            return None

        latest = candidates[0]
        try:
            return cls.load(latest)
        except Exception as exc:
            logger.error("Erreur chargement modèle {} : {}", latest, exc)
            # Essaie le suivant
            for fallback in candidates[1:]:
                try:
                    logger.info("Fallback modèle : {}", fallback)
                    return cls.load(fallback)
                except Exception:
                    continue
            return None

    # ─── Helpers internes ─────────────────────────────────────────

    def _features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convertit le dict de features en vecteur numpy ordonné.
        Les features manquantes sont imputées avec 0.0.
        """
        feat_names = self.get_feature_names()
        vector = np.array(
            [float(features.get(name, 0.0)) for name in feat_names],
            dtype=np.float32,
        ).reshape(1, -1)
        return vector

    def _adjust_for_horizon(self, score: float, horizon_days: int) -> float:
        """
        Ajuste le score selon l'horizon : l'incertitude augmente
        avec le temps (dégradation linéaire de 2% par semaine).
        """
        decay_per_week = 0.02
        weeks = horizon_days / 7
        adjusted = score * (1 - decay_per_week * max(0, weeks - 1))
        return float(np.clip(adjusted, 0.0, 1.0))

    def _get_top_contributeurs(
        self,
        X: np.ndarray,
        features: Dict[str, Any],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Calcule les top N features SHAP les plus influentes.
        Format compatible avec le router predictions.py.
        """
        try:
            shap_values = self._compute_shap(X)
            if shap_values is None:
                return self._top_contributeurs_fallback(features, top_n)

            # shap_values shape : (1, n_features)
            sv = np.abs(shap_values[0])
            top_indices = np.argsort(sv)[::-1][:top_n]
            feat_names  = self.get_feature_names()
            total       = np.sum(np.abs(shap_values[0])) + 1e-9

            return [
                {
                    "nom": feat_names[i] if i < len(feat_names) else f"f_{i}",
                    "valeur": float(features.get(
                        feat_names[i] if i < len(feat_names) else f"f_{i}", 0
                    )),
                    "shap_value": round(float(shap_values[0][i]), 4),
                    "contribution_pct": round(float(sv[i] / total * 100), 1),
                    "direction": "hausse" if shap_values[0][i] > 0 else "baisse",
                }
                for i in top_indices
            ]

        except Exception as exc:
            logger.debug("SHAP fallback pour {} : {}", self.MODEL_NAME, exc)
            return self._top_contributeurs_fallback(features, top_n)

    def _top_contributeurs_fallback(
        self, features: Dict[str, Any], top_n: int
    ) -> List[Dict[str, Any]]:
        """Fallback si SHAP indisponible : retourne les features par valeur absolue."""
        items = [
            {
                "nom": k,
                "valeur": float(v) if isinstance(v, (int, float)) else 0.0,
                "shap_value": 0.0,
                "contribution_pct": 0.0,
                "direction": "inconnu",
            }
            for k, v in features.items()
            if isinstance(v, (int, float)) and k != "region_id"
        ]
        items.sort(key=lambda x: abs(x["valeur"]), reverse=True)
        return items[:top_n]

    def _compute_confidence(self, score: float) -> float:
        """
        Confiance du modèle : maximale aux extrêmes (score proche de 0 ou 1),
        minimale au centre (score ≈ 0.5, zone d'incertitude).
        """
        distance_from_center = abs(score - 0.5) * 2  # 0 au centre, 1 aux extrêmes
        base_confidence = 0.6 + 0.35 * distance_from_center
        # Bonus si modèle récemment entraîné
        if self._trained_at:
            days_since_training = (datetime.utcnow() - self._trained_at).days
            freshness_penalty = min(0.15, days_since_training / 200)
            base_confidence -= freshness_penalty
        return float(np.clip(base_confidence, 0.5, 0.99))

    def _compute_statut(self) -> str:
        if self._drift_score > 0.15:
            return "retraining_requis"
        elif self._drift_score > 0.10:
            return "surveillance"
        return "optimal"

    @abc.abstractmethod
    def _post_process(
        self,
        score: float,
        features: Dict[str, Any],
        horizon_days: int,
    ) -> Dict[str, Any]:
        """
        Post-processing spécifique au modèle.
        Retourne les champs additionnels à merger dans le résultat.
        Ex : cas_prevus_14j pour malaria, gam_prevu_pct pour nutrition.
        """

    @staticmethod
    def _checksum(path: Path) -> str:
        """Calcule le SHA256 du fichier modèle pour intégrité."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────
# ModelRegistry — Gestion des versions et MLflow
# ─────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Registre centralisé des modèles ML.
    Interface avec MLflow pour le tracking des expériences.
    Gère la promotion des modèles (staging → production).
    """

    def __init__(self):
        self._registered: Dict[str, BasePredictor] = {}
        self._mlflow_enabled = self._check_mlflow()

    def _check_mlflow(self) -> bool:
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            return True
        except ImportError:
            logger.warning("MLflow non disponible — tracking désactivé")
            return False

    def register(self, model: BasePredictor, stage: str = "production") -> None:
        """Enregistre un modèle dans le registre."""
        self._registered[model.MODEL_NAME] = model
        if self._mlflow_enabled:
            self._log_to_mlflow(model, stage)
        logger.info(
            "Modèle {} v{} enregistré (stage={})",
            model.MODEL_NAME, model.MODEL_VERSION, stage
        )

    def get(self, model_name: str) -> Optional[BasePredictor]:
        return self._registered.get(model_name)

    def log_run(
        self,
        model: BasePredictor,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        artifacts: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Log une expérience MLflow et retourne le run_id."""
        if not self._mlflow_enabled:
            return None
        try:
            import mlflow
            mlflow.set_experiment(settings.mlflow_experiment_name)
            with mlflow.start_run(run_name=f"{model.MODEL_NAME}_{datetime.utcnow().date()}") as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_param("model_name", model.MODEL_NAME)
                mlflow.log_param("model_version", model.MODEL_VERSION)
                if artifacts:
                    for name, path in artifacts.items():
                        mlflow.log_artifact(path, artifact_path=name)
                return run.info.run_id
        except Exception as exc:
            logger.warning("MLflow log échoué : {}", exc)
            return None

    def _log_to_mlflow(self, model: BasePredictor, stage: str) -> None:
        if not self._mlflow_enabled:
            return
        try:
            import mlflow
            with mlflow.start_run(
                run_name=f"register_{model.MODEL_NAME}_{stage}"
            ):
                mlflow.log_param("stage", stage)
                mlflow.log_param("model_version", model.MODEL_VERSION)
                mlflow.log_metrics(model._metrics)
        except Exception as exc:
            logger.debug("MLflow register échoué : {}", exc)


# Instance globale du registre
model_registry = ModelRegistry()