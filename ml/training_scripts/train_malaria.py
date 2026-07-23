"""
ml/training_scripts/train_malaria.py
======================================
Script d'entraînement du modèle XGBoost de prédiction paludisme.

Interface publique (contrat avec scheduler.py) :
  train_malaria_model(date_debut, date_fin) → dict
    {
      "metriques":   {"auc_roc": float, "f1_score": float, ...},
      "model_path":  str,
      "version":     str,
      "run_id":      str,   # MLflow run ID
      "nb_samples":  int,
      "nb_features": int,
      "duree_sec":   float,
    }

Pipeline :
  1. Export données d'entraînement via FeatureEngineer
  2. Preprocessing (nettoyage, imputation, scaling)
  3. Split train/test (80/20 stratifié par région)
  4. Cross-validation 5-fold
  5. Optimisation hyperparamètres (Optuna si disponible)
  6. Entraînement final
  7. Évaluation + génération Model Card
  8. Sauvegarde modèle + log MLflow
  9. Déploiement si AUC ≥ seuil

Usage direct :
  python -m ml.training_scripts.train_malaria
  python -m ml.training_scripts.train_malaria --date-debut 2022-01-01 --date-fin 2024-01-01

─────────────────────────────────────────────────────────────────────
CORRECTIF (voir conversation) :
  _charger_donnees_malaria créait FeatureEngineer() SANS session DB
  (FeatureEngineer(db=None)) → _is_real_db() retournait toujours False
  → toutes les requêtes DB de feature_engineering.py court-circuitaient
  avec un résultat vide, AVANT même d'être exécutées. Résultat : 0
  échantillon réel à chaque entraînement, fallback synthétique
  silencieux — les AUC obtenus ne mesuraient que la capacité du modèle
  à réapprendre la formule synthétique elle-même, pas un vrai signal
  épidémiologique.

  Corrigé : on crée maintenant une vraie session SQLAlchemy async
  (asyncpg) à partir de settings.database.sync_url, et on la passe à
  FeatureEngineer(db=session, training_mode=True). training_mode=True
  désactive aussi les fallbacks réseau live (DHIS2/WHO GHO/NASA POWER)
  pendant l'entraînement — un entraînement batch ne doit jamais
  dépendre d'appels réseau par échantillon manquant.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger

# Ajout racine projet au path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ─────────────────────────────────────────────────────────────────
# Fonction principale — contrat avec scheduler.py
# ─────────────────────────────────────────────────────────────────

def train_malaria_model(
    date_debut: Optional[date] = None,
    date_fin:   Optional[date] = None,
    valider_seuil_auc: float = 0.70,
    sauvegarder: bool = True,
    optimiser_hyperparams: bool = False,
) -> Dict[str, Any]:
    """
    Entraîne le modèle XGBoost de prédiction paludisme.

    Args:
        date_debut            : Début de la période d'entraînement
        date_fin              : Fin de la période d'entraînement
        valider_seuil_auc     : AUC minimum pour valider le modèle (défaut 0.70)
        sauvegarder           : Si True → sauvegarde le modèle sur disque
        optimiser_hyperparams : Si True → optimisation Optuna (plus lent)

    Returns:
        dict avec clés "metriques", "model_path", "version", "run_id",
        "nb_samples", "nb_features", "duree_sec"
    """
    debut_total = time.time()

    date_fin   = date_fin   or date.today()
    date_debut = date_debut or date_fin - timedelta(days=730)

    logger.info(
        "🧠 Début entraînement MalariaPredictor — période {} → {}",
        date_debut, date_fin
    )

    result: Dict[str, Any] = {
        "metriques":   {},
        "model_path":  None,
        "version":     _generate_version(),
        "run_id":      None,
        "nb_samples":  0,
        "nb_features": 0,
        "duree_sec":   0.0,
        "valide":      False,
    }

    try:
        # ── 1. Chargement des données ──────────────────────────────
        logger.info("📥 Chargement des données d'entraînement...")
        X, y, feature_names, meta = _charger_donnees_malaria(date_debut, date_fin)

        if X is None or len(X) == 0:
            logger.warning("⚠️  Données insuffisantes — utilisation données synthétiques")
            X, y, feature_names = _generer_donnees_synthetiques_malaria()
            meta = []

        result["nb_samples"]  = len(X)
        result["nb_features"] = X.shape[1] if len(X) > 0 else 0

        logger.info(
            "✅ Données chargées : {} samples × {} features",
            result["nb_samples"], result["nb_features"]
        )

        # ── 2. Split Train / Test ──────────────────────────────────
        X_train, X_test, y_train, y_test = _train_test_split_malaria(
            X, y, test_size=0.20, meta=meta
        )
        logger.info(
            "Split — Train: {} | Test: {}",
            len(X_train), len(X_test)
        )

        # ── 3. Preprocessing ──────────────────────────────────────
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # ── 4. Optimisation hyperparamètres (optionnel) ────────────
        xgb_params = {}
        if optimiser_hyperparams:
            xgb_params = _optimiser_hyperparams_malaria(X_train, y_train, scaler)
            logger.info("✅ Hyperparamètres optimisés : {}", xgb_params)

        # ── 5. Entraînement ───────────────────────────────────────
        logger.info("🏋️  Entraînement MalariaPredictor...")
        from src.models.malaria_predictor import MalariaPredictor

        model = MalariaPredictor()
        if xgb_params:
            model.XGB_CLF_PARAMS.update(xgb_params)

        # Labels binaires pour le classifier + continus pour le régresseur
        y_clf = (y_train >= 0.25).astype(int)
        y_reg = y_train * 100

        # Split validation pour early stopping XGBoost
        n_val      = max(50, int(len(X_train) * 0.15))
        X_val      = X_train[-n_val:]
        y_val_clf  = y_clf[-n_val:]
        X_train_f  = X_train[:-n_val]
        y_clf_f    = y_clf[:-n_val]
        y_reg_f    = y_reg[:-n_val]

        model.fit(
            scaler.fit_transform(X_train_f),
            (y_clf_f > 0).astype(float),
            feature_names=list(feature_names),
            scaler=None,          # Scaler déjà appliqué
            y_clf=y_clf_f,
            y_reg=y_reg_f,
            X_val=scaler.transform(X_val),
            y_val_clf=y_val_clf,
        )
        # Réassignation du scaler pour inférence
        model._scaler = scaler

        # ── 6. Évaluation ─────────────────────────────────────────
        logger.info("📊 Évaluation sur jeu de test...")
        X_test_scaled = scaler.transform(X_test)
        try:
            metriques = model.evaluate(X_test_scaled, y_test)
        except ValueError as exc:
            logger.warning(
                "Évaluation impossible ({}) — jeu de test probablement mono-classe "
                "malgré le split par blocs. Vérifie la distribution de y_test.",
                exc
            )
            metriques = {"auc_roc": 0.0, "erreur_evaluation": str(exc)}

        # Cross-validation 5-fold (chronologique)
        cv_scores = _cross_validate_malaria(model, X, y, feature_names, scaler, meta=meta)
        metriques["cv_auc_mean"] = round(float(np.mean(cv_scores)), 4)
        metriques["cv_auc_std"]  = round(float(np.std(cv_scores)),  4)

        result["metriques"] = metriques
        logger.info(
            "Métriques — AUC: {} | F1: {} | CV_AUC: {} ± {}",
            metriques.get("auc_roc"),
            metriques.get("f1_score"),
            metriques.get("cv_auc_mean"),
            metriques.get("cv_auc_std"),
        )

        # ── 7. Validation seuil ───────────────────────────────────
        auc = metriques.get("auc_roc", 0)
        result["valide"] = auc >= valider_seuil_auc

        if not result["valide"]:
            logger.warning(
                "⚠️  Modèle rejeté — AUC={:.3f} < seuil={:.2f}",
                auc, valider_seuil_auc
            )
        else:
            logger.info("✅ Modèle validé — AUC={:.3f}", auc)

        # ── 8. Sauvegarde ─────────────────────────────────────────
        if sauvegarder and result["valide"]:
            model_path = model.save()
            result["model_path"] = str(model_path)
            logger.info("💾 Modèle sauvegardé → {}", model_path)

            # Génération Model Card
            _generer_model_card_malaria(model, metriques, date_debut, date_fin)

        elif sauvegarder and not result["valide"]:
            logger.info("⏭️  Sauvegarde ignorée (modèle non validé)")

        # ── 9. Log MLflow ─────────────────────────────────────────
        run_id = _log_mlflow_malaria(
            model=model,
            metriques=metriques,
            params={
                "date_debut":       str(date_debut),
                "date_fin":         str(date_fin),
                "nb_samples":       result["nb_samples"],
                "nb_features":      result["nb_features"],
                "test_size":        0.20,
                "valider_seuil_auc": valider_seuil_auc,
            },
            model_path=result.get("model_path"),
        )
        result["run_id"] = run_id

    except Exception as exc:
        logger.exception("❌ Erreur entraînement malaria : {}", exc)
        result["erreur"] = str(exc)

    result["duree_sec"] = round(time.time() - debut_total, 2)
    logger.info(
        "🏁 Entraînement terminé en {:.1f}s — valide={}",
        result["duree_sec"], result["valide"]
    )
    return result


# ─────────────────────────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────────────────────────

async def _build_dataset_malaria(
    date_debut: date,
    date_fin: date,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Ouvre une vraie session SQLAlchemy async, construit le dataset
    d'entraînement malaria via FeatureEngineer, puis ferme proprement
    la connexion. Isolé dans sa propre coroutine pour un seul appel
    asyncio.run() au niveau de _charger_donnees_malaria.
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from src.preprocessing.feature_engineering import FeatureEngineer
    from src.utils.constants import REGIONS_MADAGASCAR
    from config.settings import settings

    # settings.database.sync_url est du type postgresql://... (driver psycopg2).
    # SQLAlchemy async a besoin du driver asyncpg — on bascule juste le schéma.
    async_url = settings.database.sync_url.replace(
        "postgresql://", "postgresql+asyncpg://", 1
    )
    engine = create_async_engine(async_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with SessionLocal() as session:
            engineer = FeatureEngineer(db=session, training_mode=True)
            X, y, feature_names, meta = await engineer.build_training_dataset(
                region_ids=list(REGIONS_MADAGASCAR),
                date_debut=date_debut,
                date_fin=date_fin,
                modele="malaria",
            )
        return X, y, feature_names, meta
    finally:
        await engine.dispose()


def _charger_donnees_malaria(
    date_debut: date,
    date_fin: date,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], list, list]:
    """
    Charge les données d'entraînement depuis la DB via FeatureEngineer.
    Retourne (X, y, feature_names, meta) ou (None, None, [], []) si échec.
    """
    try:
        X, y, feature_names, meta = asyncio.run(
            _build_dataset_malaria(date_debut, date_fin)
        )

        if len(X) < 50:
            logger.warning(
                "Données réelles insuffisantes ({} samples) — complétion synthétique",
                len(X)
            )
            X_syn, y_syn, fn_syn = _generer_donnees_synthetiques_malaria()
            if len(X) > 0:
                X = np.vstack([X, X_syn])
                y = np.concatenate([y, y_syn])
            else:
                X, y, feature_names = X_syn, y_syn, fn_syn

        return X, y, feature_names, meta

    except Exception as exc:
        logger.warning("Chargement données réelles échoué : {} — données synthétiques", exc)
        return None, None, [], []


def _generer_donnees_synthetiques_malaria() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Génère un dataset synthétique réaliste pour l'entraînement en l'absence de données réelles.
    Corrélations calquées sur l'épidémiologie du paludisme à Madagascar.
    """
    from src.models.malaria_predictor import MALARIA_FEATURE_NAMES

    np.random.seed(42)
    n = 3000
    n_feat = len(MALARIA_FEATURE_NAMES)

    X = np.random.randn(n, n_feat).astype(np.float32)

    # Feature indices
    idx = {name: i for i, name in enumerate(MALARIA_FEATURE_NAMES)}

    # Corrélations réalistes
    y = (
        0.25 * np.clip(X[:, idx["precipitations_30j_mm"]], -2, 2) / 2  # Pluies
        + 0.20 * np.clip(X[:, idx["ndvi"]], -1, 1) / 2                 # Végétation
        + 0.15 * np.clip(X[:, idx["cas_lag_1sem"]], -2, 2) / 2         # Auto-corrélation
        + 0.10 * np.where(X[:, idx["saison_encoded"]] > 0, 0.2, -0.1)  # Saison
        + 0.10 * np.clip(X[:, idx["humidite_moy_pct"]], -2, 2) / 2     # Humidité
        + 0.10 * np.clip(X[:, idx["endemicite_encoded"]], -1, 2) / 2   # Endémicité
        + 0.10 * np.random.randn(n) * 0.1                               # Bruit
    )
    y = np.clip(y + 0.35, 0, 1).astype(np.float32)

    # Saisonnalité (Nov-Avr = saison pluies → risque plus élevé)
    mois_idx = np.random.randint(1, 13, n)
    saison_bonus = np.where(
        np.isin(mois_idx, [11, 12, 1, 2, 3, 4]), 0.15, -0.05
    )
    y = np.clip(y + saison_bonus, 0, 1).astype(np.float32)

    # Bruit régional (hautes terres moins risquées)
    alt_idx = idx.get("altitude_m", 12)
    alt_penalty = np.where(X[:, alt_idx] > 0, -0.1, 0.05)
    y = np.clip(y + alt_penalty, 0, 1).astype(np.float32)

    logger.info("Données synthétiques malaria générées : {} samples", n)
    return X, y, MALARIA_FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────
# Split et cross-validation
# ─────────────────────────────────────────────────────────────────

def _train_test_split_malaria(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    meta: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split par BLOCS mélangés (région + année-mois) — pas un split
    aléatoire ligne par ligne, pas un pur découpage chronologique.

    CORRIGÉ (voir conversation, 2e itération) : un split purement
    chronologique (train=passé, test=futur) empêche bien la fuite entre
    blocs WHO GHO à valeur constante, mais expose un autre problème sur
    ces données : la période de test peut retomber entièrement dans un
    seul bloc constant → une seule classe dans y_test → roc_auc_score
    indéfini (ValueError).

    On regroupe donc les échantillons par bloc (region_id, année-mois),
    puis on répartit des BLOCS ENTIERS (pas des lignes individuelles)
    aléatoirement entre train/test. Ça conserve la propriété qui compte
    (jamais deux lignes du même bloc constant des deux côtés à la fois)
    tout en dispersant les blocs "haut risque" et "bas risque" à travers
    tout le train ET tout le test, donc les deux classes restent
    représentées.

    Si `meta` absent ou incomplet, retombe sur un split aléatoire simple
    (avec avertissement — perte de la garantie anti-fuite par bloc).
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    n = len(X)
    if not meta or len(meta) != n:
        logger.warning(
            "Pas de métadonnées (region/date) pour grouper par bloc — "
            "split aléatoire simple, risque de fuite intra-bloc"
        )
        return train_test_split(X, y, test_size=test_size, random_state=42)

    # Bloc = région + année-mois (proxy raisonnable pour "même valeur WHO GHO")
    groups = np.array([
        f"{m.get('region_id', 'na')}_{str(m.get('date', ''))[:7]}"
        for m in meta
    ])

    y_bins = np.digitize(y, bins=[0.25])  # binaire, cohérent avec le seuil du classifieur

    # Plusieurs tentatives avec des graines différentes pour retomber sur
    # un split où les deux classes sont représentées côté test.
    for seed in range(10):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y_bins, groups=groups))
        if len(set(y_bins[test_idx])) >= 2 and len(set(y_bins[train_idx])) >= 2:
            logger.debug("Split par blocs réussi (seed={})", seed)
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    logger.warning(
        "Impossible d'obtenir les 2 classes des deux côtés en groupant par bloc "
        "après 10 tentatives — split aléatoire simple en dernier recours "
        "(risque de fuite intra-bloc, à interpréter avec prudence)"
    )
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y_bins)


def _cross_validate_malaria(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    scaler,
    n_splits: int = 5,
    meta: Optional[list] = None,
) -> np.ndarray:
    """
    Cross-validation par blocs (région + année-mois) — même logique que
    _train_test_split_malaria : GroupKFold garantit qu'un bloc constant
    n'est jamais partagé entre train et validation sur un même fold, sans
    forcer un découpage strictement chronologique qui peut isoler une
    seule classe dans un fold sur ce type de données.
    """
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from src.models.malaria_predictor import MalariaPredictor

    n = len(X)
    if meta and len(meta) == n:
        groups = np.array([
            f"{m.get('region_id', 'na')}_{str(m.get('date', ''))[:7]}"
            for m in meta
        ])
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(X, y, groups=groups))
    else:
        logger.warning("Pas de métadonnées pour grouper la CV — StratifiedKFold classique")
        y_bins = np.digitize(y, bins=[0.25, 0.50, 0.75])
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(X, y_bins))

    scores = []

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        try:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            y_true = (y_val >= 0.25).astype(int)
            if len(set(y_true)) < 2:
                logger.debug("CV Fold {} : une seule classe dans le fold, ignoré", fold)
                continue

            # Nouveau scaler par fold
            from sklearn.preprocessing import StandardScaler
            sc      = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_val_sc = sc.transform(X_val)

            # Entraînement fold
            fold_model = MalariaPredictor()
            fold_model._build_model()
            fold_model.fit(
                X_tr_sc, y_tr,
                feature_names=list(feature_names),
                y_clf=(y_tr >= 0.25).astype(int),
            )

            y_pred = fold_model._predict_raw(X_val_sc)
            auc = roc_auc_score(y_true, y_pred)
            scores.append(auc)
            logger.debug("CV Fold {} — AUC: {:.4f}", fold, auc)

        except Exception as exc:
            logger.warning("CV Fold {} échoué : {}", fold, exc)

    return np.array(scores) if scores else np.array([0.5])


# ─────────────────────────────────────────────────────────────────
# Optimisation hyperparamètres
# ─────────────────────────────────────────────────────────────────

def _optimiser_hyperparams_malaria(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """
    Optimisation hyperparamètres XGBoost via Optuna.
    Retourne les meilleurs paramètres trouvés.
    """
    try:
        import optuna
        from sklearn.model_selection import cross_val_score
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X_sc = scaler.fit_transform(X_train)
        y_cl = (y_train >= 0.25).astype(int)

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma":            trial.suggest_float("gamma", 0, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 0, 2.0),
                "use_label_encoder": False,
                "eval_metric":      "auc",
                "random_state":     42,
                "n_jobs":           -1,
            }
            clf    = XGBClassifier(**params)
            scores = cross_val_score(clf, X_sc, y_cl, cv=3, scoring="roc_auc", n_jobs=-1)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        logger.info(
            "Optuna terminé — meilleur AUC: {:.4f}",
            study.best_value
        )
        return study.best_params

    except ImportError:
        logger.info("Optuna non disponible — paramètres par défaut")
        return {}
    except Exception as exc:
        logger.warning("Optimisation HP échouée : {}", exc)
        return {}


# ─────────────────────────────────────────────────────────────────
# Logging MLflow et Model Card
# ─────────────────────────────────────────────────────────────────

def _log_mlflow_malaria(
    model,
    metriques: Dict[str, float],
    params: Dict[str, Any],
    model_path: Optional[str],
) -> Optional[str]:
    """Log l'expérience dans MLflow. Retourne le run_id."""
    try:
        import mlflow
        from config.settings import settings

        mlflow.set_tracking_uri(settings.ml.mlflow_tracking_uri)
        mlflow.set_experiment(f"{settings.ml.mlflow_experiment_name}-malaria")

        with mlflow.start_run(run_name=f"malaria_{model.MODEL_VERSION}") as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metriques)

            if model_path:
                mlflow.log_artifact(model_path, artifact_path="model")

            mlflow.set_tag("modele", "MalariaPredictor")
            mlflow.set_tag("framework", "XGBoost")
            mlflow.set_tag("version", model.MODEL_VERSION)

            run_id = run.info.run_id
            logger.info("MLflow run_id={}", run_id)
            return run_id

    except Exception as exc:
        logger.debug("MLflow log malaria : {}", exc)
        return None


def _generer_model_card_malaria(
    model,
    metriques: Dict[str, float],
    date_debut: date,
    date_fin: date,
) -> None:
    """Génère la Model Card UNICEF pour le modèle malaria."""
    try:
        from src.models.explainability import SHAPExplainer
        from pathlib import Path

        explainer  = SHAPExplainer(model)
        card       = explainer.generate_model_card()
        card["performances"] = metriques
        card["donnees_entrainement"]["periode"] = f"{date_debut} → {date_fin}"

        card_path = Path("docs/model_cards/malaria_predictor_card.json")
        card_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with card_path.open("w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)

        logger.info("Model Card malaria → {}", card_path)
    except Exception as exc:
        logger.debug("Model Card malaria : {}", exc)


def _generate_version() -> str:
    """Génère un numéro de version basé sur la date."""
    return f"1.{date.today().strftime('%Y%m%d')}"


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement MalariaPredictor XGBoost"
    )
    parser.add_argument(
        "--date-debut",
        type=lambda s: date.fromisoformat(s),
        default=date.today() - timedelta(days=730),
        help="Date début période d'entraînement (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-fin",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
        help="Date fin période d'entraînement (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--seuil-auc",
        type=float,
        default=0.70,
        help="AUC minimum pour valider le modèle",
    )
    parser.add_argument(
        "--optimiser-hp",
        action="store_true",
        help="Activer l'optimisation Optuna des hyperparamètres",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Entraîner sans sauvegarder (test uniquement)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()

    args = _parse_args()
    result = train_malaria_model(
        date_debut=args.date_debut,
        date_fin=args.date_fin,
        valider_seuil_auc=args.seuil_auc,
        sauvegarder=not args.dry_run,
        optimiser_hyperparams=args.optimiser_hp,
    )

    print("\n" + "=" * 60)
    print("RÉSULTATS ENTRAÎNEMENT MALARIA")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")

    sys.exit(0 if result.get("valide") else 1)