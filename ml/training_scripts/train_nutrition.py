"""
Script d'entraînement du modèle Ensemble (RF + MLP) de prédiction nutrition.

Interface publique (contrat avec scheduler.py) :
  train_nutrition_model(date_debut, date_fin) → dict
    {
      "metriques":   {"auc_roc": float, "f1_score": float, "mae_gam": float, ...},
      "model_path":  str,
      "version":     str,
      "run_id":      str,
      "nb_samples":  int,
      "nb_features": int,
      "duree_sec":   float,
      "valide":      bool,
    }

Pipeline :
  1. Export données (FeatureEngineer)
  2. Preprocessing + scaling
  3. Entraînement RF + MLP (parallèle)
  4. Calibration des probabilités
  5. Évaluation ensemble
  6. Cross-validation 5-fold
  7. Sauvegarde + MLflow + Model Card

Usage direct :
  python -m ml.training_scripts.train_nutrition
  python -m ml.training_scripts.train_nutrition --date-debut 2022-01-01
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ─────────────────────────────────────────────────────────────────
# Fonction principale — contrat avec scheduler.py
# ─────────────────────────────────────────────────────────────────

def train_nutrition_model(
    date_debut: Optional[date] = None,
    date_fin:   Optional[date] = None,
    valider_seuil_auc: float = 0.70,
    sauvegarder: bool = True,
    optimiser_hyperparams: bool = False,
) -> Dict[str, Any]:
    """
    Entraîne le modèle Ensemble (RF + MLP) de prédiction malnutrition.

    Returns:
        dict avec clés "metriques", "model_path", "version", "run_id",
        "nb_samples", "nb_features", "duree_sec", "valide"
    """
    debut_total = time.time()

    date_fin   = date_fin   or date.today()
    date_debut = date_debut or date_fin - timedelta(days=730)

    logger.info(
        "🧠 Début entraînement NutritionPredictor — période {} → {}",
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
        logger.info("📥 Chargement des données nutrition...")
        X, y, y_gam, feature_names, meta = _charger_donnees_nutrition(date_debut, date_fin)

        if X is None or len(X) == 0:
            logger.warning("⚠️  Données insuffisantes — utilisation données synthétiques")
            X, y, y_gam, feature_names = _generer_donnees_synthetiques_nutrition()
            meta = []

        result["nb_samples"]  = len(X)
        result["nb_features"] = X.shape[1] if len(X) > 0 else 0

        logger.info(
            "✅ Données chargées : {} samples × {} features | GAM moyen: {:.1f}%",
            result["nb_samples"], result["nb_features"],
            float(np.mean(y_gam)) if y_gam is not None else 0
        )

        # ── 2. Split Train / Test ──────────────────────────────────
        X_train, X_test, y_train, y_test, y_gam_train, y_gam_test = \
            _train_test_split_nutrition(X, y, y_gam, test_size=0.20)

        logger.info(
            "Split — Train: {} | Test: {}",
            len(X_train), len(X_test)
        )

        # ── 3. Scaling ────────────────────────────────────────────
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # ── 4. Entraînement ───────────────────────────────────────
        logger.info("🏋️  Entraînement NutritionPredictor (RF + MLP)...")
        from src.models.nutrition_predictor import NutritionPredictor

        model = NutritionPredictor()
        model._build_model()

        X_tr_sc = scaler.fit_transform(X_train)
        model.fit(
            X_tr_sc,
            y_train,
            feature_names=list(feature_names),
            scaler=None,
            y_gam=y_gam_train,
        )
        model._scaler = scaler

        # ── 5. Calibration des probabilités ───────────────────────
        logger.info("🎯 Calibration des probabilités...")
        model = _calibrer_probabilites(model, X_train, y_train, scaler)

        # ── 6. Évaluation ─────────────────────────────────────────
        logger.info("📊 Évaluation sur jeu de test...")
        X_test_sc = scaler.transform(X_test)
        metriques = model.evaluate(X_test_sc, y_test, y_gam_test=y_gam_test)

        # Cross-validation 5-fold
        cv_scores = _cross_validate_nutrition(model, X, y, feature_names, scaler)
        metriques["cv_auc_mean"] = round(float(np.mean(cv_scores)), 4)
        metriques["cv_auc_std"]  = round(float(np.std(cv_scores)),  4)

        # Analyse par région si métadonnées disponibles
        if meta:
            regional_metrics = _evaluer_par_region(model, X_test_sc, y_test, meta)
            metriques["regional"] = regional_metrics

        result["metriques"] = metriques
        logger.info(
            "Métriques — AUC: {} | F1: {} | MAE_GAM: {} | CV_AUC: {} ± {}",
            metriques.get("auc_roc"),
            metriques.get("f1_score"),
            metriques.get("mae_gam"),
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
            _generer_model_card_nutrition(model, metriques, date_debut, date_fin)

        # ── 9. Log MLflow ─────────────────────────────────────────
        run_id = _log_mlflow_nutrition(
            model=model,
            metriques=metriques,
            params={
                "date_debut":       str(date_debut),
                "date_fin":         str(date_fin),
                "nb_samples":       result["nb_samples"],
                "nb_features":      result["nb_features"],
                "test_size":        0.20,
                "ensemble_rf_poids": 0.60,
                "ensemble_mlp_poids": 0.40,
            },
            model_path=result.get("model_path"),
        )
        result["run_id"] = run_id

    except Exception as exc:
        logger.exception("❌ Erreur entraînement nutrition : {}", exc)
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

def _charger_donnees_nutrition(
    date_debut: date,
    date_fin: date,
) -> Tuple:
    """Charge les données nutrition via FeatureEngineer."""
    try:
        from src.preprocessing.feature_engineering import FeatureEngineer
        from src.utils.constants import REGIONS_MADAGASCAR

        engineer = FeatureEngineer()
        X, y, feature_names, meta = asyncio.run(
            engineer.build_training_dataset(
                region_ids=list(REGIONS_MADAGASCAR),
                date_debut=date_debut,
                date_fin=date_fin,
                modele="nutrition",
            )
        )

        # Labels GAM réels (en %) pour le régresseur
        y_gam = _extraire_labels_gam(meta, y)

        if len(X) < 50:
            logger.warning(
                "Données réelles insuffisantes ({} samples) — complétion synthétique",
                len(X)
            )
            X_s, y_s, y_gam_s, fn_s = _generer_donnees_synthetiques_nutrition()
            if len(X) > 0:
                X    = np.vstack([X, X_s])
                y    = np.concatenate([y, y_s])
                y_gam = np.concatenate([y_gam, y_gam_s])
            else:
                X, y, y_gam, feature_names = X_s, y_s, y_gam_s, fn_s

        return X, y, y_gam, feature_names, meta

    except Exception as exc:
        logger.warning("Chargement données nutrition échoué : {} — synthétiques", exc)
        return None, None, None, [], []


def _extraire_labels_gam(meta: list, y_norm: np.ndarray) -> np.ndarray:
    """Extrait les valeurs GAM réelles depuis les métadonnées ou dénormalise."""
    # y_norm est normalisé [0,1] → GAM = y_norm * 20 (GAM max ~20%)
    return y_norm * 20 if y_norm is not None else np.zeros(len(y_norm))


def _generer_donnees_synthetiques_nutrition() -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Génère un dataset synthétique nutrition réaliste.
    Basé sur les corrélations épidémiologiques Madagascar.
    """
    from src.models.nutrition_predictor import NUTRITION_FEATURE_NAMES

    np.random.seed(43)
    n      = 3000
    n_feat = len(NUTRITION_FEATURE_NAMES)
    idx    = {name: i for i, name in enumerate(NUTRITION_FEATURE_NAMES)}

    X = np.random.randn(n, n_feat).astype(np.float32)

    # Score de risque nutrition
    y = (
        -0.30 * np.clip(X[:, idx["score_fcs"]], -2, 2) / 2          # FCS (inverse)
        + 0.25 * np.clip(X[:, idx["indice_vulnerabilite"]], -1, 2) / 2  # Vulnérabilité
        + 0.20 * np.clip(X[:, idx["score_paludisme"]], -1, 2) / 2    # Co-morbidité
        + 0.15 * X[:, idx["en_periode_soudure"]] * 0.3               # Soudure
        + 0.10 * np.clip(X[:, idx["variation_prix_pct_1m"]], -1, 2)  # Prix
        + 0.10 * np.random.randn(n) * 0.05                           # Bruit
    )
    y = np.clip(y + 0.35, 0, 1).astype(np.float32)

    # GAM réel en % (corrélé à y mais avec plus de variance)
    y_gam = np.clip(
        3.0 + y * 18.0 + np.random.randn(n) * 1.5,
        0.5, 30.0
    ).astype(np.float32)

    # Grand Sud (vulnérabilité haute) → GAM plus élevé
    zone_idx = idx.get("zone_climatique_encoded", 29)
    grand_sud_mask = X[:, zone_idx] > 1.0
    y_gam[grand_sud_mask] = np.clip(y_gam[grand_sud_mask] * 1.4, 0, 30)
    y[grand_sud_mask]     = np.clip(y[grand_sud_mask] * 1.3, 0, 1)

    logger.info("Données synthétiques nutrition générées : {} samples", n)
    return X, y, y_gam, NUTRITION_FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────
# Split, cross-validation, calibration
# ─────────────────────────────────────────────────────────────────

def _train_test_split_nutrition(
    X: np.ndarray,
    y: np.ndarray,
    y_gam: np.ndarray,
    test_size: float = 0.20,
) -> Tuple:
    """Split stratifié pour nutrition (stratification sur bins GAM)."""
    from sklearn.model_selection import train_test_split

    y_bins = np.digitize(y, bins=[0.25, 0.50, 0.75])
    try:
        (X_tr, X_te,
         y_tr, y_te,
         yg_tr, yg_te) = train_test_split(
            X, y, y_gam,
            test_size=test_size,
            random_state=42,
            stratify=y_bins,
        )
    except ValueError:
        (X_tr, X_te,
         y_tr, y_te,
         yg_tr, yg_te) = train_test_split(
            X, y, y_gam,
            test_size=test_size,
            random_state=42,
        )
    return X_tr, X_te, y_tr, y_te, yg_tr, yg_te


def _cross_validate_nutrition(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    scaler,
    n_splits: int = 5,
) -> np.ndarray:
    """Cross-validation 5-fold — retourne les scores AUC par fold."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from src.models.nutrition_predictor import NutritionPredictor
    from sklearn.preprocessing import StandardScaler

    y_bins = np.digitize(y, bins=[0.25, 0.50, 0.75])
    kf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=43)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_bins), 1):
        try:
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            sc       = StandardScaler()
            X_tr_sc  = sc.fit_transform(X_tr)
            X_val_sc = sc.transform(X_val)

            fold_model = NutritionPredictor()
            fold_model._build_model()
            fold_model.fit(
                X_tr_sc, y_tr,
                feature_names=list(feature_names),
            )

            y_pred = fold_model._predict_raw(X_val_sc)
            y_true = (y_val >= 0.25).astype(int)
            auc    = roc_auc_score(y_true, y_pred)
            scores.append(auc)
            logger.debug("CV Fold {} nutrition — AUC: {:.4f}", fold, auc)

        except Exception as exc:
            logger.warning("CV Fold {} nutrition échoué : {}", fold, exc)
            scores.append(0.5)

    return np.array(scores)


def _calibrer_probabilites(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler,
):
    """
    Calibre les probabilités du modèle via CalibratedClassifierCV.
    Améliore la fiabilité des scores de risque pour les décisions UNICEF.
    """
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.preprocessing import StandardScaler

        X_sc  = scaler.transform(X_train)
        y_bin = (y_train >= 0.25).astype(int)

        # Calibration isotonique sur le RF (plus stable que Platt pour RF)
        cal_rf = CalibratedClassifierCV(
            estimator=model._rf,
            method="isotonic",
            cv=3,
        )
        cal_rf.fit(X_sc, y_bin)
        model._rf = cal_rf
        logger.debug("Calibration RF terminée")

    except Exception as exc:
        logger.debug("Calibration échouée : {} — modèle non calibré", exc)

    return model


def _evaluer_par_region(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta: list,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Évalue les performances par région si les métadonnées sont disponibles."""
    try:
        from sklearn.metrics import roc_auc_score
        from collections import defaultdict

        region_scores: dict = defaultdict(list)
        region_labels: dict = defaultdict(list)

        for i, m in enumerate(meta[:len(X_test)]):
            rid = m.get("region_id", "unknown")
            region_scores[rid].append(float(model._predict_raw(X_test[i:i+1])[0]))
            region_labels[rid].append(float(y_test[i] >= 0.25))

        regional = {}
        for rid in list(region_scores.keys())[:top_n]:
            s = region_scores[rid]
            l = region_labels[rid]
            if len(set(l)) < 2:
                continue
            try:
                regional[rid] = {
                    "auc":       round(roc_auc_score(l, s), 3),
                    "n_samples": len(s),
                }
            except Exception:
                pass

        return regional
    except Exception as exc:
        logger.debug("Évaluation régionale : {}", exc)
        return {}


# ─────────────────────────────────────────────────────────────────
# MLflow et Model Card
# ─────────────────────────────────────────────────────────────────

def _log_mlflow_nutrition(
    model,
    metriques: Dict[str, float],
    params: Dict[str, Any],
    model_path: Optional[str],
) -> Optional[str]:
    try:
        import mlflow
        from config.settings import settings

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(f"{settings.mlflow_experiment_name}-nutrition")

        # Aplatit les métriques imbriquées
        flat_metrics = {
            k: v for k, v in metriques.items()
            if isinstance(v, (int, float))
        }

        with mlflow.start_run(run_name=f"nutrition_{model.MODEL_VERSION}") as run:
            mlflow.log_params(params)
            mlflow.log_metrics(flat_metrics)

            if model_path:
                mlflow.log_artifact(model_path, artifact_path="model")

            mlflow.set_tag("modele", "NutritionPredictor")
            mlflow.set_tag("framework", "Ensemble-RF-MLP")
            mlflow.set_tag("version", model.MODEL_VERSION)

            return run.info.run_id

    except Exception as exc:
        logger.debug("MLflow log nutrition : {}", exc)
        return None


def _generer_model_card_nutrition(
    model,
    metriques: Dict,
    date_debut: date,
    date_fin: date,
) -> None:
    try:
        from src.models.explainability import SHAPExplainer
        from pathlib import Path
        import json

        explainer = SHAPExplainer(model)
        card = explainer.generate_model_card()
        card["performances"] = {k: v for k, v in metriques.items() if isinstance(v, (int, float))}
        card["donnees_entrainement"]["periode"] = f"{date_debut} → {date_fin}"

        card_path = Path("docs/model_cards/nutrition_predictor_card.json")
        card_path.parent.mkdir(parents=True, exist_ok=True)
        with card_path.open("w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)
        logger.info("Model Card nutrition → {}", card_path)
    except Exception as exc:
        logger.debug("Model Card nutrition : {}", exc)


def _generate_version() -> str:
    return f"1.{date.today().strftime('%Y%m%d')}"


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement NutritionPredictor Ensemble "
    )
    parser.add_argument(
        "--date-debut",
        type=lambda s: date.fromisoformat(s),
        default=date.today() - timedelta(days=730),
    )
    parser.add_argument(
        "--date-fin",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
    )
    parser.add_argument("--seuil-auc",  type=float, default=0.70)
    parser.add_argument("--optimiser-hp", action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()

    args = _parse_args()
    result = train_nutrition_model(
        date_debut=args.date_debut,
        date_fin=args.date_fin,
        valider_seuil_auc=args.seuil_auc,
        sauvegarder=not args.dry_run,
        optimiser_hyperparams=args.optimiser_hp,
    )

    print("\n" + "=" * 60)
    print("RÉSULTATS ENTRAÎNEMENT NUTRITION")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")

    sys.exit(0 if result.get("valide") else 1)