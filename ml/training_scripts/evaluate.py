"""
ml/training_scripts/evaluate.py
=================================
Script d'évaluation standalone des modèles ML —  .

Fonctions exportées (utilisées par train_malaria.py, train_nutrition.py, scheduler.py) :
  evaluate_malaria_model(model, X_test, y_test, feature_names) → Dict
  evaluate_nutrition_model(model, X_test, y_test, y_gam_test, feature_names) → Dict
  compare_models(results_list) → Dict
  evaluate_drift(model, X_recent, X_reference) → Dict
  generate_evaluation_report(results, output_path) → Path

Métriques calculées :
  Classification : AUC-ROC, Average Precision, F1, Precision, Recall, MCC
  Régression     : MAE, RMSE, MAPE, R², Biais
  Calibration    : Brier Score, ECE (Expected Calibration Error)
  Dérive         : PSI (Population Stability Index) par feature
  Backtesting    : Prédictions vs réel sur fenêtre glissante

Usage CLI :
  python -m ml.training_scripts.evaluate --modele paludisme
  python -m ml.training_scripts.evaluate --modele nutrition --region MDG-ANA
  python -m ml.training_scripts.evaluate --compare --rapport rapport.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ─────────────────────────────────────────────────────────────────
# Seuils de performance UNICEF (standards opérationnels)
# ─────────────────────────────────────────────────────────────────
SEUILS_PERFORMANCE = {
    "malaria": {
        "auc_roc":           0.75,   # Minimum acceptable
        "auc_roc_target":    0.85,   # Cible qualité
        "f1_score":          0.65,
        "recall":            0.70,   # Sensibilité critique (ne pas rater les cas)
        "brier_score_max":   0.20,   # Calibration acceptable
    },
    "nutrition": {
        "auc_roc":           0.72,
        "auc_roc_target":    0.82,
        "f1_score":          0.60,
        "recall":            0.68,
        "mae_gam_max":       3.0,    # MAE GAM < 3 points de %
        "brier_score_max":   0.22,
    },
}


# ─────────────────────────────────────────────────────────────────
# Évaluation modèle paludisme
# ─────────────────────────────────────────────────────────────────

def evaluate_malaria_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    seuil_classification: float = 0.25,
) -> Dict[str, Any]:
    """
    Évaluation complète du MalariaPredictor.

    Args:
        model                : instance MalariaPredictor entraîné
        X_test               : features jeu de test (déjà scalées)
        y_test               : labels continus [0,1]
        feature_names        : noms des features (pour analyse d'importance)
        seuil_classification : seuil pour binarisation (défaut 0.25 = risque moyen)

    Returns:
        dict avec toutes les métriques et diagnostics
    """
    start = time.time()
    logger.info("📊 Évaluation MalariaPredictor — {} samples", len(X_test))

    results: Dict[str, Any] = {
        "modele":       "MalariaPredictor",
        "date_eval":    datetime.utcnow().isoformat(),
        "nb_samples":   len(X_test),
    }

    # Prédictions
    y_pred_proba = model._predict_raw(X_test)
    y_true_bin   = (y_test >= seuil_classification).astype(int)
    y_pred_bin   = (y_pred_proba >= 0.5).astype(int)

    # ── Métriques de classification ──
    results["classification"] = _compute_classification_metrics(
        y_true_bin, y_pred_bin, y_pred_proba
    )

    # ── Métriques de calibration ──
    results["calibration"] = _compute_calibration_metrics(y_true_bin, y_pred_proba)

    # ── Métriques de régression (sur scores continus) ──
    results["regression"] = _compute_regression_metrics(y_test, y_pred_proba)

    # ── Analyse par décile de score ──
    results["decile_analysis"] = _decile_analysis(y_test, y_pred_proba)

    # ── Importance des features ──
    if feature_names:
        results["feature_importance"] = _compute_feature_importance(
            model, feature_names
        )

    # ── Validation vs seuils UNICEF ──
    results["validation_unicef"] = _valider_seuils(
        results["classification"], SEUILS_PERFORMANCE["malaria"]
    )

    results["duree_evaluation_sec"] = round(time.time() - start, 2)

    _log_resume_evaluation(results, "malaria")
    return results


# ─────────────────────────────────────────────────────────────────
# Évaluation modèle nutrition
# ─────────────────────────────────────────────────────────────────

def evaluate_nutrition_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_gam_test: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    seuil_classification: float = 0.25,
) -> Dict[str, Any]:
    """
    Évaluation complète du NutritionPredictor.

    Args:
        model        : instance NutritionPredictor entraîné
        X_test       : features jeu de test (scalées)
        y_test       : scores de risque normalisés [0,1]
        y_gam_test   : taux GAM réels en % (pour métriques régresseur)
        feature_names: noms des features

    Returns:
        dict métriques complètes + analyse GAM
    """
    start = time.time()
    logger.info("📊 Évaluation NutritionPredictor — {} samples", len(X_test))

    results: Dict[str, Any] = {
        "modele":       "NutritionPredictor",
        "date_eval":    datetime.utcnow().isoformat(),
        "nb_samples":   len(X_test),
    }

    y_pred_proba = model._predict_raw(X_test)
    y_true_bin   = (y_test >= seuil_classification).astype(int)
    y_pred_bin   = (y_pred_proba >= 0.5).astype(int)

    # ── Métriques de classification ──
    results["classification"] = _compute_classification_metrics(
        y_true_bin, y_pred_bin, y_pred_proba
    )

    # ── Calibration ──
    results["calibration"] = _compute_calibration_metrics(y_true_bin, y_pred_proba)

    # ── Régression GAM (si régresseur disponible) ──
    if y_gam_test is not None and hasattr(model, "_reg") and model._reg is not None:
        try:
            y_gam_pred = model._reg.predict(X_test)
            results["regression_gam"] = _compute_regression_metrics(
                y_gam_test, y_gam_pred, label="GAM (%)"
            )
            results["regression_gam"]["mae_interpretation"] = (
                "Excellent" if results["regression_gam"]["mae"] < 1.5
                else "Acceptable" if results["regression_gam"]["mae"] < 3.0
                else "À améliorer"
            )
        except Exception as exc:
            logger.debug("Régression GAM : {}", exc)

    # ── Analyse par seuil GAM (OMS) ──
    if y_gam_test is not None:
        results["analyse_seuils_gam"] = _analyser_seuils_gam(
            y_gam_test,
            y_gam_pred if "regression_gam" in results else y_pred_proba * 20,
        )

    # ── Déciles ──
    results["decile_analysis"] = _decile_analysis(y_test, y_pred_proba)

    # ── Importance features ──
    if feature_names:
        results["feature_importance"] = _compute_feature_importance(
            model, feature_names
        )

    # ── Validation seuils UNICEF ──
    results["validation_unicef"] = _valider_seuils(
        results["classification"], SEUILS_PERFORMANCE["nutrition"]
    )

    # ── Analyse des composants RF vs MLP ──
    if hasattr(model, "_rf") and hasattr(model, "_mlp"):
        results["ensemble_components"] = _analyser_composants_ensemble(
            model, X_test, y_true_bin
        )

    results["duree_evaluation_sec"] = round(time.time() - start, 2)
    _log_resume_evaluation(results, "nutrition")
    return results


# ─────────────────────────────────────────────────────────────────
# Évaluation backtesting (prédictions vs réel)
# ─────────────────────────────────────────────────────────────────

def evaluate_backtest(
    model,
    region_id: str,
    date_debut: date,
    date_fin: date,
    modele_nom: str = "malaria",
) -> Dict[str, Any]:
    """
    Évalue les performances historiques sur une fenêtre glissante.
    Compare les prédictions archivées aux valeurs réellement observées.

    Utilisé pour valider la fiabilité du modèle sur le terrain.
    """
    import asyncio
    logger.info(
        "📈 Backtest {} — région={} période={} → {}",
        modele_nom, region_id, date_debut, date_fin
    )

    results: Dict[str, Any] = {
        "modele":       modele_nom,
        "region_id":    region_id,
        "periode_debut": str(date_debut),
        "periode_fin":   str(date_fin),
    }

    try:
        from src.preprocessing.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()

        predictions_semaine = []
        current = date_debut

        while current <= date_fin:
            try:
                # Prédiction à date donnée
                if modele_nom == "malaria":
                    features = asyncio.run(
                        engineer.build_malaria_features(region_id, current)
                    )
                else:
                    features = asyncio.run(
                        engineer.build_nutrition_features(region_id, current)
                    )

                pred = model.predict(features, horizon_days=7)
                score = pred.get("score_risque", 0)

                # Label réel (récupéré depuis DB pour la semaine suivante)
                label_reel = asyncio.run(
                    _recuperer_label_reel(region_id, current, modele_nom)
                )

                if label_reel is not None:
                    predictions_semaine.append({
                        "date":         str(current),
                        "score_predit": score,
                        "valeur_reelle": label_reel,
                        "erreur":        abs(score - label_reel),
                    })

            except Exception as exc:
                logger.debug("Backtest {} {} : {}", region_id, current, exc)

            current += timedelta(weeks=1)

        if not predictions_semaine:
            return {**results, "erreur": "Aucune donnée de backtest disponible"}

        pred_vals = np.array([p["score_predit"] for p in predictions_semaine])
        real_vals = np.array([p["valeur_reelle"] for p in predictions_semaine])

        results["metriques"] = _compute_regression_metrics(real_vals, pred_vals)
        results["nb_predictions"]     = len(predictions_semaine)
        results["predictions_detail"] = predictions_semaine[:50]  # Max 50

        # Analyse des erreurs
        erreurs = np.abs(pred_vals - real_vals)
        results["analyse_erreurs"] = {
            "erreur_moy":           round(float(np.mean(erreurs)), 4),
            "erreur_mediane":       round(float(np.median(erreurs)), 4),
            "erreur_p90":           round(float(np.percentile(erreurs, 90)), 4),
            "pct_erreur_lt_01":     round(float(np.mean(erreurs < 0.10)) * 100, 1),
            "pct_erreur_lt_02":     round(float(np.mean(erreurs < 0.20)) * 100, 1),
            "nb_grandes_erreurs":   int(np.sum(erreurs > 0.30)),
        }

        logger.info(
            "Backtest terminé — {} prédictions | MAE={:.3f} | Corr={:.3f}",
            len(predictions_semaine),
            results["metriques"].get("mae", 0),
            results["metriques"].get("correlation", 0),
        )

    except Exception as exc:
        logger.error("Backtest échoué : {}", exc)
        results["erreur"] = str(exc)

    return results


# ─────────────────────────────────────────────────────────────────
# Drift detection (PSI)
# ─────────────────────────────────────────────────────────────────

def evaluate_drift(
    X_recent: np.ndarray,
    X_reference: np.ndarray,
    feature_names: Optional[List[str]] = None,
    psi_threshold_warning: float = 0.10,
    psi_threshold_alert:   float = 0.20,
) -> Dict[str, Any]:
    """
    Calcule le Population Stability Index (PSI) pour détecter
    la dérive de distribution des features d'entrée.

    PSI < 0.10  → Pas de dérive significative
    PSI 0.10-0.20 → Dérive modérée (surveiller)
    PSI > 0.20  → Dérive importante (retraining requis)

    Args:
        X_recent    : données récentes (derniers 30 jours)
        X_reference : données de référence (période d'entraînement)
        feature_names : noms des features

    Returns:
        dict avec PSI par feature + évaluation globale
    """
    n_features = X_recent.shape[1]
    feature_names = feature_names or [f"f_{i}" for i in range(n_features)]

    psi_scores: Dict[str, float] = {}
    features_en_derive: List[str] = []

    for i, fname in enumerate(feature_names):
        try:
            psi = _calculate_psi(X_reference[:, i], X_recent[:, i])
            psi_scores[fname] = round(psi, 4)
            if psi > psi_threshold_alert:
                features_en_derive.append(fname)
        except Exception:
            psi_scores[fname] = 0.0

    psi_global = round(float(np.mean(list(psi_scores.values()))), 4)
    psi_max    = round(float(max(psi_scores.values())), 4)

    statut = (
        "retraining_requis" if psi_max > psi_threshold_alert
        else "surveillance"  if psi_max > psi_threshold_warning
        else "stable"
    )

    top_derives = sorted(
        psi_scores.items(), key=lambda x: x[1], reverse=True
    )[:10]

    result = {
        "date_analyse":         datetime.utcnow().isoformat(),
        "psi_global_moyen":     psi_global,
        "psi_maximum":          psi_max,
        "statut":               statut,
        "nb_features_en_derive": len(features_en_derive),
        "features_en_derive":   features_en_derive,
        "top_10_features_psi":  dict(top_derives),
        "psi_par_feature":      psi_scores,
        "seuils": {
            "warning": psi_threshold_warning,
            "alert":   psi_threshold_alert,
        },
        "recommandation": (
            "🚨 Retraining urgent requis — distribution des données a significativement changé"
            if statut == "retraining_requis"
            else "⚠️ Surveillance renforcée recommandée — dérive modérée détectée"
            if statut == "surveillance"
            else "✅ Distribution stable — pas d'action requise"
        ),
    }

    logger.info(
        "Drift analysis — PSI_max={:.3f} PSI_moy={:.3f} statut={}",
        psi_max, psi_global, statut
    )
    return result


# ─────────────────────────────────────────────────────────────────
# Comparaison entre versions de modèles
# ─────────────────────────────────────────────────────────────────

def compare_models(
    results_list: List[Dict[str, Any]],
    metrique_principale: str = "auc_roc",
) -> Dict[str, Any]:
    """
    Compare plusieurs versions d'un modèle et identifie la meilleure.
    Utilisé pour décider du déploiement.

    Args:
        results_list       : liste de dicts retournés par evaluate_malaria/nutrition
        metrique_principale: métrique de comparaison principale

    Returns:
        dict avec classement, meilleur modèle, et recommandation de déploiement
    """
    if not results_list:
        return {"erreur": "Liste de résultats vide"}

    comparaison = []
    for i, r in enumerate(results_list):
        clf = r.get("classification", {})
        reg_gam = r.get("regression_gam", {})

        score = clf.get(metrique_principale, 0)
        comparaison.append({
            "index":          i,
            "modele":         r.get("modele", f"modele_{i}"),
            "date_eval":      r.get("date_eval", ""),
            "auc_roc":        clf.get("auc_roc", 0),
            "f1_score":       clf.get("f1_score", 0),
            "recall":         clf.get("recall", 0),
            "precision":      clf.get("precision", 0),
            "brier_score":    r.get("calibration", {}).get("brier_score", 1),
            "mae_gam":        reg_gam.get("mae"),
            "nb_samples":     r.get("nb_samples", 0),
            "score_composite": _calculer_score_composite(clf, reg_gam),
        })

    comparaison.sort(key=lambda x: x["score_composite"], reverse=True)
    meilleur = comparaison[0]

    # Décision de déploiement
    modele_nom = meilleur.get("modele", "unknown").lower()
    type_modele = "nutrition" if "nutrition" in modele_nom else "malaria"
    seuils = SEUILS_PERFORMANCE.get(type_modele, SEUILS_PERFORMANCE["malaria"])
    deployer = meilleur["auc_roc"] >= seuils["auc_roc"]

    return {
        "date_comparaison": datetime.utcnow().isoformat(),
        "nb_modeles_compares": len(comparaison),
        "metrique_principale":  metrique_principale,
        "classement":           comparaison,
        "meilleur_modele":      meilleur,
        "recommandation_deploiement": deployer,
        "raison": (
            f"AUC={meilleur['auc_roc']:.3f} ≥ seuil={seuils['auc_roc']}"
            if deployer
            else f"AUC={meilleur['auc_roc']:.3f} < seuil={seuils['auc_roc']}"
        ),
        "ameliorations_vs_precedent": _calculer_ameliorations(comparaison),
    }


# ─────────────────────────────────────────────────────────────────
# Génération rapport d'évaluation
# ─────────────────────────────────────────────────────────────────

def generate_evaluation_report(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
    format_rapport: str = "json",
) -> Path:
    """
    Génère un rapport d'évaluation structuré (JSON ou Markdown).
    Utilisé pour l'audit UNICEF et la traçabilité des modèles.

    Args:
        results      : dict retourné par evaluate_malaria/nutrition
        output_path  : chemin de sortie (auto-généré si None)
        format_rapport: "json" | "markdown"

    Returns:
        Path vers le fichier généré
    """
    if output_path is None:
        modele   = results.get("modele", "modele").lower().replace(" ", "_")
        date_str = datetime.utcnow().strftime("%Y%m%d_%H%M")
        ext      = ".md" if format_rapport == "markdown" else ".json"
        output_path = Path(f"ml/experiments/eval_{modele}_{date_str}{ext}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format_rapport == "markdown":
        content = _results_to_markdown(results)
        output_path.write_text(content, encoding="utf-8")
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Rapport d'évaluation → {}", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────
# Métriques de base
# ─────────────────────────────────────────────────────────────────

def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Calcule toutes les métriques de classification."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
        matthews_corrcoef, confusion_matrix,
        balanced_accuracy_score,
    )

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        tn = fp = fn = tp = 0

    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos

    return {
        "auc_roc":            round(roc_auc_score(y_true, y_proba), 4),
        "average_precision":  round(average_precision_score(y_true, y_proba), 4),
        "f1_score":           round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision":          round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":             round(recall_score(y_true, y_pred, zero_division=0), 4),
        "specificity":        round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
        "mcc":                round(matthews_corrcoef(y_true, y_pred), 4),
        "balanced_accuracy":  round(balanced_accuracy_score(y_true, y_pred), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_positifs":  n_pos,
        "n_negatifs":  n_neg,
        "prevalence_pct": round(n_pos / len(y_true) * 100, 1),
    }


def _compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Brier Score + Expected Calibration Error."""
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve

    brier = round(float(brier_score_loss(y_true, y_proba)), 4)

    # ECE (Expected Calibration Error)
    try:
        fraction_of_positives, mean_predicted = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy="uniform"
        )
        ece = round(float(np.mean(np.abs(fraction_of_positives - mean_predicted))), 4)
        calibration_curve_data = {
            "probas_predites":  mean_predicted.tolist(),
            "fractions_reelles": fraction_of_positives.tolist(),
        }
    except Exception:
        ece = 0.0
        calibration_curve_data = {}

    return {
        "brier_score":          brier,
        "ece":                  ece,
        "interpretation_brier": (
            "Excellent" if brier < 0.10
            else "Bon" if brier < 0.15
            else "Acceptable" if brier < 0.20
            else "À améliorer"
        ),
        "calibration_curve": calibration_curve_data,
    }


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> Dict[str, float]:
    """MAE, RMSE, MAPE, R², biais."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    biais = float(np.mean(y_pred - y_true))
    r2    = float(r2_score(y_true, y_pred))

    # MAPE (évite division par zéro)
    mape = float(np.mean(
        np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)
    ) * 100)

    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else 0.0

    return {
        "mae":         round(mae,   4),
        "rmse":        round(rmse,  4),
        "mape_pct":    round(mape,  2),
        "r2":          round(r2,    4),
        "biais":       round(biais, 4),
        "correlation": round(corr,  4),
        "label":       label,
    }


def _decile_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> List[Dict[str, float]]:
    """
    Analyse par décile de score prédit.
    Vérifie que les déciles élevés correspondent effectivement à des cas élevés.
    """
    deciles = np.percentile(y_pred, np.arange(0, 110, 10))
    results = []

    for i in range(10):
        mask = (y_pred >= deciles[i]) & (y_pred < deciles[min(i + 1, 9)])
        if mask.sum() == 0:
            continue
        results.append({
            "decile":           i + 1,
            "score_min":        round(float(deciles[i]), 3),
            "score_max":        round(float(deciles[min(i + 1, 9)]), 3),
            "n_samples":        int(mask.sum()),
            "prevalence_reelle": round(float(y_true[mask].mean()), 3),
            "score_moyen_predit": round(float(y_pred[mask].mean()), 3),
        })

    return results


def _compute_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Importance des features depuis le modèle sous-jacent."""
    importance: Dict[str, float] = {}

    # XGBoost
    clf = getattr(model, "_clf", None)
    if clf and hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
        for i, name in enumerate(feature_names[:len(scores)]):
            importance[name] = float(scores[i])

    # Random Forest (moyenne RF + MLP non disponible)
    elif hasattr(model, "_rf") and hasattr(model._rf, "feature_importances_"):
        try:
            scores = model._rf.feature_importances_
            for i, name in enumerate(feature_names[:len(scores)]):
                importance[name] = float(scores[i])
        except Exception:
            pass

    if not importance:
        return []

    total = sum(importance.values()) + 1e-9
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "rang":             i + 1,
            "feature":          name,
            "importance":       round(val, 4),
            "contribution_pct": round(val / total * 100, 1),
        }
        for i, (name, val) in enumerate(ranked[:top_n])
    ]


def _analyser_seuils_gam(
    y_gam_true: np.ndarray,
    y_gam_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Analyse des performances par seuil OMS (acceptable/alerte/urgence/crise).
    Vérifie la capacité à classifier correctement les situations humanitaires.
    """
    seuils = [5.0, 10.0, 15.0]
    resultats = {}

    for seuil in seuils:
        y_true_bin = (y_gam_true >= seuil).astype(int)
        y_pred_bin = (y_gam_pred >= seuil).astype(int)

        from sklearn.metrics import f1_score, recall_score, precision_score
        resultats[f"seuil_{int(seuil)}pct"] = {
            "f1":       round(f1_score(y_true_bin, y_pred_bin, zero_division=0), 3),
            "recall":   round(recall_score(y_true_bin, y_pred_bin, zero_division=0), 3),
            "precision":round(precision_score(y_true_bin, y_pred_bin, zero_division=0), 3),
            "n_cas_reels":   int(y_true_bin.sum()),
            "n_cas_predits": int(y_pred_bin.sum()),
        }

    return resultats


def _analyser_composants_ensemble(
    model,
    X_test: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    """Évalue chaque composant de l'ensemble séparément."""
    from sklearn.metrics import roc_auc_score

    composants = {}
    for nom_composant, composant in [("rf", model._rf), ("mlp", model._mlp)]:
        try:
            if hasattr(composant, "predict_proba"):
                proba = composant.predict_proba(X_test)[:, 1]
                composants[nom_composant] = {
                    "auc_roc": round(roc_auc_score(y_true, proba), 4),
                    "poids_ensemble": model.ENSEMBLE_WEIGHTS.get(nom_composant, 0.5),
                }
        except Exception as exc:
            logger.debug("Composant {} : {}", nom_composant, exc)

    return composants


def _valider_seuils(
    clf_metrics: Dict,
    seuils: Dict,
) -> Dict[str, Any]:
    """Vérifie si les métriques respectent les seuils UNICEF."""
    validations = {}
    for metrique, seuil in seuils.items():
        if metrique.endswith("_target") or metrique.endswith("_max"):
            continue
        valeur = clf_metrics.get(metrique, 0)
        if metrique.endswith("_max"):
            valide = valeur <= seuil
        else:
            valide = valeur >= seuil
        validations[metrique] = {
            "valeur": valeur, "seuil": seuil,
            "statut": "✅ OK" if valide else "❌ NON CONFORME",
        }

    tout_conforme = all(
        v["statut"].startswith("✅") for v in validations.values()
    )
    return {
        "conforme_unicef":  tout_conforme,
        "details":          validations,
        "score_conformite": round(
            sum(1 for v in validations.values() if v["statut"].startswith("✅"))
            / max(len(validations), 1) * 100, 1
        ),
    }


# ─────────────────────────────────────────────────────────────────
# PSI (Population Stability Index)
# ─────────────────────────────────────────────────────────────────

def _calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calcule le PSI entre la distribution de référence et la distribution actuelle.
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    """
    # Création des bins sur la distribution de référence
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0]  = -np.inf
    bins[-1] = +np.inf

    expected_pcts = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pcts   = np.histogram(actual,   bins=bins)[0] / len(actual)

    # Évite log(0)
    expected_pcts = np.where(expected_pcts == 0, 1e-4, expected_pcts)
    actual_pcts   = np.where(actual_pcts   == 0, 1e-4, actual_pcts)

    psi = float(np.sum(
        (actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)
    ))
    return max(0.0, psi)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _calculer_score_composite(
    clf: Dict,
    reg: Dict,
) -> float:
    """Score composite pour classement des modèles."""
    auc   = clf.get("auc_roc", 0)
    f1    = clf.get("f1_score", 0)
    rec   = clf.get("recall", 0)
    score = 0.40 * auc + 0.30 * f1 + 0.30 * rec

    if reg:
        mae_norm = 1 - min(1, reg.get("mae", 5) / 5)
        score    = 0.35 * auc + 0.25 * f1 + 0.25 * rec + 0.15 * mae_norm

    return round(float(score), 4)


def _calculer_ameliorations(comparaison: List[Dict]) -> Optional[Dict]:
    """Calcule l'amélioration du meilleur modèle vs le second."""
    if len(comparaison) < 2:
        return None
    best = comparaison[0]
    prev = comparaison[1]
    return {
        "delta_auc":   round(best["auc_roc"] - prev["auc_roc"], 4),
        "delta_f1":    round(best["f1_score"] - prev["f1_score"], 4),
        "delta_recall":round(best["recall"] - prev["recall"], 4),
        "amelioration_globale": (
            "oui" if best["score_composite"] > prev["score_composite"] else "non"
        ),
    }


def _log_resume_evaluation(results: Dict, modele: str) -> None:
    """Log un résumé lisible de l'évaluation."""
    clf    = results.get("classification", {})
    valid  = results.get("validation_unicef", {})
    logger.info(
        "📋 Résumé {} — AUC={} | F1={} | Recall={} | Conforme={}",
        modele,
        clf.get("auc_roc", "N/A"),
        clf.get("f1_score", "N/A"),
        clf.get("recall", "N/A"),
        valid.get("conforme_unicef", "N/A"),
    )


def _results_to_markdown(results: Dict) -> str:
    """Convertit les résultats en rapport Markdown lisible."""
    clf   = results.get("classification", {})
    calib = results.get("calibration", {})
    valid = results.get("validation_unicef", {})

    lines = [
        f"# Rapport d'Évaluation — {results.get('modele', 'Modèle')}",
        f"**Date** : {results.get('date_eval', 'N/A')}  ",
        f"**Samples évalués** : {results.get('nb_samples', 0)}",
        "",
        "## 📊 Métriques de Classification",
        f"| Métrique | Valeur |",
        f"|----------|--------|",
        f"| AUC-ROC  | **{clf.get('auc_roc', 'N/A')}** |",
        f"| F1-Score | {clf.get('f1_score', 'N/A')} |",
        f"| Recall   | {clf.get('recall', 'N/A')} |",
        f"| Précision| {clf.get('precision', 'N/A')} |",
        f"| MCC      | {clf.get('mcc', 'N/A')} |",
        "",
        "## 🎯 Calibration",
        f"| Métrique    | Valeur |",
        f"|-------------|--------|",
        f"| Brier Score | {calib.get('brier_score', 'N/A')} |",
        f"| ECE         | {calib.get('ece', 'N/A')} |",
        "",
        "## ✅ Validation Seuils UNICEF",
    ]

    for metrique, detail in valid.get("details", {}).items():
        lines.append(
            f"- {detail['statut']} **{metrique}** : "
            f"{detail['valeur']} (seuil : {detail['seuil']})"
        )

    lines += [
        "",
        f"**Score de conformité UNICEF : {valid.get('score_conformite', 0)}%**",
    ]

    reg_gam = results.get("regression_gam", {})
    if reg_gam:
        lines += [
            "",
            "## 📈 Métriques GAM (Régresseur)",
            f"- MAE : {reg_gam.get('mae', 'N/A')} points GAM%",
            f"- RMSE : {reg_gam.get('rmse', 'N/A')}",
            f"- Corrélation : {reg_gam.get('correlation', 'N/A')}",
        ]

    return "\n".join(lines)


async def _recuperer_label_reel(
    region_id: str,
    target_date: date,
    modele: str,
) -> Optional[float]:
    """Récupère le label réel depuis la DB pour le backtesting."""
    try:
        from src.preprocessing.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        return await engineer._get_label(region_id, target_date, modele)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Évaluation des modèles ML"
    )
    parser.add_argument(
        "--modele",
        choices=["malaria", "nutrition", "tous"],
        default="tous",
        help="Modèle à évaluer",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="ID région pour backtest (ex: MDG-ANA)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Activer le backtesting historique",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Analyser la dérive de distribution",
    )
    parser.add_argument(
        "--rapport",
        default=None,
        help="Chemin de sortie du rapport JSON/Markdown",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Format du rapport de sortie",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Comparer toutes les versions disponibles",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()

    args = _parse_args()
    tous_resultats = []

    modeles_a_evaluer = (
        ["malaria", "nutrition"]
        if args.modele == "tous"
        else [args.modele]
    )

    for nom_modele in modeles_a_evaluer:
        logger.info("=== Évaluation {} ===", nom_modele)

        try:
            if nom_modele == "malaria":
                from src.models.malaria_predictor import MalariaPredictor
                from ml.training_scripts.train_malaria import _generer_donnees_synthetiques_malaria
                from sklearn.preprocessing import StandardScaler

                model = MalariaPredictor.load_latest()
                if model is None:
                    logger.info("Aucun modèle sauvegardé — création démo")
                    model = MalariaPredictor.create_demo_model()

                X, y, fn = _generer_donnees_synthetiques_malaria()
                sc = StandardScaler()
                X_sc = sc.fit_transform(X[-500:])
                results = evaluate_malaria_model(
                    model, X_sc, y[-500:], feature_names=fn
                )

            else:
                from src.models.nutrition_predictor import NutritionPredictor
                from ml.training_scripts.train_nutrition import _generer_donnees_synthetiques_nutrition
                from sklearn.preprocessing import StandardScaler

                model = NutritionPredictor.load_latest()
                if model is None:
                    logger.info("Aucun modèle sauvegardé — création démo")
                    model = NutritionPredictor.create_demo_model()

                X, y, y_gam, fn = _generer_donnees_synthetiques_nutrition()
                sc = StandardScaler()
                X_sc = sc.fit_transform(X[-500:])
                results = evaluate_nutrition_model(
                    model, X_sc, y[-500:], y_gam[-500:], feature_names=fn
                )

            tous_resultats.append(results)

            # Backtest si demandé
            if args.backtest and args.region:
                bt = evaluate_backtest(
                    model, args.region,
                    date.today() - timedelta(days=90),
                    date.today(),
                    nom_modele,
                )
                results["backtest"] = bt

            # Génération rapport
            rapport_path = args.rapport
            if rapport_path is None:
                rapport_path = f"ml/experiments/eval_{nom_modele}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.{args.format if args.format == 'md' else 'json'}"

            generate_evaluation_report(
                results,
                Path(rapport_path),
                format_rapport=args.format,
            )

        except Exception as exc:
            logger.exception("Évaluation {} échouée : {}", nom_modele, exc)

    # Comparaison si plusieurs modèles
    if args.compare and len(tous_resultats) > 1:
        comp = compare_models(tous_resultats)
        print("\n=== COMPARAISON ===")
        print(json.dumps(comp, indent=2, default=str))

    print(f"\n✅ Évaluation terminée — {len(tous_resultats)} modèle(s)")
    sys.exit(0)