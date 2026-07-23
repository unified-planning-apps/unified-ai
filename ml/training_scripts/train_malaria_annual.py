"""
ml/training_scripts/train_malaria_annual.py
=============================================
Entraînement d'un modèle de risque paludisme ANNUEL par région, à partir
de malaria_risk_annual (Malaria Atlas Project) — voir conversation.

Pourquoi un script séparé de train_malaria.py (hebdomadaire) :
  - malaria_observations (who_gho_distribue) n'a pas de vraie variance
    régionale (incidence nationale redistribuée) → AUC hebdomadaire
    artificiellement parfait, aucune valeur prédictive réelle.
  - malaria_risk_annual (Malaria Atlas Project) a une vraie variance
    régionale (confirmée : de 76 à 273 cas/1000 selon les régions en 2020),
    mais seulement annuelle. Pas la même granularité, pas le même modèle.
  - weather_observations ne couvre que 2023-2026, donc le climat réel
    n'est disponible que pour 2-3 ans sur les 24 de malaria_risk_annual —
    il est utilisé en feature BONUS (imputée à 0 + flag de disponibilité
    quand absente), pas comme signal principal.

Cible :
  - incidence_pour_mille de l'année courante (régression)
  - risque_eleve = incidence > médiane nationale de l'année (classification,
    pour une lecture "priorisation" plus simple que la valeur brute)

Features :
  - Lags annuels : incidence/prévalence/mortalité de l'année précédente
    et n-2 (auto-corrélation épidémiologique reconnue, légitime ici car
    basée sur de vraies estimations MAP indépendantes par année, pas une
    redistribution artificielle comme who_gho_distribue)
  - Géographie statique (GeoProcessor) : latitude, longitude, altitude,
    zone climatique encodée, endémicité encodée, indice de vulnérabilité
  - Climat annuel réel (bonus, si disponible) : température moyenne,
    précipitations totales, humidité moyenne + flag climat_disponible

Split : les 4 dernières années (2021-2024) en test, jamais mélangées avec
l'entraînement — cohérent avec ce qu'on a appris sur le hebdomadaire
(un split aléatoire sur des données autocorrélées dans le temps gonfle
artificiellement les métriques).

Usage :
    python -m ml.training_scripts.train_malaria_annual
    python -m ml.training_scripts.train_malaria_annual --annees-test 2021 2022 2023 2024
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────
# Construction du dataset
# ─────────────────────────────────────────────────────────────────

def _charger_malaria_risk_annual(engine) -> Dict[Tuple[str, int], Dict[str, Optional[float]]]:
    """Retourne {(region_code, annee): {incidence, prevalence, mortalite}}."""
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT region_code, annee, incidence_pour_mille,
                   prevalence_pct, mortalite_pour_100k
            FROM malaria_risk_annual
            WHERE source = 'malaria_atlas_project'
        """)).fetchall()
    return {
        (r.region_code, r.annee): {
            "incidence": float(r.incidence_pour_mille) if r.incidence_pour_mille is not None else None,
            "prevalence": float(r.prevalence_pct) if r.prevalence_pct is not None else None,
            "mortalite": float(r.mortalite_pour_100k) if r.mortalite_pour_100k is not None else None,
        }
        for r in rows
    }


def _charger_climat_annuel(engine) -> Dict[Tuple[str, int], Dict[str, float]]:
    """
    Retourne {(region_id, annee): {temp_moy, precip_totale, humidite_moy}}
    depuis weather_observations, agrégé par année. Probablement disponible
    seulement pour 2-3 années récentes (2023-2026) — c'est attendu.
    """
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                region_id,
                EXTRACT(YEAR FROM horodatage)::int AS annee,
                AVG(temperature_c) AS temp_moy,
                SUM(precipitations_mm) AS precip_totale,
                AVG(humidite_pct) AS humidite_moy
            FROM weather_observations
            GROUP BY region_id, EXTRACT(YEAR FROM horodatage)
        """)).fetchall()
    return {
        (r.region_id, r.annee): {
            "temp_moy": float(r.temp_moy) if r.temp_moy is not None else 0.0,
            "precip_totale": float(r.precip_totale) if r.precip_totale is not None else 0.0,
            "humidite_moy": float(r.humidite_moy) if r.humidite_moy is not None else 0.0,
        }
        for r in rows
    }


FEATURE_NAMES = [
    "incidence_lag1", "incidence_lag2",
    "prevalence_lag1", "mortalite_lag1",
    "latitude", "longitude", "altitude_m",
    "zone_climatique_encoded", "endemicite_encoded", "indice_vulnerabilite",
    "temp_moy_annee", "precip_totale_annee", "humidite_moy_annee",
    "climat_disponible",
]


def build_dataset(engine) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], List[str]]:
    """
    Construit le dataset annuel : (X, y_incidence, y_risque_eleve, meta, feature_names).
    Un échantillon par (région, année) où l'année précédente existe dans
    malaria_risk_annual (nécessaire pour les lags).
    """
    from src.preprocessing.geo_processor import GeoProcessor
    from src.utils.constants import REGIONS_MADAGASCAR

    risk = _charger_malaria_risk_annual(engine)
    climat = _charger_climat_annuel(engine)
    geo = GeoProcessor(db=None)  # mode metadata-only, pas besoin de session ici

    annees = sorted({a for (_, a) in risk.keys()})
    if not annees:
        logger.error("malaria_risk_annual est vide — as-tu bien lancé import_map_malaria.py ?")
        return np.array([]), np.array([]), np.array([]), [], FEATURE_NAMES

    rows_X, rows_y_inc, rows_meta = [], [], []

    for region_id in REGIONS_MADAGASCAR:
        for annee in annees:
            cible = risk.get((region_id, annee))
            if cible is None or cible["incidence"] is None:
                continue

            lag1 = risk.get((region_id, annee - 1))
            lag2 = risk.get((region_id, annee - 2))
            if lag1 is None or lag1["incidence"] is None:
                continue  # pas de lag1 disponible → échantillon ignoré

            geo_feat = geo.get_geo_features(region_id)
            clim = climat.get((region_id, annee))

            row = {
                "incidence_lag1":  lag1["incidence"],
                "incidence_lag2":  lag2["incidence"] if lag2 and lag2["incidence"] is not None else lag1["incidence"],
                "prevalence_lag1": lag1["prevalence"] if lag1["prevalence"] is not None else 0.0,
                "mortalite_lag1":  lag1["mortalite"] if lag1["mortalite"] is not None else 0.0,
                "latitude":  geo_feat["latitude"],
                "longitude": geo_feat["longitude"],
                "altitude_m": geo_feat["altitude_m"],
                "zone_climatique_encoded": geo_feat["zone_climatique_encoded"],
                "endemicite_encoded": geo_feat["endemicite_encoded"],
                "indice_vulnerabilite": geo_feat["indice_vulnerabilite"],
                "temp_moy_annee":     clim["temp_moy"] if clim else 0.0,
                "precip_totale_annee": clim["precip_totale"] if clim else 0.0,
                "humidite_moy_annee": clim["humidite_moy"] if clim else 0.0,
                "climat_disponible":  1.0 if clim else 0.0,
            }

            rows_X.append([row[n] for n in FEATURE_NAMES])
            rows_y_inc.append(cible["incidence"])
            rows_meta.append({"region_id": region_id, "annee": annee})

    if not rows_X:
        return np.array([]), np.array([]), np.array([]), [], FEATURE_NAMES

    X = np.array(rows_X, dtype=np.float32)
    y_incidence = np.array(rows_y_inc, dtype=np.float32)

    # Risque élevé = incidence au-dessus de la médiane NATIONALE de l'année
    # (comparaison relative par année, plus robuste qu'un seuil absolu fixe
    # vu que l'incidence nationale moyenne varie beaucoup entre 2000 et 2024)
    y_risque_eleve = np.zeros(len(y_incidence), dtype=np.int32)
    for annee in annees:
        idx = [i for i, m in enumerate(rows_meta) if m["annee"] == annee]
        if not idx:
            continue
        mediane = np.median(y_incidence[idx])
        for i in idx:
            y_risque_eleve[i] = int(y_incidence[i] > mediane)

    logger.info(
        "Dataset annuel construit : {} échantillons ({} régions × jusqu'à {} années)",
        len(X), len(REGIONS_MADAGASCAR), len(annees) - 1
    )
    return X, y_incidence, y_risque_eleve, rows_meta, FEATURE_NAMES


# ─────────────────────────────────────────────────────────────────
# Split temporel (années récentes en test)
# ─────────────────────────────────────────────────────────────────

def split_temporel(
    X, y_inc, y_risk, meta, annees_test: List[int]
) -> Tuple[np.ndarray, ...]:
    test_mask  = np.array([m["annee"] in annees_test for m in meta])
    train_mask = ~test_mask
    return (
        X[train_mask], X[test_mask],
        y_inc[train_mask], y_inc[test_mask],
        y_risk[train_mask], y_risk[test_mask],
    )


# ─────────────────────────────────────────────────────────────────
# Entraînement + évaluation
# ─────────────────────────────────────────────────────────────────

def train_malaria_annual_model(
    annees_test: Optional[List[int]] = None,
) -> Dict[str, Any]:
    debut = time.time()
    annees_test = annees_test or [2021, 2022, 2023, 2024]

    import sqlalchemy as sa
    from config.settings import settings

    engine = sa.create_engine(settings.database.sync_url, pool_pre_ping=True)
    try:
        X, y_inc, y_risk, meta, feature_names = build_dataset(engine)
    finally:
        engine.dispose()

    result: Dict[str, Any] = {
        "nb_samples": len(X), "nb_features": len(feature_names),
        "metriques": {}, "valide": False,
    }

    if len(X) < 30:
        logger.error(
            "Seulement {} échantillons — trop peu pour un entraînement fiable. "
            "As-tu bien importé malaria_risk_annual (550 lignes attendues) ?",
            len(X)
        )
        result["erreur"] = "dataset insuffisant"
        return result

    X_train, X_test, y_inc_train, y_inc_test, y_risk_train, y_risk_test = split_temporel(
        X, y_inc, y_risk, meta, annees_test
    )
    logger.info(
        "Split temporel — Train: {} (< {}) | Test: {} (années {})",
        len(X_train), min(annees_test), len(X_test), annees_test
    )

    if len(X_test) == 0 or len(set(y_risk_test)) < 2:
        logger.warning(
            "Jeu de test vide ou mono-classe pour les années {} — "
            "essaie d'autres années de test.", annees_test
        )

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, roc_auc_score, f1_score
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Régression incidence ──────────────────────────────────────
    reg = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    reg.fit(X_train_sc, y_inc_train)
    y_pred_inc = reg.predict(X_test_sc)

    mae  = mean_absolute_error(y_inc_test, y_pred_inc)
    rmse = float(np.sqrt(mean_squared_error(y_inc_test, y_pred_inc)))

    # ── Classification risque élevé ────────────────────────────────
    clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_train_sc, y_risk_train)
    y_proba = clf.predict_proba(X_test_sc)[:, 1]
    y_pred_risk = (y_proba >= 0.5).astype(int)

    metriques: Dict[str, Any] = {
        "mae_incidence":  round(float(mae), 3),
        "rmse_incidence": round(rmse, 3),
    }

    if len(set(y_risk_test)) >= 2:
        metriques["auc_roc"]  = round(float(roc_auc_score(y_risk_test, y_proba)), 4)
        metriques["f1_score"] = round(float(f1_score(y_risk_test, y_pred_risk)), 4)
    else:
        metriques["auc_roc"] = None
        logger.warning("AUC non calculable (une seule classe dans le test)")

    # Importance des features (utile pour comprendre ce que le modèle apprend)
    importances = sorted(
        zip(feature_names, reg.feature_importances_), key=lambda x: -x[1]
    )
    metriques["top_features"] = [(n, round(float(v), 4)) for n, v in importances[:5]]

    result["metriques"] = metriques
    result["valide"] = metriques.get("auc_roc") is not None and metriques["auc_roc"] >= 0.60
    result["duree_sec"] = round(time.time() - debut, 2)

    logger.info(
        "Métriques — MAE incidence: {} | RMSE: {} | AUC risque_eleve: {} | F1: {}",
        metriques["mae_incidence"], metriques["rmse_incidence"],
        metriques.get("auc_roc"), metriques.get("f1_score")
    )
    logger.info("Top features (régression incidence) : {}", metriques["top_features"])

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement risque paludisme annuel par région (MAP)")
    parser.add_argument(
        "--annees-test", type=int, nargs="+", default=[2021, 2022, 2023, 2024],
        help="Années tenues à l'écart de l'entraînement pour l'évaluation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()

    args = _parse_args()
    result = train_malaria_annual_model(annees_test=args.annees_test)

    print("\n" + "=" * 60)
    print("RÉSULTATS ENTRAÎNEMENT MALARIA ANNUEL (MAP)")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")

    sys.exit(0 if result.get("valide") else 1)