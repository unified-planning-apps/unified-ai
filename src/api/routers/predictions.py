"""
src/api/routers/predictions.py
================================
Endpoints de prédiction combinée (pipeline complet Météo → Paludisme → Nutrition).

Couvre :
- Prédiction combinée sur une région (score unifié)
- Batch multi-régions
- Scénarios what-if (simulation)
- Confiance et explicabilité des modèles (SHAP)
- Drift detection et santé des modèles
- Historique des prédictions (backtesting)
- Comparaison prédictions vs réel
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query, status
from loguru import logger
from pydantic import BaseModel, Field

from src.api.dependencies import (
    AdminOnly,
    AuthUser,
    Cache,
    DbSession,
    MalariaModel,
    NationalOrAdmin,
    NutritionModel,
    Pagination,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────────────────────────

class PredictionCombinee(BaseModel):
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    region_id: str
    region_name: str
    date_prediction: datetime
    horizon_jours: int

    # Scores des deux modèles
    score_paludisme: float = Field(..., ge=0, le=1)
    niveau_paludisme: str
    score_nutrition: float = Field(..., ge=0, le=1)
    niveau_nutrition: str

    # Score composite UNICEF (pondéré)
    score_composite: float = Field(..., ge=0, le=1,
        description="Score unifié pondéré paludisme (60%) + nutrition (40%)")
    niveau_alerte_global: str = Field(
        description="vert | jaune | orange | rouge"
    )
    couleur_carte: str = Field(
        description="Code hex pour affichage choroplèthe"
    )

    # Projections clés
    cas_paludisme_prevus_14j: int
    gam_prevu_pct: float
    population_a_risque: int
    enfants_vulnerables: int

    # Météo sous-jacente
    temperature_prevue_c: float
    precipitations_prevues_mm: float

    # Explicabilité
    top_facteurs_risque: List[Dict] = Field(
        description="Top 5 facteurs toutes dimensions confondues"
    )
    recommandations_prioritaires: List[str]
    niveau_confiance: float = Field(..., ge=0, le=1,
        description="Confiance globale de la prédiction")


class PredictionBatchRequest(BaseModel):
    regions: List[str] = Field(
        ..., min_length=1, max_length=22,
        description="Liste d'IDs de régions (max 22)",
        example=["MDG-ANA", "MDG-ATS", "MDG-BOE"]
    )
    horizon_jours: int = Field(default=14, ge=1, le=90)
    inclure_shap: bool = Field(
        default=False,
        description="Inclure les valeurs SHAP (augmente le temps de réponse)"
    )


class PredictionBatchResult(BaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    horodatage: datetime
    horizon_jours: int
    total_regions: int
    regions_ok: int
    regions_erreur: List[str]
    predictions: List[PredictionCombinee]
    resume_national: Dict = Field(
        description="KPIs agrégés : score moyen, régions critiques, tendance"
    )


class ScenarioWhatIf(BaseModel):
    """Paramètres pour simulation de scénario climatique."""
    region_id: str
    horizon_jours: int = Field(default=30)

    # Overrides climatiques
    delta_temperature_c: float = Field(
        default=0.0,
        ge=-10, le=10,
        description="Variation température par rapport aux prévisions (°C)"
    )
    multiplicateur_precipitations: float = Field(
        default=1.0,
        ge=0, le=5,
        description="Multiplicateur précipitations (1.0 = normal, 2.0 = double)"
    )
    scenario_cyclone: bool = Field(
        default=False,
        description="Simuler l'impact d'un cyclone tropical"
    )
    scenario_secheresse: bool = Field(
        default=False,
        description="Simuler une sécheresse sévère (-70% précipitations)"
    )
    choc_prix_alimentaires_pct: float = Field(
        default=0.0,
        ge=-50, le=200,
        description="Choc prix alimentaires (%)"
    )


class ResultatScenario(BaseModel):
    scenario: ScenarioWhatIf
    prediction_baseline: PredictionCombinee
    prediction_scenario: PredictionCombinee
    delta_score_paludisme: float
    delta_score_nutrition: float
    cas_additionnels_paludisme: int
    enfants_additionnels_malnutris: int
    analyse_impact: str
    recommandations_scenario: List[str]


class SHAPExplication(BaseModel):
    region_id: str
    modele: str = Field(description="paludisme | nutrition")
    date_prediction: datetime
    valeur_predite: float
    valeur_base: float = Field(description="E[f(x)] — valeur moyenne du modèle")
    features: List[Dict] = Field(
        description="[{nom, valeur, shap_value, contribution_pct, direction}]"
    )
    force_plot_url: Optional[str] = Field(
        None, description="URL du SHAP force plot généré"
    )
    waterfall_url: Optional[str] = Field(
        None, description="URL du SHAP waterfall plot"
    )


class ModeleSante(BaseModel):
    modele: str
    version: str
    date_entrainement: date
    metriques: Dict = Field(
        description="AUC-ROC, F1, MAE selon le modèle"
    )
    drift_score: float = Field(
        description="Score de dérive (PSI) — seuil alerte 0.15"
    )
    statut: str = Field(
        description="optimal | surveillance | retraining_requis"
    )
    nb_predictions_7j: int
    derniere_prediction: datetime


class PerformanceBacktest(BaseModel):
    region_id: str
    periode_debut: date
    periode_fin: date
    modele: str
    mae: float = Field(description="Mean Absolute Error")
    rmse: float = Field(description="Root Mean Square Error")
    mape_pct: float = Field(description="Mean Absolute Percentage Error (%)")
    correlation: float = Field(description="Corrélation Pearson prédit/réel")
    biais: float = Field(description="Biais systématique (surestimation/sous-estimation)")
    nb_predictions: int
    predictions_vs_reel: List[Dict]


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@router.get(
    "/combinee/{region_id}",
    response_model=PredictionCombinee,
    summary="Prédiction combinée paludisme + nutrition",
    description=(
        "Pipeline complet : Météo → Paludisme → Nutrition. "
        "Génère un score composite UNICEF et une couleur d'alerte pour le dashboard. "
        "Pondération : paludisme 60% + nutrition 40%."
    ),
)
async def get_prediction_combinee(
    region_id: str = Path(..., example="MDG-BOE"),
    horizon_jours: int = Query(default=14, ge=1, le=90),
    user: AuthUser = None,
    malaria_model: MalariaModel = None,
    nutrition_model: NutritionModel = None,
    cache: Cache = None,
    db: DbSession = None,
):
    cache_key = f"predictions:combinee:{region_id}:{horizon_jours}"
    cached = await cache.get(cache_key)
    if cached:
        logger.debug("Cache HIT prédiction combinée — {}", region_id)
        return json.loads(cached)

    try:
        from src.preprocessing.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer(db)

        # Features parallèles
        malaria_features = await engineer.build_malaria_features(region_id)
        nutrition_features = await engineer.build_nutrition_features(region_id)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "FEATURES_ERREUR",
                    "message": f"Erreur préparation données : {exc}"},
        ) from exc

    # Prédictions
    try:
        mal_pred = malaria_model.predict(malaria_features, horizon_days=horizon_jours)
        nut_pred = nutrition_model.predict(nutrition_features, horizon_days=horizon_jours)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "PREDICTION_ERREUR",
                    "message": f"Erreur modèles ML : {exc}"},
        ) from exc

    # Score composite pondéré
    score_composite = (
        0.60 * mal_pred["score_risque"] + 0.40 * nut_pred["score_risque"]
    )

    # Couleur d'alerte
    couleur, niveau = _score_vers_couleur(score_composite)

    # Top facteurs risque fusionnés
    top_facteurs = _fusionner_top_facteurs(
        shap_malaria=mal_pred.get("top_contributeurs", []),
        shap_nutrition=nut_pred.get("facteurs_contributeurs", []),
    )

    # Recommandations prioritaires
    recommandations = _prioriser_recommandations(
        mal_recs=mal_pred.get("recommandations", []),
        nut_recs=nut_pred.get("recommandations", []),
        score_composite=score_composite,
        limit=5,
    )

    # Données région pour population
    import json as _json
    from pathlib import Path
    with Path("config/regions_metadata.json").open() as f:
        meta = _json.load(f)
    region_meta = next(
        (r for r in meta["regions"] if r["id"] == region_id), {}
    )
    population = region_meta.get("population_2023", 0)

    prediction = PredictionCombinee(
        region_id=region_id,
        region_name=region_meta.get("name", region_id),
        date_prediction=datetime.utcnow(),
        horizon_jours=horizon_jours,
        score_paludisme=mal_pred["score_risque"],
        niveau_paludisme=mal_pred["niveau_risque"],
        score_nutrition=nut_pred["score_risque"],
        niveau_nutrition=nut_pred["niveau_risque"],
        score_composite=round(score_composite, 4),
        niveau_alerte_global=niveau,
        couleur_carte=couleur,
        cas_paludisme_prevus_14j=mal_pred.get("cas_prevus_14j", 0),
        gam_prevu_pct=nut_pred.get("gam_prevu_pct", 0),
        population_a_risque=int(population * score_composite),
        enfants_vulnerables=int(population * 0.17 * nut_pred["score_risque"]),
        temperature_prevue_c=malaria_features.get("temperature_moy_c", 0),
        precipitations_prevues_mm=malaria_features.get("precipitations_7j_mm", 0),
        top_facteurs_risque=top_facteurs,
        recommandations_prioritaires=recommandations,
        niveau_confiance=min(
            mal_pred.get("fiabilite_modele", 0.8),
            nut_pred.get("fiabilite_modele", 0.8),
        ),
    )

    # Sauvegarde en DB (async, non bloquant)
    result_dict = prediction.model_dump()
    await cache.set(
        cache_key,
        json.dumps(result_dict, default=str),
        ttl=86400,
    )

    return prediction


@router.post(
    "/batch",
    response_model=PredictionBatchResult,
    summary="Prédictions batch — multiples régions",
    description=(
        "Calcule les prédictions combinées pour plusieurs régions en une seule requête. "
        "Maximum 22 régions (toutes les régions de Madagascar). "
        "Traitement séquentiel avec gestion des erreurs par région."
    ),
)
async def get_predictions_batch(
    request: PredictionBatchRequest,
    background_tasks: BackgroundTasks,
    user: AuthUser = None,
    malaria_model: MalariaModel = None,
    nutrition_model: NutritionModel = None,
    cache: Cache = None,
    db: DbSession = None,
):
    batch_id = str(uuid.uuid4())
    logger.info(
        "Batch prédictions — {} régions, horizon={}, user={}",
        len(request.regions), request.horizon_jours, user.username,
    )

    predictions = []
    erreurs = []

    for region_id in request.regions:
        try:
            from src.preprocessing.feature_engineering import FeatureEngineer
            engineer = FeatureEngineer(db)
            mal_f = await engineer.build_malaria_features(region_id)
            nut_f = await engineer.build_nutrition_features(region_id)

            mal_pred = malaria_model.predict(mal_f, horizon_days=request.horizon_jours)
            nut_pred = nutrition_model.predict(nut_f, horizon_days=request.horizon_jours)

            score_composite = 0.60 * mal_pred["score_risque"] + 0.40 * nut_pred["score_risque"]
            couleur, niveau = _score_vers_couleur(score_composite)

            predictions.append(
                PredictionCombinee(
                    region_id=region_id,
                    region_name=region_id,
                    date_prediction=datetime.utcnow(),
                    horizon_jours=request.horizon_jours,
                    score_paludisme=mal_pred["score_risque"],
                    niveau_paludisme=mal_pred["niveau_risque"],
                    score_nutrition=nut_pred["score_risque"],
                    niveau_nutrition=nut_pred["niveau_risque"],
                    score_composite=round(score_composite, 4),
                    niveau_alerte_global=niveau,
                    couleur_carte=couleur,
                    cas_paludisme_prevus_14j=mal_pred.get("cas_prevus_14j", 0),
                    gam_prevu_pct=nut_pred.get("gam_prevu_pct", 0),
                    population_a_risque=0,
                    enfants_vulnerables=0,
                    temperature_prevue_c=mal_f.get("temperature_moy_c", 0),
                    precipitations_prevues_mm=mal_f.get("precipitations_7j_mm", 0),
                    top_facteurs_risque=[],
                    recommandations_prioritaires=[],
                    niveau_confiance=mal_pred.get("fiabilite_modele", 0.8),
                )
            )
        except Exception as exc:
            logger.warning("Erreur batch région {} : {}", region_id, exc)
            erreurs.append(region_id)

    # Résumé national
    scores = [p.score_composite for p in predictions]
    resume = {
        "score_moyen_national": round(sum(scores) / len(scores), 4) if scores else 0,
        "regions_rouge": sum(1 for p in predictions if p.niveau_alerte_global == "rouge"),
        "regions_orange": sum(1 for p in predictions if p.niveau_alerte_global == "orange"),
        "regions_jaune": sum(1 for p in predictions if p.niveau_alerte_global == "jaune"),
        "regions_verte": sum(1 for p in predictions if p.niveau_alerte_global == "vert"),
        "region_plus_critique": max(
            predictions, key=lambda x: x.score_composite, default=None
        ) and max(predictions, key=lambda x: x.score_composite).region_id,
    }

    return PredictionBatchResult(
        batch_id=batch_id,
        horodatage=datetime.utcnow(),
        horizon_jours=request.horizon_jours,
        total_regions=len(request.regions),
        regions_ok=len(predictions),
        regions_erreur=erreurs,
        predictions=predictions,
        resume_national=resume,
    )


@router.post(
    "/scenario",
    response_model=ResultatScenario,
    summary="Simulation scénario what-if",
    description=(
        "Simule l'impact d'un scénario climatique ou économique sur les prédictions. "
        "Ex : cyclone (+200% pluies), sécheresse (-70% pluies), "
        "choc prix alimentaires (+50%). "
        "Compare baseline vs scénario avec delta quantifié."
    ),
)
async def simuler_scenario(
    scenario: ScenarioWhatIf,
    user: AuthUser = None,
    malaria_model: MalariaModel = None,
    nutrition_model: NutritionModel = None,
    db: DbSession = None,
):
    from src.preprocessing.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer(db)

    # Features baseline
    mal_f_base = await engineer.build_malaria_features(scenario.region_id)
    nut_f_base = await engineer.build_nutrition_features(scenario.region_id)

    # Prédictions baseline
    mal_base = malaria_model.predict(mal_f_base, horizon_days=scenario.horizon_jours)
    nut_base = nutrition_model.predict(nut_f_base, horizon_days=scenario.horizon_jours)
    score_base = 0.60 * mal_base["score_risque"] + 0.40 * nut_base["score_risque"]

    # Application des overrides du scénario
    mal_f_scen = dict(mal_f_base)
    nut_f_scen = dict(nut_f_base)

    if scenario.scenario_cyclone:
        mal_f_scen["precipitations_7j_mm"] *= 4.0
        mal_f_scen["precipitations_14j_mm"] *= 3.0
        mal_f_scen["humidite_moy_pct"] = min(100, mal_f_scen.get("humidite_moy_pct", 80) * 1.2)
        nut_f_scen["score_fcs"] = max(0, nut_f_scen.get("score_fcs", 35) - 15)
        nut_f_scen["variation_prix_pct_1m"] = 40.0
        logger.info("Scénario cyclone appliqué sur {}", scenario.region_id)

    elif scenario.scenario_secheresse:
        mal_f_scen["precipitations_7j_mm"] *= 0.3
        mal_f_scen["precipitations_30j_mm"] *= 0.3
        nut_f_scen["score_fcs"] = max(0, nut_f_scen.get("score_fcs", 35) - 10)
        nut_f_scen["variation_prix_pct_1m"] = 30.0
        logger.info("Scénario sécheresse appliqué sur {}", scenario.region_id)

    else:
        mal_f_scen["temperature_moy_c"] = (
            mal_f_scen.get("temperature_moy_c", 25) + scenario.delta_temperature_c
        )
        mal_f_scen["precipitations_7j_mm"] = (
            mal_f_scen.get("precipitations_7j_mm", 0) * scenario.multiplicateur_precipitations
        )

    if scenario.choc_prix_alimentaires_pct != 0:
        nut_f_scen["variation_prix_pct_1m"] = scenario.choc_prix_alimentaires_pct
        nut_f_scen["score_fcs"] = max(
            0,
            nut_f_scen.get("score_fcs", 35) - scenario.choc_prix_alimentaires_pct * 0.2,
        )

    # Prédictions scénario
    mal_scen = malaria_model.predict(mal_f_scen, horizon_days=scenario.horizon_jours)
    nut_scen = nutrition_model.predict(nut_f_scen, horizon_days=scenario.horizon_jours)
    score_scen = 0.60 * mal_scen["score_risque"] + 0.40 * nut_scen["score_risque"]

    couleur_base, niveau_base = _score_vers_couleur(score_base)
    couleur_scen, niveau_scen = _score_vers_couleur(score_scen)

    # Construction objets PredictionCombinee
    pred_base = PredictionCombinee(
        region_id=scenario.region_id,
        region_name=scenario.region_id,
        date_prediction=datetime.utcnow(),
        horizon_jours=scenario.horizon_jours,
        score_paludisme=mal_base["score_risque"],
        niveau_paludisme=mal_base["niveau_risque"],
        score_nutrition=nut_base["score_risque"],
        niveau_nutrition=nut_base["niveau_risque"],
        score_composite=round(score_base, 4),
        niveau_alerte_global=niveau_base,
        couleur_carte=couleur_base,
        cas_paludisme_prevus_14j=mal_base.get("cas_prevus_14j", 0),
        gam_prevu_pct=nut_base.get("gam_prevu_pct", 0),
        population_a_risque=0,
        enfants_vulnerables=0,
        temperature_prevue_c=mal_f_base.get("temperature_moy_c", 0),
        precipitations_prevues_mm=mal_f_base.get("precipitations_7j_mm", 0),
        top_facteurs_risque=[],
        recommandations_prioritaires=[],
        niveau_confiance=0.85,
    )

    pred_scen = PredictionCombinee(
        region_id=scenario.region_id,
        region_name=scenario.region_id,
        date_prediction=datetime.utcnow(),
        horizon_jours=scenario.horizon_jours,
        score_paludisme=mal_scen["score_risque"],
        niveau_paludisme=mal_scen["niveau_risque"],
        score_nutrition=nut_scen["score_risque"],
        niveau_nutrition=nut_scen["niveau_risque"],
        score_composite=round(score_scen, 4),
        niveau_alerte_global=niveau_scen,
        couleur_carte=couleur_scen,
        cas_paludisme_prevus_14j=mal_scen.get("cas_prevus_14j", 0),
        gam_prevu_pct=nut_scen.get("gam_prevu_pct", 0),
        population_a_risque=0,
        enfants_vulnerables=0,
        temperature_prevue_c=mal_f_scen.get("temperature_moy_c", 0),
        precipitations_prevues_mm=mal_f_scen.get("precipitations_7j_mm", 0),
        top_facteurs_risque=[],
        recommandations_prioritaires=[],
        niveau_confiance=0.75,
    )

    analyse = _analyser_impact_scenario(
        score_base=score_base,
        score_scen=score_scen,
        delta_mal=mal_scen["score_risque"] - mal_base["score_risque"],
        delta_nut=nut_scen["score_risque"] - nut_base["score_risque"],
        scenario=scenario,
    )

    return ResultatScenario(
        scenario=scenario,
        prediction_baseline=pred_base,
        prediction_scenario=pred_scen,
        delta_score_paludisme=round(
            mal_scen["score_risque"] - mal_base["score_risque"], 4
        ),
        delta_score_nutrition=round(
            nut_scen["score_risque"] - nut_base["score_risque"], 4
        ),
        cas_additionnels_paludisme=max(
            0,
            mal_scen.get("cas_prevus_14j", 0) - mal_base.get("cas_prevus_14j", 0),
        ),
        enfants_additionnels_malnutris=int(
            max(0, nut_scen.get("gam_prevu_pct", 0) - nut_base.get("gam_prevu_pct", 0))
            * 1000  # approximatif
        ),
        analyse_impact=analyse,
        recommandations_scenario=_recommandations_scenario(scenario, score_scen),
    )


@router.get(
    "/explicabilite/{region_id}/{modele}",
    response_model=SHAPExplication,
    summary="Valeurs SHAP — explicabilité du modèle",
    description=(
        "Retourne les valeurs SHAP détaillées pour comprendre pourquoi le modèle "
        "a prédit ce score. Requis par les standards UNICEF de transparence algorithmique. "
        "Modèles disponibles : paludisme | nutrition"
    ),
)
async def get_shap_explication(
    region_id: str = Path(..., example="MDG-ATS"),
    modele: str = Path(..., description="paludisme | nutrition"),
    user: AuthUser = None,
    malaria_model: MalariaModel = None,
    nutrition_model: NutritionModel = None,
    cache: Cache = None,
    db: DbSession = None,
):
    if modele not in ("paludisme", "nutrition"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "MODELE_INVALIDE",
                "message": "Modèle doit être 'paludisme' ou 'nutrition'.",
            },
        )

    cache_key = f"predictions:shap:{region_id}:{modele}:{date.today()}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.preprocessing.feature_engineering import FeatureEngineer
    from src.models.explainability import SHAPExplainer

    engineer = FeatureEngineer(db)

    if modele == "paludisme":
        features = await engineer.build_malaria_features(region_id)
        explainer = SHAPExplainer(malaria_model)
    else:
        features = await engineer.build_nutrition_features(region_id)
        explainer = SHAPExplainer(nutrition_model)

    shap_result = explainer.explain(features, region_id=region_id)
    await cache.set(
        cache_key, json.dumps(shap_result, default=str), ttl=43200
    )
    return shap_result


@router.get(
    "/sante-modeles",
    response_model=List[ModeleSante],
    summary="Santé et performances des modèles ML",
    description=(
        "Retourne les métriques de performance et le score de dérive (PSI) "
        "pour chaque modèle. Alerte si dérive > 0.15 (retraining requis). "
        "Accès restreint : admin/national."
    ),
    dependencies=[NationalOrAdmin],
)
async def get_sante_modeles(
    user: AuthUser = None,
    cache: Cache = None,
    malaria_model: MalariaModel = None,
    nutrition_model: NutritionModel = None,
):
    cache_key = f"predictions:sante:{date.today()}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    sante = []
    for model, nom in [(malaria_model, "paludisme"), (nutrition_model, "nutrition")]:
        try:
            info = model.get_health_info()
            sante.append(
                {
                    "modele": nom,
                    "version": info.get("version", "1.0"),
                    "date_entrainement": info.get("date_entrainement"),
                    "metriques": info.get("metriques", {}),
                    "drift_score": info.get("drift_score", 0.0),
                    "statut": (
                        "retraining_requis"
                        if info.get("drift_score", 0) > 0.15
                        else "surveillance"
                        if info.get("drift_score", 0) > 0.10
                        else "optimal"
                    ),
                    "nb_predictions_7j": info.get("nb_predictions_7j", 0),
                    "derniere_prediction": info.get("derniere_prediction"),
                }
            )
        except Exception as exc:
            logger.warning("Erreur info modèle {} : {}", nom, exc)

    await cache.set(cache_key, json.dumps(sante, default=str), ttl=3600)
    return sante


@router.get(
    "/backtest/{region_id}",
    response_model=PerformanceBacktest,
    summary="Backtesting — prédictions vs réalité",
    description=(
        "Compare les prédictions historiques aux valeurs réellement observées. "
        "Permet d'évaluer la qualité du modèle sur une région spécifique. "
        "Accès restreint : admin/national."
    ),
    dependencies=[NationalOrAdmin],
)
async def get_backtest(
    region_id: str = Path(..., example="MDG-ANA"),
    modele: str = Query(default="paludisme",
        description="paludisme | nutrition"),
    periode_mois: int = Query(default=6, ge=1, le=24),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"predictions:backtest:{region_id}:{modele}:{periode_mois}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.malaria_repo import MalariaRepository
    from src.database.repositories.nutrition_repo import NutritionRepository

    date_fin = date.today()
    date_debut = date_fin - timedelta(days=periode_mois * 30)

    if modele == "paludisme":
        repo = MalariaRepository(db)
        backtest = await repo.get_backtest_data(region_id, date_debut, date_fin)
    else:
        repo = NutritionRepository(db)
        backtest = await repo.get_backtest_data(region_id, date_debut, date_fin)

    await cache.set(cache_key, json.dumps(backtest, default=str), ttl=86400)
    return backtest


@router.post(
    "/forcer-retraining/{modele}",
    summary="Forcer le retraining d'un modèle",
    description=(
        "Déclenche manuellement le pipeline de retraining. "
        "Accès restreint : admin uniquement."
    ),
    dependencies=[AdminOnly],
)
async def forcer_retraining(
    modele: str = Path(..., description="paludisme | nutrition | tous"),
    background_tasks: BackgroundTasks = None,
    user: AuthUser = None,
):
    if modele not in ("paludisme", "nutrition", "tous"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "MODELE_INVALIDE",
                    "message": "Modèle : paludisme | nutrition | tous"},
        )

    # Lance le retraining en arrière-plan
    from src.data_collection.scheduler import trigger_retraining
    background_tasks.add_task(trigger_retraining, modele=modele)

    logger.info("Retraining forcé — modèle={} par {}", modele, user.username)
    return {
        "statut": "démarré",
        "message": f"Retraining du modèle '{modele}' lancé en arrière-plan.",
        "declenche_par": user.username,
        "horodatage": datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────

def _score_vers_couleur(score: float) -> tuple[str, str]:
    """Convertit un score 0-1 en couleur hex et niveau d'alerte."""
    if score >= 0.75:
        return "#D32F2F", "rouge"       # Crise
    elif score >= 0.50:
        return "#F57C00", "orange"      # Urgent
    elif score >= 0.25:
        return "#F9A825", "jaune"       # Alerte
    else:
        return "#388E3C", "vert"        # Normal


def _fusionner_top_facteurs(
    shap_malaria: List[Dict],
    shap_nutrition: List[Dict],
    limit: int = 5,
) -> List[Dict]:
    """
    Fusionne et trie les top features SHAP des deux modèles.
    Préfixe chaque feature avec son modèle d'origine.
    """
    combined = []
    for f in shap_malaria[:3]:
        combined.append({**f, "modele": "paludisme"})
    for f in shap_nutrition[:3]:
        combined.append({**f, "modele": "nutrition"})

    combined.sort(key=lambda x: abs(x.get("shap_value", 0)), reverse=True)
    return combined[:limit]


def _prioriser_recommandations(
    mal_recs: List[str],
    nut_recs: List[str],
    score_composite: float,
    limit: int = 5,
) -> List[str]:
    """
    Fusionne et déduplique les recommandations des deux modèles.
    Priorise celles commençant par 🚨 puis ⚠️.
    """
    all_recs = mal_recs + nut_recs
    # Tri : urgences d'abord
    priority_order = {"🚨": 0, "⚠": 1, "📊": 2, "✅": 3, "🌧": 2, "🌿": 2, "🌡": 2}
    all_recs.sort(
        key=lambda r: next(
            (v for k, v in priority_order.items() if r.startswith(k)), 4
        )
    )
    # Déduplique en préservant l'ordre
    seen = set()
    unique = []
    for r in all_recs:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique[:limit]


def _analyser_impact_scenario(
    score_base: float,
    score_scen: float,
    delta_mal: float,
    delta_nut: float,
    scenario: ScenarioWhatIf,
) -> str:
    delta = score_scen - score_base
    pct = delta / score_base * 100 if score_base > 0 else 0

    if scenario.scenario_cyclone:
        type_evenement = "cyclone tropical"
    elif scenario.scenario_secheresse:
        type_evenement = "sécheresse sévère"
    else:
        type_evenement = "modification climatique"

    direction = "aggrave" if delta > 0 else "améliore"
    return (
        f"Le scénario '{type_evenement}' {direction} la situation de {abs(pct):.1f}% "
        f"(score composite {score_base:.3f} → {score_scen:.3f}). "
        f"Impact paludisme : {delta_mal:+.3f} | Impact nutrition : {delta_nut:+.3f}."
    )


def _recommandations_scenario(
    scenario: ScenarioWhatIf,
    score_scen: float,
) -> List[str]:
    recs = []
    if scenario.scenario_cyclone:
        recs += [
            "Pré-positionner les équipes de réponse d'urgence avant le cyclone.",
            "Évacuer les populations des zones à risque d'inondation.",
            "Préparer les stocks d'urgence (médicaments antipaludéens, RUTF).",
            "Activer le plan de contingence UNICEF — Cluster Santé + Nutrition.",
        ]
    elif scenario.scenario_secheresse:
        recs += [
            "Lancer un programme d'urgence de transferts monétaires (cash transfer).",
            "Activer les stocks stratégiques d'alimentation thérapeutique.",
            "Mettre en place des jardins potagers d'urgence avec irrigation.",
        ]
    if score_scen > 0.75:
        recs.append("Déclencher le niveau de réponse ROUGE — mobilisation nationale requise.")
    return recs