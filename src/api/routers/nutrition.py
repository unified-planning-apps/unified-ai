"""
src/api/routers/nutrition.py
=============================
Endpoints nutrition et sécurité alimentaire.

Couvre :
- Statut nutritionnel actuel par région (GAM, SAM, MAM)
- Prédiction risque malnutrition (modèle Ensemble ML)
- Disponibilité et prix des denrées alimentaires
- Groupes de population vulnérables (enfants < 5 ans, femmes enceintes)
- Score de diversité alimentaire (HDDS)
- Recettes contextualisées par région et saison
- Alertes malnutrition aiguë
- Stocks humanitaires (RUTF, suppléments)
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Path, Query, status
from loguru import logger
from pydantic import BaseModel, Field

from src.api.dependencies import (
    AdminOnly,
    AuthUser,
    Cache,
    DbSession,
    NationalOrAdmin,
    NutritionModel,
    Pagination,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────────────────────────

class StatutNutritionnel(BaseModel):
    region_id: str
    region_name: str
    date_enquete: date
    source: str = Field(description="SMART survey | UNICEF | MICS | DHIS2")

    # Taux malnutrition aiguë globale (Global Acute Malnutrition)
    gam_pct: float = Field(..., ge=0, le=100,
        description="Taux MAG — enfants < 5 ans (poids/taille < -2 Z-score)")
    # Malnutrition aiguë sévère
    sam_pct: float = Field(..., ge=0, le=100,
        description="Taux MAS — enfants < 5 ans (Z-score < -3 ou œdèmes)")
    # Malnutrition aiguë modérée
    mam_pct: float = Field(..., ge=0, le=100,
        description="Taux MAM — enfants < 5 ans (-3 ≤ Z-score < -2)")
    # Malnutrition chronique (stunting)
    stunting_pct: float = Field(..., ge=0, le=100,
        description="Retard de croissance — taille/âge < -2 Z-score")
    # Insuffisance pondérale
    underweight_pct: float = Field(..., ge=0, le=100)

    enfants_5ans_affectes: int
    femmes_enceintes_malnutries: int
    classification_who: str = Field(
        description="acceptable (<5%) | alerte (5-10%) | urgence (10-15%) | crise (>15%)"
    )
    tendance_vs_periode_prec: str = Field(
        description="amélioration | stable | dégradation"
    )
    fiabilite_donnees: str = Field(default="confirmée")


class PredictionRisqueNutrition(BaseModel):
    region_id: str
    region_name: str
    score_risque: float = Field(..., ge=0, le=1)
    niveau_risque: str
    gam_prevu_pct: float = Field(description="GAM prévu sur l'horizon")
    sam_prevu_pct: float
    date_prediction: datetime
    horizon_jours: int
    intervalles_confiance: Dict = Field(
        description="{'bas': float, 'haut': float} pour GAM et SAM"
    )
    facteurs_contributeurs: List[Dict] = Field(
        description="Features SHAP + valeur + contribution"
    )
    populations_vulnerables: List[Dict] = Field(
        description="Groupes à risque identifiés avec effectifs estimés"
    )
    recommandations: List[str]
    interventions_prioritaires: List[str]


class DisponibiliteAlimentaire(BaseModel):
    region_id: str
    date_observation: date
    score_fcs: float = Field(..., ge=0, le=112,
        description="Food Consumption Score (FCS) WFP — optimal > 42")
    classification_fcs: str = Field(
        description="pauvre (<21) | limite (21-35) | acceptable (>35)"
    )
    hdds: float = Field(..., ge=0, le=12,
        description="Household Dietary Diversity Score (0-12 groupes)")
    rcsi: float = Field(ge=0,
        description="Reduced Coping Strategies Index — stress alimentaire")

    # Prix denrées de base (USD/kg ou local)
    prix_riz_kg: Optional[float] = None
    prix_manioc_kg: Optional[float] = None
    prix_mais_kg: Optional[float] = None
    prix_haricots_kg: Optional[float] = None
    prix_huile_litre: Optional[float] = None
    variation_prix_pct_1m: Optional[float] = Field(
        None, description="Variation des prix sur 1 mois (%)"
    )

    # Disponibilité par groupe alimentaire (échelle 0-3)
    disponibilite_cereales: int = Field(..., ge=0, le=3)
    disponibilite_legumineuses: int = Field(..., ge=0, le=3)
    disponibilite_proteines_animales: int = Field(..., ge=0, le=3)
    disponibilite_legumes: int = Field(..., ge=0, le=3)
    disponibilite_fruits: int = Field(..., ge=0, le=3)

    source: str = Field(default="WFP VAM")


class RecetteNutritionnelle(BaseModel):
    recette_id: str
    nom: str
    nom_malgache: Optional[str] = None
    region_adaptee: List[str] = Field(description="Régions pour lesquelles adaptée")
    saison: List[str] = Field(description="saison_pluies | saison_seche | toute_annee")

    # Valeurs nutritionnelles (par portion)
    calories_kcal: float
    proteines_g: float
    glucides_g: float
    lipides_g: float
    fer_mg: float
    vitamine_a_ug: float
    zinc_mg: float
    score_nutritionnel: float = Field(..., ge=0, le=100,
        description="Score composite UNICEF 0-100")

    ingredients: List[Dict] = Field(
        description="[{nom, quantite_g, disponible_localement}]"
    )
    instructions: str
    temps_preparation_min: int
    cout_estime_ariary: Optional[float] = None
    cible: List[str] = Field(
        description="enfants_6_23m | enfants_2_5ans | femmes_enceintes | famille"
    )
    image_url: Optional[str] = None


class StockHumanitaire(BaseModel):
    region_id: str
    date_inventaire: date
    rutf_sachets: int = Field(description="Ready-to-Use Therapeutic Food (sachets)")
    rusf_sachets: int = Field(description="Ready-to-Use Supplementary Food")
    plumpy_nut_sachets: int
    spiruline_kg: float
    sel_iode_kg: float
    vitamine_a_capsules: int
    fer_folate_comprimes: int
    zinc_comprimes: int
    jours_couverture_sam: float = Field(
        description="Jours de couverture SAM avec stock actuel"
    )
    jours_couverture_mam: float
    statut_stock: str = Field(
        description="adéquat | alerte | rupture_imminente | rupture"
    )
    derniere_livraison: Optional[date] = None
    prochaine_livraison_prevue: Optional[date] = None


class AlerteNutrition(BaseModel):
    alerte_id: str
    region_id: str
    region_name: str
    type_alerte: str = Field(
        description=(
            "seuil_gam_depasse | tendance_degradation | "
            "rupture_stock_rutf | choc_alimentaire | "
            "saison_soudure_critique"
        )
    )
    severite: str = Field(description="surveillance | alerte | urgence | crise")
    indicateur_declencheur: str
    valeur_actuelle: float
    seuil_alerte: float
    population_affectee: int
    enfants_a_risque: int
    date_detection: datetime
    statut: str = Field(default="active")
    actions_requises: List[str]


class SaisonSoudure(BaseModel):
    region_id: str
    region_name: str
    en_periode_soudure: bool
    semaines_avant_soudure: Optional[int] = None
    duree_soudure_historique_semaines: int
    niveau_risque_soudure: str
    denrees_principales_affectees: List[str]
    strategies_coping_observees: List[str]


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@router.get(
    "/statut/{region_id}",
    response_model=StatutNutritionnel,
    summary="Statut nutritionnel actuel d'une région",
    description=(
        "Retourne les taux de malnutrition aiguë (GAM, SAM, MAM), retard de croissance "
        "et insuffisance pondérale pour les enfants < 5 ans. "
        "Source : enquêtes SMART + DHIS2 Madagascar."
    ),
)
async def get_statut_nutritionnel(
    region_id: str = Path(..., example="MDG-AND",
        description="ID région Madagascar"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"nutrition:statut:{region_id}:{date.today().strftime('%Y-%W')}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.nutrition_repo import NutritionRepository
    repo = NutritionRepository(db)
    statut = await repo.get_statut_actuel(region_id)

    if not statut:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "STATUT_INTROUVABLE",
                "message": f"Aucune donnée nutritionnelle disponible pour {region_id}.",
            },
        )

    await cache.set(cache_key, json.dumps(statut, default=str), ttl=86400)
    return statut


@router.get(
    "/risque/{region_id}",
    response_model=PredictionRisqueNutrition,
    summary="Prédiction risque malnutrition — modèle ML",
    description=(
        "Prédit le risque de malnutrition aiguë pour une région. "
        "Le modèle Ensemble (Random Forest + Neural Network) intègre : "
        "prédiction paludisme, disponibilité alimentaire, prix, saisonnalité, "
        "indicateurs socio-économiques. Inclut les SHAP values."
    ),
)
async def get_risque_nutrition(
    region_id: str = Path(..., example="MDG-AND"),
    horizon_jours: int = Query(default=30, ge=7, le=90,
        description="Horizon de prédiction"),
    user: AuthUser = None,
    model: NutritionModel = None,
    cache: Cache = None,
    db: DbSession = None,
):
    cache_key = f"nutrition:risque:{region_id}:{horizon_jours}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Construction features
    try:
        from src.preprocessing.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer(db)
        features = await engineer.build_nutrition_features(region_id)
    except Exception as exc:
        logger.error("Erreur features nutrition {} : {}", region_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "FEATURES_ERREUR",
                    "message": "Impossible de préparer les features nutrition."},
        ) from exc

    # Prédiction
    try:
        prediction = model.predict(features, horizon_days=horizon_jours)
    except Exception as exc:
        logger.error("Erreur prédiction nutrition {} : {}", region_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "PREDICTION_ERREUR",
                    "message": "Erreur lors de la prédiction nutrition."},
        ) from exc

    prediction["recommandations"] = _generer_recommandations_nutrition(
        score=prediction["score_risque"],
        gam_prevu=prediction.get("gam_prevu_pct", 0),
        features=features,
    )
    prediction["interventions_prioritaires"] = _get_interventions_prioritaires(
        region_id=region_id,
        gam_prevu=prediction.get("gam_prevu_pct", 0),
    )

    await cache.set(cache_key, json.dumps(prediction, default=str), ttl=86400)
    return prediction


@router.get(
    "/disponibilite/{region_id}",
    response_model=DisponibiliteAlimentaire,
    summary="Disponibilité et prix alimentaires",
    description=(
        "Score de consommation alimentaire (FCS), diversité alimentaire (HDDS), "
        "indice de stratégies de survie (rCSI) et prix des denrées de base. "
        "Source : WFP VAM + marchés locaux."
    ),
)
async def get_disponibilite_alimentaire(
    region_id: str = Path(..., example="MDG-ASO"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"nutrition:disponibilite:{region_id}:{date.today()}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.nutrition_repo import NutritionRepository
    repo = NutritionRepository(db)
    dispo = await repo.get_disponibilite(region_id)

    if not dispo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "DISPO_INTROUVABLE",
                "message": f"Données de disponibilité alimentaire non disponibles pour {region_id}.",
            },
        )

    await cache.set(cache_key, json.dumps(dispo, default=str), ttl=43200)
    return dispo


@router.get(
    "/recettes",
    response_model=List[RecetteNutritionnelle],
    summary="Recettes nutritionnelles adaptées",
    description=(
        "Retourne des recettes nutritionnelles adaptées au contexte local : "
        "région, saison, ingrédients disponibles, groupe cible. "
        "Optimisées pour couvrir les besoins nutritionnels des enfants < 5 ans."
    ),
)
async def get_recettes(
    region_id: Optional[str] = Query(None, description="Filtrer par région"),
    saison: Optional[str] = Query(
        None,
        description="saison_pluies | saison_seche | toute_annee",
    ),
    cible: Optional[str] = Query(
        None,
        description="enfants_6_23m | enfants_2_5ans | femmes_enceintes | famille",
    ),
    score_min: float = Query(default=60.0, ge=0, le=100,
        description="Score nutritionnel minimum"),
    limit: int = Query(default=10, ge=1, le=50),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = (
        f"nutrition:recettes:{region_id or 'all'}:{saison or 'all'}"
        f":{cible or 'all'}:{score_min}:{limit}"
    )
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.nutrition_repo import NutritionRepository
    repo = NutritionRepository(db)
    recettes = await repo.get_recettes(
        region_id=region_id,
        saison=saison,
        cible=cible,
        score_min=score_min,
        limit=limit,
    )

    await cache.set(cache_key, json.dumps(recettes, default=str), ttl=86400)
    return recettes


@router.get(
    "/recettes/{recette_id}",
    response_model=RecetteNutritionnelle,
    summary="Détail d'une recette nutritionnelle",
)
async def get_recette_detail(
    recette_id: str = Path(..., description="ID de la recette"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"nutrition:recette:{recette_id}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.nutrition_repo import NutritionRepository
    repo = NutritionRepository(db)
    recette = await repo.get_recette_by_id(recette_id)

    if not recette:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RECETTE_INTROUVABLE",
                    "message": f"Recette {recette_id} introuvable."},
        )

    await cache.set(cache_key, json.dumps(recette, default=str), ttl=604800)  # 1 sem
    return recette


@router.get(
    "/recettes/generer/{region_id}",
    response_model=List[RecetteNutritionnelle],
    summary="Générer des recettes contextualisées par algorithme",
    description=(
        "Algorithme de sélection intelligent : "
        "(1) interroge les ingrédients disponibles en saison, "
        "(2) optimise les valeurs nutritionnelles par Linear Programming, "
        "(3) respecte les contraintes culturelles régionales."
    ),
)
async def generer_recettes_contextuelles(
    region_id: str = Path(..., example="MDG-AND"),
    cible: str = Query(
        default="enfants_6_23m",
        description="Groupe cible : enfants_6_23m | enfants_2_5ans | femmes_enceintes | famille",
    ),
    nombre: int = Query(default=5, ge=1, le=20),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = (
        f"nutrition:recettes:gen:{region_id}:{cible}"
        f":{date.today().strftime('%Y-%W')}:{nombre}"
    )
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.reports.recipe_selector import RecipeSelector
    selector = RecipeSelector(db)
    recettes = await selector.generer_recettes_optimales(
        region_id=region_id,
        cible=cible,
        nombre=nombre,
    )

    await cache.set(cache_key, json.dumps(recettes, default=str), ttl=86400)
    return recettes


@router.get(
    "/stocks/{region_id}",
    response_model=StockHumanitaire,
    summary="Stocks humanitaires (RUTF, RUSF, micronutriments)",
    description=(
        "Retourne l'inventaire des stocks humanitaires nutritionnels d'une région : "
        "RUTF, RUSF, Plumpy'Nut, suppléments en micronutriments. "
        "Calcule les jours de couverture estimés."
    ),
    dependencies=[NationalOrAdmin],
)
async def get_stocks_humanitaires(
    region_id: str = Path(..., example="MDG-SAV"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"nutrition:stocks:{region_id}:{date.today()}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.nutrition_repo import NutritionRepository
    repo = NutritionRepository(db)
    stocks = await repo.get_stocks(region_id)

    await cache.set(cache_key, json.dumps(stocks, default=str), ttl=3600)
    return stocks


@router.get(
    "/alertes",
    response_model=List[AlerteNutrition],
    summary="Alertes malnutrition actives",
    description=(
        "Alertes nutrition actives : dépassements de seuils GAM, "
        "tendances de dégradation, ruptures de stock RUTF, "
        "chocs alimentaires, périodes de soudure critiques."
    ),
)
async def get_alertes_nutrition(
    region_id: Optional[str] = Query(None),
    type_alerte: Optional[str] = Query(None),
    severite: Optional[str] = Query(None),
    statut: str = Query(default="active"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = (
        f"nutrition:alertes:{region_id or 'all'}"
        f":{type_alerte or 'all'}:{severite or 'all'}:{statut}"
    )
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.nutrition_repo import NutritionRepository
    repo = NutritionRepository(db)
    alertes = await repo.get_alertes(
        region_id=region_id,
        type_alerte=type_alerte,
        severite=severite,
        statut=statut,
    )

    await cache.set(cache_key, json.dumps(alertes, default=str), ttl=900)
    return alertes


@router.get(
    "/soudure",
    response_model=List[SaisonSoudure],
    summary="Suivi des périodes de soudure",
    description=(
        "Identifie les régions en période de soudure alimentaire "
        "(période entre deux récoltes où la disponibilité alimentaire est minimale). "
        "Critique pour Madagascar : Grand Sud principalement."
    ),
)
async def get_saison_soudure(
    region_id: Optional[str] = Query(None,
        description="Filtrer par région (toutes si absent)"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"nutrition:soudure:{region_id or 'all'}:{date.today().strftime('%Y-%m')}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.nutrition_repo import NutritionRepository
    repo = NutritionRepository(db)
    soudure_data = await repo.get_saison_soudure(region_id)

    await cache.set(cache_key, json.dumps(soudure_data, default=str), ttl=86400)
    return soudure_data


@router.get(
    "/tendance/{region_id}",
    summary="Tendance GAM mensuelle — courbe temporelle",
    description=(
        "Retourne la courbe d'évolution du taux GAM sur N mois "
        "avec la tendance lissée et les seuils OMS."
    ),
)
async def get_tendance_nutrition(
    region_id: str = Path(..., example="MDG-ASO"),
    mois: int = Query(default=24, ge=6, le=60,
        description="Nombre de mois à retourner"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"nutrition:tendance:{region_id}:{mois}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    date_fin = date.today()
    date_debut = date_fin - timedelta(days=mois * 30)

    from src.database.repositories.nutrition_repo import NutritionRepository
    repo = NutritionRepository(db)
    data = await repo.get_gam_trend(region_id, date_debut, date_fin)

    # Enrichissement : seuils OMS
    for point in data:
        gam = point.get("gam_pct", 0)
        point["seuil_oms"] = (
            "acceptable" if gam < 5
            else "alerte" if gam < 10
            else "urgence" if gam < 15
            else "crise"
        )

    result = {
        "region_id": region_id,
        "mois": mois,
        "data": data,
        "gam_actuel": data[-1].get("gam_pct") if data else None,
        "gam_moyen_periode": (
            sum(d.get("gam_pct", 0) for d in data) / len(data)
            if data else None
        ),
        "seuils_oms": {
            "acceptable": 5.0,
            "alerte": 10.0,
            "urgence": 15.0,
        },
    }

    await cache.set(cache_key, json.dumps(result, default=str), ttl=86400)
    return result


@router.get(
    "/carte-risque",
    summary="Carte de risque nutrition nationale — 22 régions",
    description="Score de risque malnutrition + GAM actuel pour toutes les régions.",
)
async def get_carte_risque_nutrition(
    user: AuthUser = None,
    model: NutritionModel = None,
    cache: Cache = None,
    db: DbSession = None,
):
    cache_key = f"nutrition:carte:{date.today().strftime('%Y-%W')}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    import json as _json
    from pathlib import Path

    with Path("config/regions_metadata.json").open() as f:
        meta = _json.load(f)

    from src.preprocessing.feature_engineering import FeatureEngineer
    from src.database.repositories.nutrition_repo import NutritionRepository

    engineer = FeatureEngineer(db)
    repo = NutritionRepository(db)
    carte = []

    for region in meta["regions"]:
        rid = region["id"]
        try:
            features = await engineer.build_nutrition_features(rid)
            pred = model.predict(features, horizon_days=30)
            statut = await repo.get_statut_actuel(rid)

            carte.append(
                {
                    "region_id": rid,
                    "region_name": region["name"],
                    "latitude": region["latitude"],
                    "longitude": region["longitude"],
                    "score_risque": pred["score_risque"],
                    "niveau_risque": pred["niveau_risque"],
                    "gam_actuel_pct": statut.get("gam_pct") if statut else None,
                    "gam_prevu_pct": pred.get("gam_prevu_pct"),
                    "population": region["population_2023"],
                    "food_insecurity_risk": region.get("food_insecurity_risk", "normal"),
                }
            )
        except Exception as exc:
            logger.warning("Erreur carte nutrition {} : {}", rid, exc)

    result = {
        "carte": carte,
        "genere_le": datetime.utcnow().isoformat(),
        "regions_ok": len(carte),
    }
    await cache.set(cache_key, json.dumps(result, default=str), ttl=86400)
    return result


@router.post(
    "/stocks/{region_id}",
    summary="Mettre à jour les stocks humanitaires",
    description="Enregistre un nouvel inventaire de stocks humanitaires. Admin/National uniquement.",
    dependencies=[NationalOrAdmin],
    status_code=status.HTTP_201_CREATED,
)
async def update_stocks(
    region_id: str = Path(...),
    stock_data: StockHumanitaire = ...,
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    from src.database.repositories.nutrition_repo import NutritionRepository
    repo = NutritionRepository(db)
    created = await repo.save_stocks(region_id, stock_data.model_dump())

    await cache.invalidate_pattern(f"nutrition:stocks:{region_id}*")

    logger.info(
        "Stocks mis à jour — région={} user={} jours_couv_sam={}",
        region_id, user.username, stock_data.jours_couverture_sam
    )
    return {
        "statut": "créé",
        "region_id": region_id,
        "inventaire_id": created.get("id"),
        "message": "Stocks enregistrés avec succès.",
    }


@router.get(
    "/statistiques/national",
    summary="Tableau de bord nutrition national",
    description=(
        "KPIs nationaux : régions en crise (GAM > 15%), "
        "total enfants malnutris, stocks critiques, alertes actives."
    ),
    dependencies=[NationalOrAdmin],
)
async def get_statistiques_nationales(
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"nutrition:stats:national:{date.today().strftime('%Y-%W')}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.nutrition_repo import NutritionRepository
    repo = NutritionRepository(db)
    stats = await repo.get_national_stats()

    await cache.set(cache_key, json.dumps(stats, default=str), ttl=3600)
    return stats


# ─────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────

def _generer_recommandations_nutrition(
    score: float,
    gam_prevu: float,
    features: dict,
) -> List[str]:
    recs = []

    if gam_prevu >= 15.0:
        recs += [
            "Situation de CRISE nutritionnelle (GAM ≥ 15%) — réponse d'urgence requise.",
            "Déclencher le protocole de prise en charge intégrée de la malnutrition aiguë (PCIMA).",
            "Mobiliser les stocks RUTF / Plumpy'Nut en urgence.",
            "Activer les Unités de Nutrition Thérapeutique Ambulatoire (UNTA).",
            "Notifier UNICEF et le Cluster Nutrition pour appui logistique.",
        ]
    elif gam_prevu >= 10.0:
        recs += [
            "Seuil d'URGENCE nutrition (GAM ≥ 10%) — intervention renforcée.",
            "Intensifier le dépistage actif de la malnutrition dans les CSB.",
            "Lancer des distributions de suppléments alimentaires (RUSF, Plumpy'Sup).",
            "Renforcer les activités ANJE (Alimentation du Nourrisson et du Jeune Enfant).",
        ]
    elif gam_prevu >= 5.0:
        recs += [
            "Seuil d'ALERTE nutrition (GAM ≥ 5%) — surveillance renforcée.",
            "Organiser des séances de nutrition communautaire (SEECALINE).",
            "Promouvoir la diversification alimentaire et les pratiques ANJE.",
        ]
    else:
        recs += [
            "Situation nutritionnelle acceptable — maintien des activités de routine.",
            "Poursuivre le suivi de la croissance (pesées mensuelles).",
        ]

    # Recommandations contextuelles
    fcs = features.get("score_fcs", 35)
    if fcs < 21:
        recs.append(
            f"Score de consommation alimentaire très faible (FCS={fcs:.1f}) — "
            "cibler les ménages en insécurité alimentaire sévère."
        )

    prix_var = features.get("variation_prix_pct_1m", 0)
    if prix_var and prix_var > 20:
        recs.append(
            f"Inflation alimentaire rapide (+{prix_var:.1f}% sur 1 mois) — "
            "envisager des transferts monétaires ciblés."
        )

    return recs


def _get_interventions_prioritaires(
    region_id: str,
    gam_prevu: float,
) -> List[str]:
    interventions = []

    if gam_prevu >= 15.0:
        interventions += [
            "Traitement thérapeutique ambulatoire (UNTA) — enfants MAG sévère",
            "Distribution d'urgence RUTF (Plumpy'Nut 92g/sachet)",
            "Suivi hospitalier (UNTI) pour cas compliqués",
            "Blanket Supplementary Feeding Program (BSFP) — <5 ans",
        ]
    elif gam_prevu >= 10.0:
        interventions += [
            "Programme Supplémentaire Ciblé (PSC) — enfants MAM",
            "Distribution RUSF (Plumpy'Sup) + fortifiant Sprinkles",
            "Éducation nutritionnelle pour mères allaitantes",
        ]
    elif gam_prevu >= 5.0:
        interventions += [
            "Supplémentation préventive en micronutriments (Zinc, Fer-Folate)",
            "Promotion diversité alimentaire — jardins potagers familiaux",
            "Renforcement capacités agents santé communautaires (ASC)",
        ]
    else:
        interventions += [
            "Suivi de croissance de routine (pesées mensuelles)",
            "Supplémentation vitamine A bi-annuelle",
        ]

    return interventions