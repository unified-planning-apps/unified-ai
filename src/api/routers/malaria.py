"""
src/api/routers/malaria.py
===========================
Endpoints paludisme (Plasmodium falciparum — contexte Madagascar).

Couvre :
- Cas confirmés par région / district (temps réel + historique)
- Niveau de risque calculé (basé sur modèle ML + données météo)
- Carte de chaleur risque national
- Alertes épidémiologiques actives
- Facteurs de risque détaillés par région
- Comparaison inter-régionale
- Données saisonnières
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from unittest import result

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query, status
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.api.dependencies import (
    AuthUser,
    Cache,
    DbSession,
    MalariaModel,
    NationalOrAdmin,
    Pagination,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────────────────────────

class RisqueNiveau(str):
    FAIBLE      = "faible"
    MOYEN       = "moyen"
    ELEVE       = "élevé"
    TRES_ELEVE  = "très élevé"


class CasPaludisme(BaseModel):
    region_id: str
    region_name: str
    district: Optional[str] = None
    semaine_epidemio: int = Field(..., ge=1, le=53)
    annee: int
    date_rapport: date
    cas_confirmes: int = Field(..., ge=0)
    cas_confirmes_mixte: int = Field(..., ge=0)
    deces: int = Field(..., ge=0)
    hospitalisations: int = Field(..., ge=0)
    taux_incidence_pour_mille: float
    taux_positivite_tdr_pct: float = Field(..., ge=0, le=100,
        description="Test de Diagnostic Rapide positif (%)")
    population_a_risque: int
    source: str = Field(default="DHIS2")
    fiabilite_donnees: str = Field(
        default="confirmée",
        description="confirmée | estimée | incomplète"
    )


class FacteursRisque(BaseModel):
    temperature_moy_c: float
    precipitations_7j_mm: float
    precipitations_14j_mm: float
    precipitations_30j_mm: float
    humidite_moy_pct: float
    ndvi: Optional[float] = Field(None, description="Couverture végétale satellite")
    zones_humides_pct: Optional[float] = None
    altitude_m: float
    saison: str = Field(description="saison_pluies | saison_seche | transition")
    semaines_depuis_pics_pluies: int
    cas_historiques_4sem: int = Field(description="Cumul cas 4 semaines précédentes")
    endemicite: str


class PredictionRisqueMalaria(BaseModel):
    region_id: str
    region_name: str
    score_risque: float = Field(..., ge=0, le=1,
        description="Probabilité risque (0=aucun, 1=certain)")
    niveau_risque: str = Field(description="faible | moyen | élevé | très élevé")
    cas_prevus_7j: int
    cas_prevus_14j: int
    intervalle_confiance_bas: float
    intervalle_confiance_haut: float
    date_prediction: datetime
    horizon_jours: int
    facteurs_risque: FacteursRisque
    top_contributeurs: List[Dict] = Field(
        description="Top 5 features SHAP qui contribuent le plus au risque"
    )
    recommandations: List[str]
    fiabilite_modele: float = Field(..., ge=0, le=1,
        description="Confiance du modèle sur cette prédiction")


class AlerteEpidemiologique(BaseModel):
    alerte_id: str
    region_id: str
    region_name: str
    type_alerte: str = Field(
        description="seuil_depasse | tendance_hausse_rapide | anomalie_cluster | post_cyclone"
    )
    severite: str = Field(description="surveillance | alerte | urgence | crise")
    seuil_depasse: Optional[float] = None
    valeur_actuelle: float
    date_detection: datetime
    statut: str = Field(default="active", description="active | resolue | sous_surveillance")
    description: str
    actions_requises: List[str]
    responsable_notification: str


class ComparaisonRegionale(BaseModel):
    date_reference: date
    regions: List[Dict]
    region_plus_risquee: str
    region_moins_risquee: str
    moyenne_nationale_score: float
    tendance_nationale: str = Field(
        description="hausse | stable | baisse"
    )


class StatsSaisonnieres(BaseModel):
    region_id: str
    saison_courante: str
    semaine_dans_saison: int
    pic_historique_semaine: int = Field(
        description="Semaine épidémio du pic historique"
    )
    semaines_avant_pic_estime: int
    cas_cumules_saison: int
    cas_cumules_saison_precedente: int
    variation_pct: float
    tendance: str


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@router.get(
    "/cas/{region_id}",
    response_model=List[CasPaludisme],
    summary="Cas confirmés de paludisme par région",
    description=(
        "Retourne les cas confirmés, suspects, décès et hospitalisations pour une région. "
        "Source : DHIS2 Ministère de la Santé Madagascar."
    ),
)
async def get_cas_paludisme(
    region_id: str = Path(..., example="MDG-ATS",
        description="ID région Madagascar"),
    date_debut: date = Query(
        default_factory=lambda: date.today() - timedelta(days=90),
        description="Début de la période"
    ),
    date_fin: date = Query(
        default_factory=lambda: date.today(),
        description="Fin de la période"
    ),
    district: Optional[str] = Query(None,
        description="Filtrer par district (optionnel)"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
    pagination: Pagination = None,
):
    if (date_fin - date_debut).days > 730:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "PERIODE_TROP_LONGUE",
                "message": "La période ne peut pas dépasser 2 ans (730 jours).",
            },
        )

    cache_key = f"malaria:cas:{region_id}:{date_debut}:{date_fin}:{district or 'all'}"
    cached = await cache.get(cache_key)
    if cached:
        logger.debug("Cache HIT cas paludisme — {}", region_id)
        return json.loads(cached)

    from src.database.repositories.malaria_repo import MalariaRepository
    repo = MalariaRepository(db)
    cas = await repo.get_cas_by_region(
        region_id=region_id,
        date_debut=date_debut,
        date_fin=date_fin,
        district=district,
        limit=pagination.limit,
        offset=pagination.offset,
    )

    await cache.set(cache_key, json.dumps(cas, default=str), ttl=1800)
    return cas


@router.get(
    "/risque/{region_id}",
    response_model=PredictionRisqueMalaria,
    summary="Score de risque paludisme — prédiction ML",
    description=(
        "Calcule le score de risque paludisme pour une région sur l'horizon demandé. "
        "Utilise le modèle XGBoost avec features météo + historique épidémio. "
        "Inclut les valeurs SHAP pour l'explicabilité (requis UNICEF)."
    ),
)
async def get_risque_paludisme(
    region_id: str = Path(..., example="MDG-BOE",
        description="ID région"),
    horizon_jours: int = Query(
        default=14,
        ge=1,
        le=90,
        description="Horizon de prédiction en jours"
    ),
    user: AuthUser = None,
    model: MalariaModel = None,
    cache: Cache = None,
    db: DbSession = None,
):
    cache_key = f"malaria:risque:{region_id}:{horizon_jours}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Collecte des features pour le modèle
    try:
        from src.preprocessing.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer(db)
        features = await engineer.build_malaria_features(region_id)
    except Exception as exc:
        logger.error("Erreur construction features malaria {} : {}", region_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "FEATURES_ERREUR",
                "message": "Impossible de préparer les données pour la prédiction.",
            },
        ) from exc

    # Prédiction ML
    try:
        prediction = model.predict(features, horizon_days=horizon_jours)
    except Exception as exc:
        logger.error("Erreur prédiction malaria {} : {}", region_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "PREDICTION_ERREUR",
                "message": "Le modèle de prédiction a rencontré une erreur.",
            },
        ) from exc

    # Génération recommandations contextuelles
    recommandations = _generer_recommandations_malaria(
        score=prediction["score_risque"],
        features=features,
        region_id=region_id,
    )
    prediction["recommandations"] = recommandations

    # Cache 24h (prédictions ML coûteuses)
    await cache.set(cache_key, json.dumps(prediction, default=str), ttl=86400)
    return prediction


@router.get(
    "/carte-risque",
    summary="Carte de risque nationale — 22 régions",
    description=(
        "Retourne le score de risque pour toutes les régions de Madagascar. "
        "Données optimisées pour affichage choroplèthe. Cache 24h."
    ),
)
async def get_carte_risque_nationale(
    horizon_jours: int = Query(default=14, ge=1, le=90),
    user: AuthUser = None,
    model: MalariaModel = None,
    cache: Cache = None,
    db: DbSession = None,
    background_tasks: BackgroundTasks = None,
):
    cache_key = f"malaria:carte:{horizon_jours}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    import json as _json
    from pathlib import Path

    with Path("config/regions_metadata.json").open() as f:
        meta = _json.load(f)

    from src.preprocessing.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer(db)

    carte = []
    erreurs = []

    for region in meta["regions"]:
        rid = region["id"]
        try:
            features = await engineer.build_malaria_features(rid)
            pred = model.predict(features, horizon_days=horizon_jours)
            carte.append(
                {
                    "region_id": rid,
                    "region_name": region["name"],
                    "latitude": region["latitude"],
                    "longitude": region["longitude"],
                    "score_risque": pred["score_risque"],
                    "niveau_risque": pred["niveau_risque"],
                    "cas_prevus_14j": pred.get("cas_prevus_14j", 0),
                    "population": region["population_2023"],
                }
            )
        except Exception as exc:
            logger.warning("Erreur carte risque {} : {}", rid, exc)
            erreurs.append(rid)

    result = {
        "carte": carte,
        "horizon_jours": horizon_jours,
        "regions_ok": len(carte),
        "regions_erreur": erreurs,
        "genere_le": datetime.utcnow().isoformat(),
    }

    await cache.set(cache_key, json.dumps(result, default=str), ttl=86400)
    return result


@router.get(
    "/alertes",
    response_model=List[AlerteEpidemiologique],
    summary="Alertes épidémiologiques actives",
    description=(
        "Retourne les alertes épidémiologiques en cours : dépassements de seuils, "
        "tendances anormales, clusters détectés, alertes post-cyclone."
    ),
)
async def get_alertes_actives(
    region_id: Optional[str] = Query(None,
        description="Filtrer par région (toutes si absent)"),
    severite: Optional[str] = Query(
        None,
        description="surveillance | alerte | urgence | crise",
    ),
    statut: str = Query(default="active",
        description="active | resolue | sous_surveillance | all"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"malaria:alertes:{region_id or 'all'}:{severite or 'all'}:{statut}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.malaria_repo import MalariaRepository
    repo = MalariaRepository(db)
    alertes = await repo.get_alertes(
        region_id=region_id,
        severite=severite,
        statut=statut,
    )

    await cache.set(cache_key, json.dumps(alertes, default=str), ttl=900)  # 15min
    return alertes


@router.get(
    "/comparaison-regionale",
    response_model=ComparaisonRegionale,
    summary="Comparaison inter-régionale du risque paludisme",
    description="Classe les 22 régions par score de risque et donne la tendance nationale.",
)
async def get_comparaison_regionale(
    date_reference: Optional[date] = Query(
        default_factory=date.today,
        description="Date de référence pour la comparaison"
    ),
    user: AuthUser = None,
    model: MalariaModel = None,
    cache: Cache = None,
    db: DbSession = None,
):
    cache_key = f"malaria:comparaison:{date_reference}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    import json as _json
    from pathlib import Path

    with Path("config/regions_metadata.json").open() as f:
        meta = _json.load(f)

    from src.preprocessing.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer(db)

    regions_data = []
    for region in meta["regions"]:
        rid = region["id"]
        try:
            features = await engineer.build_malaria_features(rid)
            pred = model.predict(features, horizon_days=7)
            regions_data.append(
                {
                    "region_id": rid,
                    "region_name": region["name"],
                    "score_risque": pred["score_risque"],
                    "niveau_risque": pred["niveau_risque"],
                    "rang": 0,  # calculé après tri
                }
            )
        except Exception:
            pass

    # Tri par score décroissant + rang
    regions_data.sort(key=lambda x: x["score_risque"], reverse=True)
    for i, r in enumerate(regions_data):
        r["rang"] = i + 1

    scores = [r["score_risque"] for r in regions_data]
    moyenne = sum(scores) / len(scores) if scores else 0.0

    result = {
        "date_reference": str(date_reference),
        "regions": regions_data,
        "region_plus_risquee": regions_data[0]["region_id"] if regions_data else "",
        "region_moins_risquee": regions_data[-1]["region_id"] if regions_data else "",
        "moyenne_nationale_score": round(moyenne, 4),
        "tendance_nationale": _calculer_tendance_nationale(db, date_reference),
    }

    await cache.set(cache_key, json.dumps(result, default=str), ttl=43200)
    return result


@router.get(
    "/saisonnalite/{region_id}",
    response_model=StatsSaisonnieres,
    summary="Statistiques saisonnières paludisme",
    description=(
        "Analyse la saisonnalité du paludisme : position dans la saison, "
        "semaines avant le pic estimé, comparaison avec saison précédente."
    ),
)
async def get_stats_saisonnieres(
    region_id: str = Path(..., example="MDG-SAV"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"malaria:saisonnalite:{region_id}:{date.today().isocalendar()[1]}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.malaria_repo import MalariaRepository
    repo = MalariaRepository(db)
    stats = await repo.get_seasonal_stats(region_id)

    await cache.set(cache_key, json.dumps(stats, default=str), ttl=86400)
    return stats


@router.get(
    "/facteurs-risque/{region_id}",
    response_model=FacteursRisque,
    summary="Facteurs de risque détaillés d'une région",
)
async def get_facteurs_risque(
    region_id: str = Path(..., example="MDG-VAT"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    try:
        # ─────────────────────────────
        # 1. CACHE (versionné proprement)
        # ─────────────────────────────
        cache_key = f"malaria:facteurs:v1:{region_id}:{date.today()}"

        cached = await cache.get(cache_key)
        if cached:
            logger.info(f"[CACHE HIT] facteurs-risque {region_id}")
            return json.loads(cached)

        # ─────────────────────────────
        # 2. FEATURE ENGINEERING
        # ─────────────────────────────
        from src.preprocessing.feature_engineering import FeatureEngineer
        from src.utils.constants import get_saison_courante

        engineer = FeatureEngineer(db)
        features = await engineer.build_malaria_features(region_id)

        # ─────────────────────────────
        # 3. DERIVED FEATURES
        # ─────────────────────────────
        mois = date.today().month
        saison_obj = get_saison_courante(mois)

        # endemicité
        enc_map = {0: "low", 1: "medium", 2: "high", 3: "very_high"}
        endemicite_encoded = int(features.get("endemicite_encoded", 1))
        endemicite_str = enc_map.get(endemicite_encoded, "medium")

        # saison pluie approximation
        saison_encoded = int(features.get("saison_encoded", 0))
        if saison_encoded == 2:
            semaines_depuis_pic = 4
        elif saison_encoded == 1:
            semaines_depuis_pic = 10
        else:
            semaines_depuis_pic = 20

        # cas historiques 4 semaines
        cas_historiques = int(
            features.get("cas_lag_1sem", 0)
            + features.get("cas_lag_2sem", 0)
            + features.get("cas_lag_3sem", 0)
            + features.get("cas_lag_4sem", 0)
        )

        # ─────────────────────────────
        # 4. RESPONSE FINAL (IMPORTANT)
        # ─────────────────────────────
        result = {
            "temperature_moy_c": features.get("temperature_moy_c", 0.0),
            "precipitations_7j_mm": features.get("precipitations_7j_mm", 0.0),
            "precipitations_14j_mm": features.get("precipitations_14j_mm", 0.0),
            "precipitations_30j_mm": features.get("precipitations_30j_mm", 0.0),
            "humidite_moy_pct": features.get("humidite_moy_pct", 0.0),
            "ndvi": features.get("ndvi"),
            "zones_humides_pct": features.get("zones_humides_pct"),
            "altitude_m": features.get("altitude_m", 0.0),

            # dérivés métier
            "saison": saison_obj.value,
            "semaines_depuis_pics_pluies": semaines_depuis_pic,
            "cas_historiques_4sem": cas_historiques,
            "endemicite": endemicite_str,
        }

        # ─────────────────────────────
        # 5. CACHE SAVE (IMPORTANT)
        # ─────────────────────────────
        await cache.set(
            cache_key,
            json.dumps(result, default=str),
            ttl=3600
        )

        logger.info(f"[OK] facteurs-risque {region_id}")

        return result

    except Exception as e:
        logger.exception(f"Erreur get_facteurs_risque {region_id}")
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

@router.get(
    "/tendance/{region_id}",
    summary="Tendance hebdomadaire des cas — courbe temporelle",
    description=(
        "Retourne la courbe des cas confirmés semaine par semaine "
        "sur les N dernières semaines, avec moyenne mobile."
    ),
)
async def get_tendance_hebdo(
    region_id: str = Path(..., example="MDG-ANA"),
    semaines: int = Query(default=26, ge=4, le=104,
        description="Nombre de semaines à retourner"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"malaria:tendance:{region_id}:{semaines}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    date_fin = date.today()
    date_debut = date_fin - timedelta(weeks=semaines)

    from src.database.repositories.malaria_repo import MalariaRepository
    repo = MalariaRepository(db)
    data = await repo.get_weekly_trend(region_id, date_debut, date_fin)

    # Calcul moyenne mobile 4 semaines
    cas_list = [d.get("cas_confirmes", 0) for d in data]
    for i, point in enumerate(data):
        window = cas_list[max(0, i - 3): i + 1]
        point["moyenne_mobile_4sem"] = round(sum(window) / len(window), 1)

    result = {
        "region_id": region_id,
        "semaines": semaines,
        "data": data,
        "total_cas_periode": sum(cas_list),
        "semaine_pic": max(data, key=lambda x: x.get("cas_confirmes", 0), default={}).get(
            "semaine_epidemio"
        ),
    }

    await cache.set(cache_key, json.dumps(result, default=str), ttl=3600)
    return result


@router.post(
    "/alertes/{alerte_id}/acquitter",
    summary="Acquitter une alerte épidémiologique",
    description=(
        "Marque une alerte comme prise en charge. "
        "Réservé aux rôles national et admin."
    ),
    dependencies=[NationalOrAdmin],
)
async def acquitter_alerte(
    alerte_id: str = Path(..., description="ID de l'alerte"),
    commentaire: Optional[str] = Query(None,
        description="Commentaire de l'agent qui acquitte"),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    from src.database.repositories.malaria_repo import MalariaRepository
    repo = MalariaRepository(db)

    updated = await repo.acquitter_alerte(
        alerte_id=alerte_id,
        user_id=user.user_id,
        commentaire=commentaire,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "ALERTE_INTROUVABLE",
                "message": f"Alerte {alerte_id} introuvable.",
            },
        )

    # Invalidation cache alertes
    await cache.invalidate_pattern("malaria:alertes:*")

    return {
        "statut": "succès",
        "message": f"Alerte {alerte_id} acquittée par {user.username}.",
        "alerte_id": alerte_id,
        "acquittee_par": user.username,
        "horodatage": datetime.utcnow().isoformat(),
    }


@router.get(
    "/statistiques/national",
    summary="Tableau de bord national paludisme",
    description=(
        "KPIs nationaux : total cas, taux de positivité moyen, "
        "régions en alerte, évolution vs semaine précédente."
    ),
    dependencies=[NationalOrAdmin],
)
async def get_statistiques_nationales(
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"malaria:stats:national:{date.today().isocalendar()[1]}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.malaria_repo import MalariaRepository
    repo = MalariaRepository(db)
    stats = await repo.get_national_stats()

    await cache.set(cache_key, json.dumps(stats, default=str), ttl=3600)
    return stats


# ─────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────

def _generer_recommandations_malaria(
    score: float,
    features: dict,
    region_id: str,
) -> List[str]:
    """
    Génère des recommandations textuelles contextualisées
    selon le score de risque et les features dominantes.
    """
    recs = []

    if score >= 0.75:
        recs += [
            "Déclencher le plan de riposte épidémique immédiatement.",
            "Renforcer le stock de médicaments antipaludéens (ACT) dans les CSB.",
            "Mobiliser les agents de santé communautaires pour le dépistage actif.",
            "Renforcer la distribution de moustiquaires imprégnées (MILDA).",
            "Activer la surveillance épidémiologique hebdomadaire renforcée.",
        ]
    elif score >= 0.50:
        recs += [
            "Renforcer la surveillance épidémiologique dans les districts à risque.",
            "Prépositionner les stocks de TDR et d'ACT dans les formations sanitaires.",
            "Campagnes de sensibilisation sur l'utilisation des moustiquaires.",
            "Organiser des séances de pulvérisation intradomiciliaire (PID).",
        ]
    elif score >= 0.25:
        recs += [
            "Surveillance de routine — pas d'action d'urgence requise.",
            "Maintenir les activités de prévention saisonnière du paludisme (PSP).",
            "Vérifier la disponibilité des stocks de médicaments dans les CSB.",
        ]
    else:
        recs += [
            "Risque faible — poursuivre les activités de prévention habituelles.",
            "Mettre à jour les registres épidémiologiques de routine.",
        ]

    # Recommandations spécifiques aux features
    pluies_30j = features.get("precipitations_30j_mm", 0)
    if pluies_30j > 200:
        recs.append(
            f"Précipitations élevées ({pluies_30j:.0f}mm/30j) — "
            "surveiller les gîtes larvaires dans les zones inondées."
        )

    ndvi = features.get("ndvi")
    if ndvi and ndvi > 0.6:
        recs.append(
            "Végétation dense (NDVI élevé) — conditions favorables aux moustiques. "
            "Intensifier la lutte antivectorielle."
        )

    temp = features.get("temperature_moy_c", 0)
    if 20 <= temp <= 30:
        recs.append(
            f"Température optimale ({temp:.1f}°C) pour le développement du parasite. "
            "Vigilance accrue recommandée."
        )

    return recs


def _calculer_tendance_nationale(db, date_ref: date) -> str:
    """Calcule la tendance nationale (hausse/stable/baisse) — simplifié."""
    # Dans une implémentation complète, interroger la DB
    # Ici on retourne une valeur par défaut
    return "stable"