"""
src/api/routers/reports.py
===========================
Endpoints de génération et gestion des rapports UNICEF.

Couvre :
- Rapport hebdomadaire paludisme (par région ou national)
- Rapport hebdomadaire nutrition
- Rapport combiné paludisme + nutrition
- Rapport d'urgence (déclenchement immédiat)
- Historique et téléchargement des rapports
- Planification de rapports automatiques
- Export données brutes (CSV, JSON)
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Path,
    Query,
    Response,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from src.api.dependencies import (
    AdminOnly,
    AuthUser,
    Cache,
    DbSession,
    NationalOrAdmin,
    Pagination,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────
# Enums et Schémas Pydantic
# ─────────────────────────────────────────────────────────────────

class TypeRapport(str, Enum):
    PALUDISME_HEBDO      = "paludisme_hebdomadaire"
    NUTRITION_HEBDO      = "nutrition_hebdomadaire"
    COMBINE_HEBDO        = "combine_hebdomadaire"
    URGENCE              = "urgence"
    MENSUEL              = "mensuel"
    ALERTE_EPIDEMIQUE    = "alerte_epidemique"


class FormatRapport(str, Enum):
    PDF  = "pdf"
    HTML = "html"
    JSON = "json"


class StatutRapport(str, Enum):
    EN_ATTENTE   = "en_attente"
    EN_COURS     = "en_cours"
    TERMINE      = "termine"
    ERREUR       = "erreur"


class LangueRapport(str, Enum):
    FRANCAIS = "fr"
    MALGACHE = "mg"


class DemandeRapport(BaseModel):
    type_rapport: TypeRapport
    format: FormatRapport = FormatRapport.PDF
    langue: LangueRapport = LangueRapport.FRANCAIS

    # Portée géographique
    region_id: Optional[str] = Field(
        None,
        description="Si absent → rapport national (22 régions)"
    )

    # Période
    date_debut: Optional[date] = Field(
        default_factory=lambda: date.today() - timedelta(days=7),
        description="Début période d'analyse"
    )
    date_fin: Optional[date] = Field(
        default_factory=date.today,
        description="Fin période d'analyse"
    )

    # Options avancées
    inclure_cartes: bool = Field(
        default=True,
        description="Inclure les cartes choroplèthes"
    )
    inclure_shap: bool = Field(
        default=True,
        description="Inclure les graphiques SHAP (explicabilité)"
    )
    inclure_recettes: bool = Field(
        default=True,
        description="Inclure les recettes nutritionnelles recommandées"
    )
    inclure_stocks: bool = Field(
        default=False,
        description="Inclure l'état des stocks humanitaires"
    )
    destinataires_email: Optional[List[str]] = Field(
        None,
        description="Envoyer par email après génération"
    )


class StatutGenerationRapport(BaseModel):
    rapport_id: str
    type_rapport: TypeRapport
    statut: StatutRapport
    region_id: Optional[str]
    demande_par: str
    demande_le: datetime
    termine_le: Optional[datetime] = None
    duree_generation_sec: Optional[float] = None
    url_telechargement: Optional[str] = None
    taille_fichier_ko: Optional[float] = None
    message_erreur: Optional[str] = None


class MetadataRapport(BaseModel):
    rapport_id: str
    type_rapport: TypeRapport
    format: FormatRapport
    langue: LangueRapport
    region_id: Optional[str]
    region_name: Optional[str]
    date_debut: date
    date_fin: date
    genere_le: datetime
    genere_par: str
    taille_ko: float
    nb_pages: Optional[int]
    url_telechargement: str
    valide_jusqu_au: datetime = Field(
        description="Date d'expiration du lien de téléchargement"
    )


class PlanificationRapport(BaseModel):
    planification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type_rapport: TypeRapport
    format: FormatRapport = FormatRapport.PDF
    langue: LangueRapport = LangueRapport.FRANCAIS
    region_id: Optional[str] = None
    frequence: str = Field(
        description="hebdomadaire | mensuel | bimensuel",
        default="hebdomadaire"
    )
    jour_generation: int = Field(
        default=1,
        ge=1, le=7,
        description="Jour de la semaine (1=Lundi, 7=Dimanche)"
    )
    heure_generation: str = Field(
        default="06:00",
        description="Heure de génération (HH:MM — fuseau Antananarivo)"
    )
    destinataires_email: List[str]
    actif: bool = True
    creee_par: Optional[str] = None
    creee_le: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/generer",
    response_model=StatutGenerationRapport,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Générer un rapport UNICEF",
    description=(
        "Lance la génération asynchrone d'un rapport. "
        "Retourne immédiatement un `rapport_id` pour suivre l'avancement. "
        "La génération s'effectue en arrière-plan (Celery). "
        "Types disponibles : paludisme_hebdomadaire, nutrition_hebdomadaire, "
        "combine_hebdomadaire, urgence, mensuel."
    ),
)
async def generer_rapport(
    demande: DemandeRapport,
    background_tasks: BackgroundTasks,
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    rapport_id = str(uuid.uuid4())

    logger.info(
        "Génération rapport demandée — type={} region={} format={} user={}",
        demande.type_rapport, demande.region_id or "national",
        demande.format, user.username,
    )

    # Enregistrement du statut initial en DB
    statut_initial = {
        "rapport_id": rapport_id,
        "type_rapport": demande.type_rapport,
        "statut": StatutRapport.EN_ATTENTE,
        "region_id": demande.region_id,
        "demande_par": user.username,
        "demande_le": datetime.utcnow().isoformat(),
    }

    from src.database.repositories.weather_repo import WeatherRepository
    # Dans une vraie implémentation, on a un ReportRepository
    # Ici on sauvegarde le statut dans Redis pour la démo
    await cache.set(
        f"rapports:statut:{rapport_id}",
        json.dumps(statut_initial, default=str),
        ttl=86400 * 7,  # 7 jours
    )

    # Lance la génération en arrière-plan
    background_tasks.add_task(
        _generer_rapport_async,
        rapport_id=rapport_id,
        demande=demande,
        user_name=user.username,
        cache=cache,
    )

    return StatutGenerationRapport(
        rapport_id=rapport_id,
        type_rapport=demande.type_rapport,
        statut=StatutRapport.EN_ATTENTE,
        region_id=demande.region_id,
        demande_par=user.username,
        demande_le=datetime.utcnow(),
    )


@router.get(
    "/statut/{rapport_id}",
    response_model=StatutGenerationRapport,
    summary="Suivre l'avancement d'une génération",
    description="Polling sur le statut de génération d'un rapport (en_attente/en_cours/termine/erreur).",
)
async def get_statut_rapport(
    rapport_id: str = Path(..., description="ID du rapport"),
    user: AuthUser = None,
    cache: Cache = None,
):
    cached = await cache.get(f"rapports:statut:{rapport_id}")
    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "RAPPORT_INTROUVABLE",
                "message": f"Rapport {rapport_id} introuvable ou expiré.",
            },
        )
    return json.loads(cached)


@router.get(
    "/telecharger/{rapport_id}",
    summary="Télécharger un rapport généré",
    description=(
        "Télécharge le fichier PDF/HTML d'un rapport terminé. "
        "Le lien est valable 7 jours après génération."
    ),
    response_class=FileResponse,
)
async def telecharger_rapport(
    rapport_id: str = Path(...),
    user: AuthUser = None,
    cache: Cache = None,
    db: DbSession = None,
):
    # Vérification statut
    cached = await cache.get(f"rapports:statut:{rapport_id}")
    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RAPPORT_INTROUVABLE",
                    "message": f"Rapport {rapport_id} introuvable."},
        )

    statut = json.loads(cached)
    if statut.get("statut") != StatutRapport.TERMINE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "RAPPORT_NON_PRET",
                "message": (
                    f"Le rapport n'est pas encore prêt "
                    f"(statut actuel : {statut.get('statut')})."
                ),
            },
        )

    fichier_path = statut.get("chemin_fichier")
    if not fichier_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "FICHIER_MANQUANT",
                    "message": "Fichier rapport introuvable sur le serveur."},
        )

    import os
    if not os.path.exists(fichier_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FICHIER_EXPIRE",
                    "message": "Le fichier rapport a expiré ou a été supprimé."},
        )

    nom_fichier = os.path.basename(fichier_path)
    format_rapport = statut.get("format", "pdf")
    media_types = {
        "pdf":  "application/pdf",
        "html": "text/html",
        "json": "application/json",
    }

    return FileResponse(
        path=fichier_path,
        media_type=media_types.get(format_rapport, "application/octet-stream"),
        filename=nom_fichier,
        headers={"Content-Disposition": f"attachment; filename={nom_fichier}"},
    )


@router.get(
    "/historique",
    response_model=List[MetadataRapport],
    summary="Historique des rapports générés",
    description=(
        "Liste les rapports générés avec leurs métadonnées et liens de téléchargement. "
        "Filtrables par type, région, période."
    ),
)
async def get_historique_rapports(
    type_rapport: Optional[TypeRapport] = Query(None),
    region_id: Optional[str] = Query(None),
    date_debut: Optional[date] = Query(None),
    date_fin: Optional[date] = Query(
        default_factory=date.today
    ),
    user: AuthUser = None,
    db: DbSession = None,
    pagination: Pagination = None,
):
    from src.database.repositories.weather_repo import WeatherRepository
    # Dans une implémentation complète → ReportRepository
    # Retourne une liste mock pour la démonstration de l'API
    return []


@router.get(
    "/hebdomadaire/{region_id}",
    summary="Rapport hebdomadaire rapide (aperçu JSON)",
    description=(
        "Retourne le contenu d'un rapport hebdomadaire au format JSON "
        "sans génération PDF. Utile pour affichage dashboard en temps réel."
    ),
)
async def get_rapport_hebdo_json(
    region_id: str = Path(..., example="MDG-ANA"),
    semaine: Optional[int] = Query(
        None,
        description="Numéro de semaine ISO (défaut : semaine courante)"
    ),
    annee: Optional[int] = Query(
        None,
        description="Année (défaut : année courante)"
    ),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    today = date.today()
    iso = today.isocalendar()
    semaine = semaine or iso[1]
    annee = annee or iso[0]

    cache_key = f"rapports:hebdo_json:{region_id}:{annee}:S{semaine}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Calcul dates de la semaine
    date_debut = date.fromisocalendar(annee, semaine, 1)
    date_fin = date.fromisocalendar(annee, semaine, 7)

    from src.database.repositories.malaria_repo import MalariaRepository
    from src.database.repositories.nutrition_repo import NutritionRepository

    mal_repo = MalariaRepository(db)
    nut_repo = NutritionRepository(db)

    # Collecte données
    cas_semaine = await mal_repo.get_cas_by_region(
        region_id=region_id,
        date_debut=date_debut,
        date_fin=date_fin,
    )
    statut_nut = await nut_repo.get_statut_actuel(region_id)
    alertes_mal = await mal_repo.get_alertes(region_id=region_id, statut="active")
    alertes_nut = await nut_repo.get_alertes(region_id=region_id, statut="active")

    rapport = {
        "region_id": region_id,
        "semaine": f"S{semaine:02d}-{annee}",
        "date_debut": str(date_debut),
        "date_fin": str(date_fin),
        "genere_le": datetime.utcnow().isoformat(),
        "paludisme": {
            "cas_confirmes_semaine": sum(
                c.get("cas_confirmes", 0) for c in cas_semaine
            ) if cas_semaine else 0,
            "deces_semaine": sum(
                c.get("deces", 0) for c in cas_semaine
            ) if cas_semaine else 0,
            "taux_positivite_tdr": (
                cas_semaine[-1].get("taux_positivite_tdr_pct")
                if cas_semaine else None
            ),
            "alertes_actives": len(alertes_mal) if alertes_mal else 0,
        },
        "nutrition": {
            "gam_pct": statut_nut.get("gam_pct") if statut_nut else None,
            "sam_pct": statut_nut.get("sam_pct") if statut_nut else None,
            "classification_who": (
                statut_nut.get("classification_who") if statut_nut else "non disponible"
            ),
            "alertes_actives": len(alertes_nut) if alertes_nut else 0,
        },
    }

    await cache.set(cache_key, json.dumps(rapport, default=str), ttl=3600)
    return rapport


@router.post(
    "/urgence",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Rapport d'urgence — génération prioritaire",
    description=(
        "Génère un rapport d'urgence avec priorité maximale. "
        "Destiné aux situations de crise (épidémie, cyclone, famine). "
        "Envoyé automatiquement aux responsables UNICEF. "
        "Accès : national + admin."
    ),
    dependencies=[NationalOrAdmin],
)
async def generer_rapport_urgence(
    region_id: str = Query(..., description="Région en crise"),
    type_crise: str = Query(
        ...,
        description="epidemie_paludisme | crise_nutrition | cyclone | secheresse | autre",
    ),
    description_crise: str = Query(
        ...,
        description="Description courte de la situation d'urgence",
        max_length=500,
    ),
    background_tasks: BackgroundTasks = None,
    user: AuthUser = None,
    cache: Cache = None,
    db: DbSession = None,
):
    rapport_id = f"URG-{str(uuid.uuid4())[:8].upper()}"

    logger.warning(
        "🚨 RAPPORT URGENCE déclenché — region={} type={} par={}",
        region_id, type_crise, user.username,
    )

    statut_initial = {
        "rapport_id": rapport_id,
        "type_rapport": TypeRapport.URGENCE,
        "statut": StatutRapport.EN_ATTENTE,
        "region_id": region_id,
        "type_crise": type_crise,
        "description_crise": description_crise,
        "demande_par": user.username,
        "demande_le": datetime.utcnow().isoformat(),
        "priorite": "URGENT",
    }

    await cache.set(
        f"rapports:statut:{rapport_id}",
        json.dumps(statut_initial, default=str),
        ttl=86400 * 30,
    )

    demande_urgence = DemandeRapport(
        type_rapport=TypeRapport.URGENCE,
        format=FormatRapport.PDF,
        langue=LangueRapport.FRANCAIS,
        region_id=region_id,
        inclure_cartes=True,
        inclure_shap=False,
        inclure_recettes=False,
        inclure_stocks=True,
    )

    background_tasks.add_task(
        _generer_rapport_async,
        rapport_id=rapport_id,
        demande=demande_urgence,
        user_name=user.username,
        cache=cache,
        urgence=True,
        type_crise=type_crise,
        description_crise=description_crise,
    )

    return {
        "rapport_id": rapport_id,
        "statut": "démarré",
        "priorite": "URGENT",
        "message": (
            f"Rapport d'urgence {rapport_id} en cours de génération. "
            "Il sera transmis aux responsables UNICEF dès finalisation."
        ),
        "region_id": region_id,
        "type_crise": type_crise,
        "declenche_par": user.username,
        "horodatage": datetime.utcnow().isoformat(),
    }


@router.get(
    "/planifications",
    response_model=List[PlanificationRapport],
    summary="Lister les planifications de rapports automatiques",
    dependencies=[NationalOrAdmin],
)
async def get_planifications(
    actif_seulement: bool = Query(default=True),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"rapports:planifications:{actif_seulement}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Dans une implémentation complète → ReportScheduleRepository
    return []


@router.post(
    "/planifications",
    response_model=PlanificationRapport,
    status_code=status.HTTP_201_CREATED,
    summary="Créer une planification de rapport automatique",
    description=(
        "Configure la génération automatique de rapports récurrents. "
        "Ex : rapport paludisme hebdomadaire chaque lundi 06h00 → 5 emails. "
        "Exécuté par Celery Beat."
    ),
    dependencies=[NationalOrAdmin],
)
async def creer_planification(
    planification: PlanificationRapport,
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    planification.creee_par = user.username
    planification.creee_le = datetime.utcnow()

    logger.info(
        "Planification créée — type={} freq={} par={}",
        planification.type_rapport, planification.frequence, user.username,
    )

    # Sauvegarde en DB + programmation Celery Beat
    # Dans une implémentation complète → ScheduleRepository.save()
    await cache.invalidate_pattern("rapports:planifications:*")

    return planification


@router.delete(
    "/planifications/{planification_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Supprimer une planification",
    dependencies=[NationalOrAdmin],
)
async def supprimer_planification(
    planification_id: str = Path(...),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    logger.info(
        "Planification supprimée — id={} par={}", planification_id, user.username
    )
    await cache.invalidate_pattern("rapports:planifications:*")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/export/{region_id}",
    summary="Export données brutes CSV / JSON",
    description=(
        "Exporte les données brutes d'une région (paludisme + nutrition + météo) "
        "au format CSV ou JSON pour analyse externe."
    ),
)
async def export_donnees(
    region_id: str = Path(...),
    format_export: str = Query(
        default="csv",
        description="csv | json"
    ),
    date_debut: date = Query(
        default_factory=lambda: date.today() - timedelta(days=90)
    ),
    date_fin: date = Query(default_factory=date.today),
    inclure_meteo: bool = Query(default=True),
    inclure_paludisme: bool = Query(default=True),
    inclure_nutrition: bool = Query(default=True),
    user: AuthUser = None,
    db: DbSession = None,
):
    if format_export not in ("csv", "json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "FORMAT_INVALIDE",
                    "message": "Format : csv | json"},
        )

    from src.database.repositories.malaria_repo import MalariaRepository
    from src.database.repositories.nutrition_repo import NutritionRepository
    from src.database.repositories.weather_repo import WeatherRepository

    donnees: Dict = {}

    if inclure_meteo:
        w_repo = WeatherRepository(db)
        donnees["meteo"] = await w_repo.get_history(
            region_id=region_id,
            date_debut=date_debut,
            date_fin=date_fin,
        )

    if inclure_paludisme:
        m_repo = MalariaRepository(db)
        donnees["paludisme"] = await m_repo.get_cas_by_region(
            region_id=region_id,
            date_debut=date_debut,
            date_fin=date_fin,
        )

    if inclure_nutrition:
        n_repo = NutritionRepository(db)
        donnees["nutrition"] = await n_repo.get_gam_trend(
            region_id=region_id,
            date_debut=date_debut,
            date_fin=date_fin,
        )

    nom_fichier = f"export_{region_id}_{date_debut}_{date_fin}"

    if format_export == "json":
        content = json.dumps(donnees, default=str, ensure_ascii=False, indent=2)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={nom_fichier}.json"
            },
        )
    else:
        # CSV : aplatissement des données pour format tabulaire
        csv_content = _donnees_vers_csv(donnees)
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename={nom_fichier}.csv"
            },
        )


@router.get(
    "/statistiques",
    summary="Statistiques de génération de rapports",
    description=(
        "Métriques d'utilisation : nombre de rapports générés, "
        "temps moyen de génération, formats les plus utilisés."
    ),
    dependencies=[NationalOrAdmin],
)
async def get_stats_rapports(
    periode_jours: int = Query(default=30, ge=1, le=365),
    user: AuthUser = None,
    cache: Cache = None,
    db: DbSession = None,
):
    cache_key = f"rapports:stats:{periode_jours}:{date.today()}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Dans une implémentation complète → ReportRepository.get_stats()
    stats = {
        "periode_jours": periode_jours,
        "total_rapports": 0,
        "par_type": {},
        "par_format": {"pdf": 0, "html": 0, "json": 0},
        "temps_moyen_generation_sec": 0,
        "taux_succes_pct": 100.0,
        "rapports_urgence": 0,
    }

    await cache.set(cache_key, json.dumps(stats, default=str), ttl=3600)
    return stats


# ─────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────

async def _generer_rapport_async(
    rapport_id: str,
    demande: DemandeRapport,
    user_name: str,
    cache: Cache,
    urgence: bool = False,
    type_crise: Optional[str] = None,
    description_crise: Optional[str] = None,
):
    """
    Tâche de génération asynchrone du rapport.
    Exécutée en BackgroundTask (ou déléguée à Celery en production).
    """
    import time

    start_time = time.time()
    logger.info("Début génération rapport {} (type={})", rapport_id, demande.type_rapport)

    try:
        # Mise à jour statut → EN_COURS
        await _update_statut(cache, rapport_id, StatutRapport.EN_COURS)

        from src.reports.generator import ReportGenerator
        generator = ReportGenerator()

        chemin_fichier = await generator.generate(
            rapport_id=rapport_id,
            type_rapport=demande.type_rapport,
            format_rapport=demande.format,
            langue=demande.langue,
            region_id=demande.region_id,
            date_debut=demande.date_debut,
            date_fin=demande.date_fin,
            options={
                "inclure_cartes": demande.inclure_cartes,
                "inclure_shap": demande.inclure_shap,
                "inclure_recettes": demande.inclure_recettes,
                "inclure_stocks": demande.inclure_stocks,
                "urgence": urgence,
                "type_crise": type_crise,
                "description_crise": description_crise,
            },
        )

        duree = time.time() - start_time
        taille_ko = _get_file_size_ko(chemin_fichier)

        # Mise à jour statut → TERMINÉ
        statut_final = {
            "statut": StatutRapport.TERMINE,
            "termine_le": datetime.utcnow().isoformat(),
            "duree_generation_sec": round(duree, 2),
            "chemin_fichier": str(chemin_fichier),
            "url_telechargement": f"/api/v1/rapports/telecharger/{rapport_id}",
            "taille_fichier_ko": taille_ko,
            "format": demande.format,
        }
        await _update_statut(cache, rapport_id, StatutRapport.TERMINE, statut_final)

        logger.info(
            "Rapport {} terminé en {:.1f}s — {:.0f} Ko",
            rapport_id, duree, taille_ko,
        )

        # Envoi email si destinataires configurés
        if demande.destinataires_email:
            await _envoyer_rapport_email(
                rapport_id=rapport_id,
                chemin_fichier=chemin_fichier,
                destinataires=demande.destinataires_email,
                type_rapport=demande.type_rapport,
                region_id=demande.region_id,
            )

    except Exception as exc:
        duree = time.time() - start_time
        logger.error("Erreur génération rapport {} : {}", rapport_id, exc)

        await _update_statut(
            cache,
            rapport_id,
            StatutRapport.ERREUR,
            {
                "statut": StatutRapport.ERREUR,
                "termine_le": datetime.utcnow().isoformat(),
                "duree_generation_sec": round(duree, 2),
                "message_erreur": str(exc),
            },
        )


async def _update_statut(
    cache: Cache,
    rapport_id: str,
    nouveau_statut: StatutRapport,
    extra: Optional[Dict] = None,
):
    """Met à jour le statut d'un rapport dans Redis."""
    existing_raw = await cache.get(f"rapports:statut:{rapport_id}")
    existing = json.loads(existing_raw) if existing_raw else {}
    existing["statut"] = nouveau_statut
    if extra:
        existing.update(extra)
    await cache.set(
        f"rapports:statut:{rapport_id}",
        json.dumps(existing, default=str),
        ttl=86400 * 7,
    )


def _get_file_size_ko(chemin: str) -> float:
    import os
    try:
        return os.path.getsize(chemin) / 1024
    except Exception:
        return 0.0


def _donnees_vers_csv(donnees: Dict) -> str:
    """Convertit les données exportées en format CSV simple."""
    import csv
    import io

    output = io.StringIO()

    for section, records in donnees.items():
        if not records:
            continue
        output.write(f"\n# === {section.upper()} ===\n")
        if isinstance(records, list) and len(records) > 0:
            writer = csv.DictWriter(
                output,
                fieldnames=list(records[0].keys()),
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in records:
                writer.writerow(
                    {k: str(v) if v is not None else "" for k, v in row.items()}
                )

    return output.getvalue()


async def _envoyer_rapport_email(
    rapport_id: str,
    chemin_fichier: str,
    destinataires: List[str],
    type_rapport: TypeRapport,
    region_id: Optional[str],
):
    """Envoie le rapport PDF par email aux destinataires configurés."""
    logger.info(
        "Envoi rapport {} à {} destinataires",
        rapport_id, len(destinataires),
    )
    # Dans une implémentation complète → service SMTP / SendGrid
    # Ici : log uniquement
    for dest in destinataires:
        logger.info("  → {}", dest)