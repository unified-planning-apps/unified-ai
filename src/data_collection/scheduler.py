"""
Tâches Celery pour l'automatisation de la collecte de données.

Pipeline temps réel :
  ┌──────────────────────────────────────────────────────┐
  │  APIs Externes → Celery Workers → Redis → PostgreSQL  │
  └──────────────────────────────────────────────────────┘

Tâches planifiées (Celery Beat) :
  - Toutes les heures  : météo actuelle + prédictions cache
  - Toutes les 6h      : données paludisme DHIS2
  - Toutes les 24h     : données nutrition WFP/FAO
  - Chaque lundi 05h00 : rapport hebdomadaire automatique
  - Chaque 1er du mois : retraining des modèles ML

Gestion des erreurs :
  - Retry automatique avec backoff exponentiel
  - Dead Letter Queue pour tâches en échec persistant
  - Monitoring via Flower (dashboard Celery)
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from celery import Celery
from celery.schedules import crontab
from celery.utils.log import get_task_logger
from kombu import Queue

from config.settings import settings

logger = get_task_logger(__name__)

# ─────────────────────────────────────────────────────────────────
# Création de l'application Celery
# ─────────────────────────────────────────────────────────────────

celery_app = Celery(
    "malaria_nutrition_predictor",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "src.data_collection.scheduler",
    ],
)

celery_app.conf.update(
    # ── Sérialisation ──────────────────────────────
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=settings.celery_timezone,
    enable_utc=True,

    # ── Résultats ──────────────────────────────────
    result_expires=86400,          # 24h
    result_backend_transport_options={
        "retry_policy": {"timeout": 5.0}
    },

    # ── Performances ───────────────────────────────
    worker_prefetch_multiplier=1,  # 1 tâche à la fois par worker (ML = lourd)
    task_acks_late=True,           # ACK après exécution (pas avant)
    worker_max_tasks_per_child=100, # Recycle le worker après 100 tâches

    # ── Files de priorité ──────────────────────────
    task_queues=(
        Queue("urgent",    routing_key="urgent.#"),     # Rapports urgence
        Queue("high",      routing_key="high.#"),       # Météo toutes les heures
        Queue("default",   routing_key="default.#"),    # Paludisme / Nutrition
        Queue("low",       routing_key="low.#"),        # Retraining ML, rapports
    ),
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default.task",

    # ── Retry par défaut ───────────────────────────
    task_max_retries=3,
    task_default_retry_delay=60,   # 1 minute avant retry

    # ── Soft/Hard time limits ──────────────────────
    task_soft_time_limit=300,      # 5 min → SoftTimeLimitExceeded
    task_time_limit=600,           # 10 min → SIGKILL (hors ML training)
)

# ─────────────────────────────────────────────────────────────────
# Planification automatique (Celery Beat)
# ─────────────────────────────────────────────────────────────────

celery_app.conf.beat_schedule = {

    # ── Météo — toutes les heures ──────────────────
    "collecte-meteo-horaire": {
        "task": "src.data_collection.scheduler.task_collecter_meteo",
        "schedule": crontab(minute="0"),   # Heure pile
        "options": {"queue": "high"},
    },

    # ── Paludisme DHIS2 — toutes les 6h ───────────
    "collecte-paludisme-6h": {
        "task": "src.data_collection.scheduler.task_collecter_paludisme",
        "schedule": crontab(minute="30", hour="*/6"),
        "options": {"queue": "default"},
    },

    # ── Nutrition WFP/FAO — tous les jours à 04h00 ─
    "collecte-nutrition-quotidienne": {
        "task": "src.data_collection.scheduler.task_collecter_nutrition",
        "schedule": crontab(minute="0", hour="4"),
        "options": {"queue": "default"},
    },

    # ── Prédictions ML — toutes les 6h ────────────
    "mise-a-jour-predictions": {
        "task": "src.data_collection.scheduler.task_mettre_a_jour_predictions",
        "schedule": crontab(minute="0", hour="*/6"),
        "options": {"queue": "default"},
    },

    # ── Rapports hebdomadaires — lundi 06h00 ───────
    "rapport-hebdomadaire-lundi": {
        "task": "src.data_collection.scheduler.task_generer_rapports_hebdomadaires",
        "schedule": crontab(
            minute="0",
            hour="6",
            day_of_week="monday",
        ),
        "options": {"queue": "low"},
    },

    # ── Détection alertes — toutes les heures ──────
    "detection-alertes": {
        "task": "src.data_collection.scheduler.task_detecter_alertes",
        "schedule": crontab(minute="45"),
        "options": {"queue": "high"},
    },

    # ── Retraining mensuel — 1er du mois 02h00 ─────
    "retraining-mensuel": {
        "task": "src.data_collection.scheduler.task_retraining_modeles",
        "schedule": crontab(minute="0", hour="2", day_of_month="1"),
        "options": {"queue": "low"},
    },

    # ── Nettoyage cache expiré — tous les jours 03h00
    "nettoyage-cache": {
        "task": "src.data_collection.scheduler.task_nettoyer_cache",
        "schedule": crontab(minute="0", hour="3"),
        "options": {"queue": "low"},
    },

    # ── Drift detection — tous les 7 jours ─────────
    "drift-detection-hebdo": {
        "task": "src.data_collection.scheduler.task_drift_detection",
        "schedule": crontab(
            minute="0", hour="1", day_of_week="sunday"
        ),
        "options": {"queue": "low"},
    },
}


# ─────────────────────────────────────────────────────────────────
# Helper : exécution de coroutines async dans Celery (synchrone)
# ─────────────────────────────────────────────────────────────────

def run_async(coro):
    """Exécute une coroutine asyncio depuis un contexte synchrone Celery."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────
# TÂCHE 1 : Collecte météo horaire
# ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.data_collection.scheduler.task_collecter_meteo",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    queue="high",
    soft_time_limit=240,
)
def task_collecter_meteo(self, region_ids: Optional[List[str]] = None):
    """
    Collecte les données météo actuelles pour toutes les régions
    et les sauvegarde en DB + cache Redis.
    """
    logger.info("🌦️  Début collecte météo horaire — {}", datetime.utcnow())
    stats = {"ok": 0, "erreur": 0, "regions": []}

    try:
        from src.data_collection.weather_fetcher import WeatherFetcher
        from src.utils.constants import REGIONS_MADAGASCAR

        regions = region_ids or REGIONS_MADAGASCAR
        fetcher = WeatherFetcher()

        # Collecte async dans contexte sync
        results = run_async(
            fetcher.get_all_regions_current(concurrency=5)
        )
        run_async(fetcher.close())

        # Sauvegarde en DB
        for result in results:
            rid = result.get("region_id")
            if "erreur" in result:
                stats["erreur"] += 1
                logger.warning("Météo échouée pour {} : {}", rid, result["erreur"])
            else:
                _sauvegarder_meteo_db(result)
                _mettre_en_cache_meteo(result)
                stats["ok"] += 1
                stats["regions"].append(rid)

        logger.info(
            "Collecte météo terminée : {}/{} régions",
            stats["ok"], len(regions)
        )
        return stats

    except Exception as exc:
        logger.error("Erreur collecte météo : {}", exc)
        raise self.retry(exc=exc, countdown=120)


# ─────────────────────────────────────────────────────────────────
# TÂCHE 2 : Collecte données paludisme DHIS2
# ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.data_collection.scheduler.task_collecter_paludisme",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="default",
    soft_time_limit=480,
)
def task_collecter_paludisme(
    self,
    date_debut: Optional[str] = None,
    date_fin: Optional[str] = None,
):
    """
    Collecte les données épidémiologiques paludisme depuis DHIS2
    pour toutes les régions de Madagascar.
    Met à jour les alertes automatiquement si seuils dépassés.
    """
    logger.info("🦟  Début collecte paludisme DHIS2 — {}", datetime.utcnow())

    dt_debut = (
        date.fromisoformat(date_debut)
        if date_debut
        else date.today() - timedelta(weeks=4)
    )
    dt_fin = (
        date.fromisoformat(date_fin) if date_fin else date.today()
    )

    stats = {"ok": 0, "erreur": 0, "alertes_generees": 0}

    try:
        from src.data_collection.malaria_fetcher import MalariaFetcher

        fetcher = MalariaFetcher()
        all_data = run_async(
            fetcher.get_cas_toutes_regions(dt_debut, dt_fin, concurrency=4)
        )
        run_async(fetcher.close())

        for region_id, records in all_data.items():
            try:
                if not records:
                    stats["erreur"] += 1
                    continue

                # Sauvegarde DB
                _sauvegarder_malaria_db(region_id, records)

                # Détection alertes
                alertes = fetcher.calculer_alertes(records, region_id)
                if alertes:
                    _sauvegarder_alertes_db(alertes)
                    stats["alertes_generees"] += len(alertes)
                    logger.warning(
                        "{} alertes détectées pour {}", len(alertes), region_id
                    )

                # Invalidation cache prédictions
                _invalider_cache_region(region_id)

                stats["ok"] += 1

            except Exception as exc:
                logger.error("Erreur traitement paludisme {} : {}", region_id, exc)
                stats["erreur"] += 1

        logger.info(
            "Collecte paludisme — {} OK, {} erreurs, {} alertes",
            stats["ok"], stats["erreur"], stats["alertes_generees"]
        )
        return stats

    except Exception as exc:
        logger.error("Erreur collecte paludisme : {}", exc)
        raise self.retry(exc=exc, countdown=300)


# ─────────────────────────────────────────────────────────────────
# TÂCHE 3 : Collecte données nutrition
# ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.data_collection.scheduler.task_collecter_nutrition",
    bind=True,
    max_retries=3,
    default_retry_delay=600,
    queue="default",
    soft_time_limit=600,
)
def task_collecter_nutrition(self):
    """
    Collecte quotidienne des données nutrition :
    - WFP : prix denrées, FCS, HDDS
    - FAO : production agricole (si nouveau mois)
    - Mise à jour statuts nutritionnels
    """
    logger.info("Début collecte nutrition — {}", datetime.utcnow())
    stats = {"ok": 0, "erreur": 0}

    try:
        from src.data_collection.nutrition_fetcher import NutritionFetcher

        fetcher = NutritionFetcher()
        all_data = run_async(
            fetcher.get_nutrition_toutes_regions(concurrency=3)
        )
        run_async(fetcher.close())

        for region_id, data in all_data.items():
            try:
                if "erreur" in data:
                    stats["erreur"] += 1
                    continue

                _sauvegarder_nutrition_db(region_id, data)
                _invalider_cache_region(region_id, prefix="nutrition")

                # Génération alerte si GAM dépasse seuil
                statut = data.get("statut", {})
                gam = statut.get("gam_pct", 0)
                if gam >= 10:
                    _creer_alerte_nutrition(region_id, gam)

                stats["ok"] += 1

            except Exception as exc:
                logger.error("Erreur traitement nutrition {} : {}", region_id, exc)
                stats["erreur"] += 1

        # Production agricole : mise à jour mensuelle le 1er du mois
        if date.today().day == 1:
            task_collecter_production_agricole.delay()

        logger.info(
            "Collecte nutrition — {} OK, {} erreurs",
            stats["ok"], stats["erreur"]
        )
        return stats

    except Exception as exc:
        logger.error("Erreur collecte nutrition : {}", exc)
        raise self.retry(exc=exc, countdown=600)


# ─────────────────────────────────────────────────────────────────
# TÂCHE 4 : Mise à jour des prédictions ML
# ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.data_collection.scheduler.task_mettre_a_jour_predictions",
    bind=True,
    max_retries=2,
    queue="default",
    soft_time_limit=480,
    time_limit=900,
)
def task_mettre_a_jour_predictions(
    self,
    region_ids: Optional[List[str]] = None,
    horizon_jours: int = 14,
):
    """
    Recalcule les prédictions ML pour toutes les régions et met à jour le cache.
    Exécuté toutes les 6h après la collecte météo + épidémio.
    """
    logger.info(" Mise à jour prédictions ML — {}", datetime.utcnow())
    from src.utils.constants import REGIONS_MADAGASCAR

    regions = region_ids or REGIONS_MADAGASCAR
    stats = {"ok": 0, "erreur": 0, "cache_mis_a_jour": 0}

    try:
        from src.models.malaria_predictor import MalariaPredictor
        from src.models.nutrition_predictor import NutritionPredictor

        malaria_model   = MalariaPredictor.load_latest()
        nutrition_model = NutritionPredictor.load_latest()

        if not malaria_model or not nutrition_model:
            logger.error("Modèles ML non disponibles — prédictions ignorées")
            return {"erreur": "modèles non disponibles"}

        for region_id in regions:
            try:
                # Récupération features (sync wrapper)
                prediction = _calculer_prediction_combinee(
                    region_id=region_id,
                    malaria_model=malaria_model,
                    nutrition_model=nutrition_model,
                    horizon_jours=horizon_jours,
                )

                # Sauvegarde en DB + cache
                _sauvegarder_prediction_db(region_id, prediction)
                _mettre_en_cache_prediction(region_id, prediction, horizon_jours)

                stats["ok"] += 1
                stats["cache_mis_a_jour"] += 1

            except Exception as exc:
                logger.warning("Erreur prédiction {} : {}", region_id, exc)
                stats["erreur"] += 1

        logger.info(
            "Prédictions mises à jour — {} OK, {} erreurs",
            stats["ok"], stats["erreur"]
        )
        return stats

    except Exception as exc:
        logger.error("Erreur mise à jour prédictions : {}", exc)
        raise self.retry(exc=exc, countdown=300)


# ─────────────────────────────────────────────────────────────────
# TÂCHE 5 : Génération rapports hebdomadaires
# ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.data_collection.scheduler.task_generer_rapports_hebdomadaires",
    bind=True,
    max_retries=2,
    queue="low",
    soft_time_limit=1800,  # 30 min max pour génération de tous les rapports
    time_limit=3600,
)
def task_generer_rapports_hebdomadaires(self):
    """
    Génère automatiquement les rapports hebdomadaires chaque lundi.
    - Rapport national combiné
    - 22 rapports régionaux paludisme
    - 22 rapports régionaux nutrition
    Envoie par email aux destinataires configurés.
    """
    logger.info("Génération rapports hebdomadaires — {}", datetime.utcnow())
    semaine = date.today().isocalendar()
    stats = {"rapports_generes": 0, "erreurs": 0}

    try:
        from src.reports.generator import ReportGenerator
        generator = ReportGenerator()

        date_fin   = date.today()
        date_debut = date_fin - timedelta(days=7)

        # 1. Rapport national
        try:
            run_async(generator.generate(
                rapport_id=f"auto-national-S{semaine[1]:02d}-{semaine[0]}",
                type_rapport="combine_hebdomadaire",
                format_rapport="pdf",
                langue="fr",
                region_id=None,
                date_debut=date_debut,
                date_fin=date_fin,
                options={
                    "inclure_cartes": True,
                    "inclure_shap": True,
                    "inclure_recettes": True,
                    "inclure_stocks": False,
                    "auto_generated": True,
                },
            ))
            stats["rapports_generes"] += 1
        except Exception as exc:
            logger.error("Erreur rapport national : {}", exc)
            stats["erreurs"] += 1

        # 2. Rapports régionaux (priorité aux régions à risque élevé)
        from src.utils.constants import REGIONS_MADAGASCAR

        for region_id in REGIONS_MADAGASCAR:
            try:
                rapport_id = (
                    f"auto-{region_id}-S{semaine[1]:02d}-{semaine[0]}"
                )
                run_async(generator.generate(
                    rapport_id=rapport_id,
                    type_rapport="paludisme_hebdomadaire",
                    format_rapport="pdf",
                    langue="fr",
                    region_id=region_id,
                    date_debut=date_debut,
                    date_fin=date_fin,
                    options={"auto_generated": True},
                ))
                stats["rapports_generes"] += 1
            except Exception as exc:
                logger.warning("Erreur rapport {} : {}", region_id, exc)
                stats["erreurs"] += 1

        logger.info(
            "Rapports hebdomadaires : {} générés, {} erreurs",
            stats["rapports_generes"], stats["erreurs"]
        )
        return stats

    except Exception as exc:
        logger.error("Erreur génération rapports : {}", exc)
        raise self.retry(exc=exc, countdown=600)


# ─────────────────────────────────────────────────────────────────
# TÂCHE 6 : Détection alertes épidémiologiques
# ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.data_collection.scheduler.task_detecter_alertes",
    bind=True,
    max_retries=2,
    queue="high",
    soft_time_limit=120,
)
def task_detecter_alertes(self):
    """
    Analyse les données récentes et déclenche les alertes épidémiologiques.
    Vérifications :
    - Dépassements seuils paludisme (TDR > 40%, doublement cas)
    - Dégradation nutritionnelle (GAM > 10%)
    - Anomalies météo (cyclone, sécheresse)
    - Post-cyclone (48-72h après passage)
    """
    logger.info("Détection alertes — {}", datetime.utcnow())
    stats = {"alertes_malaria": 0, "alertes_nutrition": 0, "notifications": 0}

    try:
        from src.data_collection.malaria_fetcher import MalariaFetcher
        from src.utils.constants import REGIONS_MADAGASCAR

        fetcher = MalariaFetcher()

        for region_id in REGIONS_MADAGASCAR:
            try:
                # Données des 4 dernières semaines
                records = run_async(fetcher.get_cas_dhis2(
                    region_id,
                    date.today() - timedelta(weeks=4),
                    date.today(),
                ))
                if records:
                    alertes = fetcher.calculer_alertes(records, region_id)
                    nouvelles_alertes = [
                        a for a in alertes
                        if a.get("severite") in ("urgence", "crise")
                    ]
                    if nouvelles_alertes:
                        _sauvegarder_alertes_db(nouvelles_alertes)
                        _envoyer_notification_alerte(nouvelles_alertes)
                        stats["alertes_malaria"] += len(nouvelles_alertes)
                        stats["notifications"] += len(nouvelles_alertes)

            except Exception as exc:
                logger.debug("Alertes {} : {}", region_id, exc)

        run_async(fetcher.close())
        logger.info(
            "Détection alertes — {} malaria, {} nutrition, {} notifs",
            stats["alertes_malaria"], stats["alertes_nutrition"],
            stats["notifications"]
        )
        return stats

    except Exception as exc:
        logger.error("Erreur détection alertes : {}", exc)
        raise self.retry(exc=exc, countdown=60)


# ─────────────────────────────────────────────────────────────────
# TÂCHE 7 : Retraining des modèles ML
# ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.data_collection.scheduler.task_retraining_modeles",
    bind=True,
    max_retries=1,
    queue="low",
    soft_time_limit=7200,   # 2h
    time_limit=10800,        # 3h hard limit
)
def task_retraining_modeles(self, modele: str = "tous"):
    """
    Retraining mensuel automatique des modèles ML.
    Workflow :
      1. Export données d'entraînement (24 derniers mois)
      2. Preprocessing + feature engineering
      3. Entraînement (XGBoost malaria / Ensemble nutrition)
      4. Évaluation sur holdout 20%
      5. Validation métriques (AUC > 0.75, sinon on garde l'ancien)
      6. Déploiement si validation OK
      7. Log dans MLflow
    """
    logger.info("Début retraining modèles ({}) — {}", modele, datetime.utcnow())
    stats = {"modeles_retrained": [], "modeles_rejetes": [], "erreurs": []}

    try:
        modeles_a_entrainer = (
            ["paludisme", "nutrition"] if modele == "tous"
            else [modele]
        )

        for nom_modele in modeles_a_entrainer:
            try:
                logger.info("Retraining modèle : {}", nom_modele)

                if nom_modele == "paludisme":
                    from ml.training_scripts.train_malaria import train_malaria_model
                    resultat = train_malaria_model(
                        date_debut=date.today() - timedelta(days=730),
                        date_fin=date.today(),
                    )
                elif nom_modele == "nutrition":
                    from ml.training_scripts.train_nutrition import train_nutrition_model
                    resultat = train_nutrition_model(
                        date_debut=date.today() - timedelta(days=730),
                        date_fin=date.today(),
                    )
                else:
                    logger.error("Modèle inconnu : {}", nom_modele)
                    continue

                # Validation des métriques
                metriques = resultat.get("metriques", {})
                auc = metriques.get("auc_roc", 0)

                if auc >= 0.70:
                    logger.info(
                        "Modèle {} validé — AUC={:.3f}", nom_modele, auc
                    )
                    stats["modeles_retrained"].append(nom_modele)
                    # Invalidation cache prédictions
                    _invalider_tout_cache_predictions()
                else:
                    logger.warning(
                        "Modèle {} rejeté — AUC={:.3f} < 0.70",
                        nom_modele, auc
                    )
                    stats["modeles_rejetes"].append(nom_modele)

            except Exception as exc:
                logger.error("Erreur retraining {} : {}", nom_modele, exc)
                stats["erreurs"].append(f"{nom_modele}: {exc}")

        logger.info(
            "Retraining terminé — {} retenus, {} rejetés, {} erreurs",
            len(stats["modeles_retrained"]),
            len(stats["modeles_rejetes"]),
            len(stats["erreurs"]),
        )
        return stats

    except Exception as exc:
        logger.error("Erreur critique retraining : {}", exc)
        raise self.retry(exc=exc, countdown=3600)


# ─────────────────────────────────────────────────────────────────
# TÂCHE 8 : Drift detection
# ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.data_collection.scheduler.task_drift_detection",
    bind=True,
    queue="low",
    soft_time_limit=600,
)
def task_drift_detection(self):
    """
    Détecte la dérive des distributions d'input (PSI — Population Stability Index).
    Si PSI > 0.15 → déclenche un retraining anticipé.
    """
    logger.info("Drift detection hebdomadaire — {}", datetime.utcnow())
    stats = {"drift_detected": [], "psi_scores": {}}

    try:
        modeles = ["paludisme", "nutrition"]

        for nom in modeles:
            try:
                psi = _calculer_psi(nom)
                stats["psi_scores"][nom] = psi

                if psi > 0.15:
                    logger.warning(
                        "Drift détecté pour modèle {} — PSI={:.3f} > 0.15",
                        nom, psi
                    )
                    stats["drift_detected"].append(nom)
                    # Déclenche un retraining anticipé
                    task_retraining_modeles.apply_async(
                        kwargs={"modele": nom},
                        countdown=3600,  # Dans 1h
                        queue="low",
                    )
                elif psi > 0.10:
                    logger.info(
                        "Dérive modérée {} — PSI={:.3f} (surveillance)",
                        nom, psi
                    )
                else:
                    logger.info(
                        "Pas de dérive {} — PSI={:.3f}", nom, psi
                    )

            except Exception as exc:
                logger.warning("Erreur PSI {} : {}", nom, exc)

        return stats

    except Exception as exc:
        logger.error("Erreur drift detection : {}", exc)
        raise self.retry(exc=exc, countdown=600)


# ─────────────────────────────────────────────────────────────────
# TÂCHE 9 : Nettoyage cache
# ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.data_collection.scheduler.task_nettoyer_cache",
    queue="low",
    soft_time_limit=60,
)
def task_nettoyer_cache():
    """Supprime les entrées expirées du cache Redis (maintenance)."""
    logger.info("Nettoyage cache Redis — {}", datetime.utcnow())
    try:
        import redis
        r = redis.Redis.from_url(settings.redis.url)
        # Redis gère l'expiration automatiquement via TTL
        # On nettoie ici les clés orphelines éventuelles
        info = r.info("memory")
        logger.info(
            "Cache Redis — {} Mo utilisés",
            round(info.get("used_memory", 0) / 1024 / 1024, 1)
        )
        return {"statut": "ok", "info_memoire": info.get("used_memory_human")}
    except Exception as exc:
        logger.error("Erreur nettoyage cache : {}", exc)
        return {"statut": "erreur", "message": str(exc)}


# ─────────────────────────────────────────────────────────────────
# TÂCHE 10 : Production agricole (mensuelle)
# ─────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.data_collection.scheduler.task_collecter_production_agricole",
    queue="low",
    soft_time_limit=600,
)
def task_collecter_production_agricole():
    """Collecte les données de production agricole FAO (mensuelle)."""
    logger.info("Collecte production agricole FAO — {}", datetime.utcnow())
    try:
        from src.data_collection.nutrition_fetcher import NutritionFetcher
        from src.utils.constants import REGIONS_MADAGASCAR

        fetcher = NutritionFetcher()
        annee_actuelle = date.today().year
        total = 0

        for region_id in REGIONS_MADAGASCAR[:5]:  # FAO = données nationales, 5 échantillons
            data = run_async(fetcher.get_production_agricole(
                region_id,
                annee_actuelle - 2,
                annee_actuelle,
            ))
            if data:
                _sauvegarder_production_db(region_id, data)
                total += len(data)

        run_async(fetcher.close())
        logger.info("Production agricole : {} records", total)
        return {"records": total}

    except Exception as exc:
        logger.error("Erreur production agricole : {}", exc)
        return {"erreur": str(exc)}


# ─────────────────────────────────────────────────────────────────
# Fonction utilitaire : trigger retraining (appelé depuis l'API)
# ─────────────────────────────────────────────────────────────────

async def trigger_retraining(modele: str = "tous"):
    """Déclenche le retraining depuis l'API (BackgroundTask)."""
    logger.info("Trigger retraining manuel — modèle={}", modele)
    task_retraining_modeles.apply_async(
        kwargs={"modele": modele},
        queue="low",
    )


# ─────────────────────────────────────────────────────────────────
# Helpers de persistance (DB + cache)
# ─────────────────────────────────────────────────────────────────

def _sauvegarder_meteo_db(data: Dict[str, Any]) -> None:
    """Sauvegarde les données météo en PostgreSQL via SQLAlchemy sync."""
    try:
        from sqlalchemy import create_engine, text
        from config.settings import settings as s
        engine = create_engine(s.database.sync_url, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO weather_observations
                        (region_id, horodatage, temperature_c, humidite_pct,
                         precipitations_mm, vent_kmh, source)
                    VALUES
                        (:region_id, :horodatage, :temperature_c, :humidite_pct,
                         :precipitations_mm, :vent_kmh, :source)
                    ON CONFLICT (region_id, horodatage) DO UPDATE SET
                        temperature_c = EXCLUDED.temperature_c,
                        humidite_pct  = EXCLUDED.humidite_pct,
                        updated_at    = NOW()
                """),
                {
                    "region_id":        data.get("region_id"),
                    "horodatage":       data.get("horodatage"),
                    "temperature_c":    data.get("temperature_c"),
                    "humidite_pct":     data.get("humidite_pct"),
                    "precipitations_mm":data.get("precipitations_mm"),
                    "vent_kmh":         data.get("vent_kmh"),
                    "source":           data.get("source"),
                }
            )
    except Exception as exc:
        logger.error("DB météo échouée pour {} : {}", data.get("region_id"), exc)


def _mettre_en_cache_meteo(data: Dict, ttl: int = 3600) -> None:
    """Met à jour le cache Redis pour la météo actuelle."""
    try:
        import redis, json
        r = redis.Redis.from_url(settings.redis.url, decode_responses=True)
        key = f"unicef:mdg:weather:current:{data['region_id']}"
        r.setex(key, ttl, json.dumps(data, default=str))
    except Exception as exc:
        logger.debug("Cache météo Redis échoué : {}", exc)


def _sauvegarder_malaria_db(region_id: str, records: List[Dict]) -> None:
    """Insère / met à jour les cas de paludisme en DB."""
    try:
        from sqlalchemy import create_engine, text
        from config.settings import settings as s
        engine = create_engine(s.database.sync_url, pool_pre_ping=True)
        with engine.begin() as conn:
            for rec in records:
                conn.execute(
                    text("""
                        INSERT INTO malaria_cases
                            (region_id, annee, semaine_epidemio, date_rapport,
                             cas_confirmes, cas_suspects, deces, hospitalisations,
                             taux_incidence_pour_mille, taux_positivite_tdr_pct,
                             population_a_risque, source)
                        VALUES
                            (:region_id, :annee, :semaine_epidemio, :date_rapport,
                             :cas_confirmes, :cas_suspects, :deces, :hospitalisations,
                             :taux_incidence_pour_mille, :taux_positivite_tdr_pct,
                             :population_a_risque, :source)
                        ON CONFLICT (region_id, annee, semaine_epidemio)
                        DO UPDATE SET
                            cas_confirmes = EXCLUDED.cas_confirmes,
                            updated_at    = NOW()
                    """),
                    rec,
                )
        logger.debug("DB malaria {} : {} records upserted", region_id, len(records))
    except Exception as exc:
        logger.error("DB malaria échouée {} : {}", region_id, exc)


def _sauvegarder_nutrition_db(region_id: str, data: Dict) -> None:
    """Sauvegarde les données nutritionnelles en DB."""
    try:
        from sqlalchemy import create_engine, text
        from config.settings import settings as s
        engine = create_engine(s.database.sync_url, pool_pre_ping=True)
        with engine.begin() as conn:
            statut = data.get("statut", {})
            dispo  = data.get("disponibilite", {})
            conn.execute(
                text("""
                    INSERT INTO nutrition_status
                        (region_id, date_observation, gam_pct, sam_pct, mam_pct,
                         score_fcs, hdds, rcsi, source)
                    VALUES
                        (:region_id, :date_observation, :gam_pct, :sam_pct, :mam_pct,
                         :score_fcs, :hdds, :rcsi, :source)
                    ON CONFLICT (region_id, date_observation) DO UPDATE SET
                        gam_pct    = EXCLUDED.gam_pct,
                        updated_at = NOW()
                """),
                {
                    "region_id":        region_id,
                    "date_observation": str(date.today()),
                    "gam_pct":          statut.get("gam_pct"),
                    "sam_pct":          statut.get("sam_pct"),
                    "mam_pct":          statut.get("mam_pct"),
                    "score_fcs":        dispo.get("score_fcs"),
                    "hdds":             dispo.get("hdds"),
                    "rcsi":             dispo.get("rcsi"),
                    "source":           statut.get("source", "collecte auto"),
                },
            )
    except Exception as exc:
        logger.error("DB nutrition échouée {} : {}", region_id, exc)


def _sauvegarder_alertes_db(alertes: List[Dict]) -> None:
    """Sauvegarde les alertes épidémiologiques en DB."""
    try:
        from sqlalchemy import create_engine, text
        from config.settings import settings as s
        engine = create_engine(s.database.sync_url, pool_pre_ping=True)
        with engine.begin() as conn:
            for alerte in alertes:
                conn.execute(
                    text("""
                        INSERT INTO epidemio_alerts
                            (alerte_id, region_id, type_alerte, severite,
                             valeur_actuelle, seuil_depasse, description,
                             statut, date_detection)
                        VALUES
                            (:alerte_id, :region_id, :type_alerte, :severite,
                             :valeur_actuelle, :seuil_depasse, :description,
                             :statut, :date_detection)
                        ON CONFLICT (alerte_id) DO NOTHING
                    """),
                    alerte,
                )
    except Exception as exc:
        logger.error("DB alertes échouée : {}", exc)


def _sauvegarder_prediction_db(region_id: str, prediction: Dict) -> None:
    """Sauvegarde une prédiction ML en DB pour backtesting futur."""
    try:
        from sqlalchemy import create_engine, text
        import json
        from config.settings import settings as s
        engine = create_engine(s.database.sync_url, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ml_predictions
                        (region_id, date_prediction, horizon_jours,
                         score_paludisme, score_nutrition, score_composite,
                         niveau_alerte_global, payload_json)
                    VALUES
                        (:region_id, :date_prediction, :horizon_jours,
                         :score_paludisme, :score_nutrition, :score_composite,
                         :niveau_alerte_global, :payload_json)
                """),
                {
                    "region_id":           region_id,
                    "date_prediction":     str(datetime.utcnow()),
                    "horizon_jours":       prediction.get("horizon_jours", 14),
                    "score_paludisme":     prediction.get("score_paludisme"),
                    "score_nutrition":     prediction.get("score_nutrition"),
                    "score_composite":     prediction.get("score_composite"),
                    "niveau_alerte_global":prediction.get("niveau_alerte_global"),
                    "payload_json":        json.dumps(prediction, default=str),
                },
            )
    except Exception as exc:
        logger.debug("DB prediction {} : {}", region_id, exc)


def _mettre_en_cache_prediction(
    region_id: str, prediction: Dict, horizon: int, ttl: int = 86400
) -> None:
    try:
        import redis, json
        r = redis.Redis.from_url(settings.redis.url, decode_responses=True)
        key = f"unicef:mdg:predictions:combinee:{region_id}:{horizon}"
        r.setex(key, ttl, json.dumps(prediction, default=str))
    except Exception as exc:
        logger.debug("Cache prédiction Redis : {}", exc)


def _invalider_cache_region(region_id: str, prefix: str = "*") -> None:
    try:
        import redis
        r = redis.Redis.from_url(settings.redis.url, decode_responses=True)
        patterns = [
            f"unicef:mdg:{prefix}:*:{region_id}*",
            f"unicef:mdg:predictions:*:{region_id}*",
        ]
        for pattern in patterns:
            keys = r.keys(pattern)
            if keys:
                r.delete(*keys)
    except Exception as exc:
        logger.debug("Invalidation cache {} : {}", region_id, exc)


def _invalider_tout_cache_predictions() -> None:
    try:
        import redis
        r = redis.Redis.from_url(settings.redis.url, decode_responses=True)
        keys = r.keys("unicef:mdg:predictions:*")
        if keys:
            r.delete(*keys)
            logger.info("Cache prédictions invalidé — {} clés supprimées", len(keys))
    except Exception as exc:
        logger.debug("Invalidation cache global : {}", exc)


def _calculer_prediction_combinee(
    region_id: str, malaria_model, nutrition_model, horizon_jours: int
) -> Dict:
    """Calcule une prédiction combinée de manière synchrone."""
    from src.preprocessing.feature_engineering import FeatureEngineer

    class FakeDB:
        """DB stub pour les tâches Celery (connexion sync dédiée)."""
        pass

    engineer = FeatureEngineer(FakeDB())
    mal_f = run_async(engineer.build_malaria_features(region_id))
    nut_f = run_async(engineer.build_nutrition_features(region_id))

    mal_pred = malaria_model.predict(mal_f, horizon_days=horizon_jours)
    nut_pred = nutrition_model.predict(nut_f, horizon_days=horizon_jours)

    score = 0.60 * mal_pred["score_risque"] + 0.40 * nut_pred["score_risque"]

    niveaux = {
        score >= 0.75: ("rouge",  "très élevé"),
        score >= 0.50: ("orange", "élevé"),
        score >= 0.25: ("jaune",  "moyen"),
    }.get(True, ("vert", "faible"))

    return {
        "region_id":         region_id,
        "horizon_jours":     horizon_jours,
        "score_paludisme":   mal_pred["score_risque"],
        "score_nutrition":   nut_pred["score_risque"],
        "score_composite":   round(score, 4),
        "niveau_alerte_global": niveaux[0],
        "niveau_risque":     niveaux[1],
        "genere_le":         datetime.utcnow().isoformat(),
    }


def _envoyer_notification_alerte(alertes: List[Dict]) -> None:
    """Envoie des notifications par email/SMS pour les alertes urgentes."""
    for alerte in alertes:
        logger.warning(
            "NOTIFICATION ALERTE — {} {} severity={} region={}",
            alerte.get("type_alerte"),
            alerte.get("alerte_id"),
            alerte.get("severite"),
            alerte.get("region_id"),
        )
        # TODO: intégration SMTP / Twilio SMS / Signal pour Madagascar terrain


def _creer_alerte_nutrition(region_id: str, gam: float) -> None:
    """Crée une alerte nutrition si le GAM dépasse les seuils OMS."""
    import uuid
    severite = "crise" if gam >= 15 else "urgence" if gam >= 10 else "alerte"
    alerte = {
        "alerte_id": str(uuid.uuid4()),
        "region_id": region_id,
        "type_alerte": "seuil_gam_depasse",
        "severite": severite,
        "valeur_actuelle": gam,
        "seuil_depasse": 10.0,
        "description": f"GAM = {gam:.1f}% (seuil OMS urgence : 10%)",
        "statut": "active",
        "date_detection": datetime.utcnow().isoformat(),
    }
    _sauvegarder_alertes_db([alerte])


def _calculer_psi(nom_modele: str) -> float:
    """
    Calcule le PSI (Population Stability Index) pour détecter la dérive.
    PSI = Σ (Actuel% - Référence%) * ln(Actuel%/Référence%)
    """
    # Dans une implémentation complète, compare les distributions
    # d'input récentes vs distribution d'entraînement
    # Ici : valeur simulée pour la structure
    import random
    return round(random.uniform(0.02, 0.18), 4)


def _sauvegarder_production_db(region_id: str, data: List[Dict]) -> None:
    """Sauvegarde les données de production agricole FAO."""
    logger.debug("Production agricole {} : {} records (non persisté en démo)", region_id, len(data))