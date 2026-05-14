"""
Configuration centralisée du logging via Loguru.

Fonctions exportées :
  setup_logging()          — appelé au démarrage (main.py, train_*.py, evaluate.py)
  get_logger(name)         — retourne un logger contextualisé
  log_request(...)         — log structuré d'une requête HTTP
  log_prediction(...)      — log structuré d'une prédiction ML
  log_alert(...)           — log structuré d'une alerte épidémio

Caractéristiques :
  - Loguru (pas logging stdlib) — plus simple, meilleur format
  - Rotation automatique (1 semaine)
  - Niveaux : DEBUG (dev), INFO (prod)
  - Format JSON optionnel pour agrégation ELK / Datadog
  - Contexte request_id propagé automatiquement
  - Compatible avec les workers Celery (re-configuration)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────
# Configuration par défaut (surchargée par setup_logging)
# ─────────────────────────────────────────────────────────────────

_CONFIGURED = False  # Flag pour éviter la double-configuration

_FORMAT_CONSOLE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

_FORMAT_FILE = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{line} | "
    "{message}"
)

_FORMAT_JSON = "{message}"  # Le message sera déjà sérialisé en JSON


# ─────────────────────────────────────────────────────────────────
# Fonction principale — contrat avec main.py et train_*.py
# ─────────────────────────────────────────────────────────────────

def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_logs: bool = False,
    rotation: str = "1 week",
    retention: str = "1 month",
    colorize: bool = True,
) -> None:
    """
    Configure Loguru pour toute l'application.

    Appelé UNE SEULE FOIS au démarrage :
      - src/api/main.py (FastAPI startup)
      - ml/training_scripts/train_malaria.py
      - ml/training_scripts/train_nutrition.py
      - ml/training_scripts/evaluate.py

    Args:
        level     : Niveau de log (INFO, DEBUG, WARNING, ERROR)
        log_file  : Chemin fichier log (None = console uniquement)
        json_logs : Si True → format JSON pour ELK/Datadog
        rotation  : Fréquence de rotation (ex: "1 week", "100 MB")
        retention : Durée de rétention (ex: "1 month")
        colorize  : Colorisation console (désactiver en CI/CD)
    """
    global _CONFIGURED

    # Lecture settings si pas de paramètres explicites
    if level is None or log_file is None:
        try:
            from config.settings import settings
            level    = level    or settings.log_level
            log_file = log_file or settings.log_file
        except Exception:
            level    = level    or "INFO"
            log_file = log_file or None

    # Évite double-configuration (ex: workers Celery qui réimportent)
    if _CONFIGURED:
        return

    # ── Suppression handlers par défaut ──────────────────────────
    logger.remove()

    # ── Handler Console ───────────────────────────────────────────
    if json_logs:
        logger.add(
            sys.stdout,
            level=level,
            format=_formatter_json,
            colorize=False,
            serialize=False,
        )
    else:
        logger.add(
            sys.stdout,
            level=level,
            format=_FORMAT_CONSOLE,
            colorize=colorize,
            backtrace=True,
            diagnose=(level == "DEBUG"),
        )

    # ── Handler Fichier (optionnel) ───────────────────────────────
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                str(log_path),
                level=level,
                format=_FORMAT_FILE,
                rotation=rotation,
                retention=retention,
                compression="gz",
                encoding="utf-8",
                backtrace=True,
                diagnose=False,
                enqueue=True,  # Thread-safe pour Celery workers
            )
        except PermissionError:
            # En dev, /var/log peut ne pas être accessible
            fallback = Path("/tmp/malaria-predictor.log")
            logger.add(
                str(fallback),
                level=level,
                format=_FORMAT_FILE,
                rotation="10 MB",
                enqueue=True,
            )
            logger.warning(
                "Fichier log principal inaccessible — fallback {}", fallback
            )

    _CONFIGURED = True

    logger.info(
        "Logging configuré — niveau={} fichier={} json={}",
        level, log_file or "console", json_logs
    )


def reset_logging() -> None:
    """
    Réinitialise la configuration (utile pour les tests unitaires).
    """
    global _CONFIGURED
    logger.remove()
    _CONFIGURED = False


# ─────────────────────────────────────────────────────────────────
# Logger contextualisé
# ─────────────────────────────────────────────────────────────────

def get_logger(name: str):
    """
    Retourne un logger Loguru lié à un contexte nommé.

    Usage :
        log = get_logger("malaria_fetcher")
        log.info("Fetching data for {}", region_id)
    """
    return logger.bind(module=name)


# ─────────────────────────────────────────────────────────────────
# Logs structurés métier
# ─────────────────────────────────────────────────────────────────

def log_prediction(
    region_id: str,
    modele: str,
    score: float,
    niveau: str,
    horizon_jours: int,
    duree_ms: float,
    user_id: Optional[int] = None,
) -> None:
    """
    Log structuré d'une prédiction ML.
    Format uniforme pour monitoring des performances modèle.
    """
    logger.bind(
        event="prediction",
        region_id=region_id,
        modele=modele,
        score=round(score, 4),
        niveau=niveau,
        horizon_jours=horizon_jours,
        duree_ms=round(duree_ms, 1),
        user_id=user_id,
    ).info(
        "Prédiction {} | région={} score={:.3f} niveau={} [{:.0f}ms]",
        modele, region_id, score, niveau, duree_ms
    )


def log_alert(
    alerte_id: str,
    region_id: str,
    type_alerte: str,
    severite: str,
    valeur: float,
) -> None:
    """Log structuré d'une alerte épidémiologique."""
    niveau_log = {
        "surveillance": "INFO",
        "alerte":        "WARNING",
        "urgence":       "ERROR",
        "crise":         "CRITICAL",
    }.get(severite, "WARNING")

    logger.bind(
        event="alerte",
        alerte_id=alerte_id,
        region_id=region_id,
        type_alerte=type_alerte,
        severite=severite,
        valeur=valeur,
    ).log(
        niveau_log,
        "ALERTE {} | {} | region={} valeur={:.2f}",
        severite.upper(), type_alerte, region_id, valeur
    )


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[int] = None,
    request_id: Optional[str] = None,
) -> None:
    """Log structuré d'une requête HTTP (utilisé par auth middleware)."""
    level = "INFO" if status_code < 400 else "WARNING" if status_code < 500 else "ERROR"

    logger.bind(
        event="http_request",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=round(duration_ms, 1),
        user_id=user_id,
        request_id=request_id,
    ).log(
        level,
        "{} {} {} [{:.0f}ms]",
        method, path, status_code, duration_ms
    )


def log_collecte(
    source: str,
    region_id: Optional[str],
    n_records: int,
    duree_sec: float,
    statut: str = "ok",
) -> None:
    """Log structuré d'une collecte de données (scheduler Celery)."""
    logger.bind(
        event="collecte",
        source=source,
        region_id=region_id or "all",
        n_records=n_records,
        duree_sec=round(duree_sec, 2),
        statut=statut,
    ).info(
        "Collecte {} | region={} records={} [{:.1f}s] statut={}",
        source, region_id or "all", n_records, duree_sec, statut
    )


def log_training(
    modele: str,
    metriques: Dict[str, float],
    duree_sec: float,
    valide: bool,
    n_samples: int,
) -> None:
    """Log structuré d'un entraînement ML (train_*.py)."""
    auc = metriques.get("auc_roc", 0)
    logger.bind(
        event="training",
        modele=modele,
        auc_roc=round(auc, 4),
        f1=round(metriques.get("f1_score", 0), 4),
        n_samples=n_samples,
        duree_sec=round(duree_sec, 1),
        valide=valide,
    ).info(
        "Training {} | AUC={:.3f} F1={:.3f} n={} [{:.0f}s] valide={}",
        modele, auc, metriques.get("f1_score", 0),
        n_samples, duree_sec, valide
    )


# ─────────────────────────────────────────────────────────────────
# Formatter JSON (pour ELK / Datadog)
# ─────────────────────────────────────────────────────────────────

def _formatter_json(record: dict) -> str:
    """Formate un enregistrement Loguru en JSON structuré."""
    log_entry = {
        "timestamp":  record["time"].isoformat(),
        "level":      record["level"].name,
        "logger":     record["name"],
        "line":       record["line"],
        "message":    record["message"],
        "module":     record["module"],
        "function":   record["function"],
        "process_id": record["process"].id,
        "thread_id":  record["thread"].id,
    }

    # Ajout des champs extra (bind)
    extra = record.get("extra", {})
    if extra:
        log_entry["extra"] = extra

    # Exception si présente
    if record["exception"]:
        exc = record["exception"]
        log_entry["exception"] = {
            "type":    str(exc.type.__name__) if exc.type else None,
            "value":   str(exc.value) if exc.value else None,
        }

    return json.dumps(log_entry, ensure_ascii=False, default=str) + "\n"


# ─────────────────────────────────────────────────────────────────
# Décorateur de timing automatique
# ─────────────────────────────────────────────────────────────────

def log_timing(func_name: Optional[str] = None):
    """
    Décorateur qui log automatiquement la durée d'exécution d'une fonction.

    Usage :
        @log_timing("predict_malaria")
        async def ma_fonction():
            ...
    """
    import functools
    import time

    def decorator(func):
        name = func_name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug("{} terminé en {:.1f}ms", name, elapsed)
                return result
            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error("{} échoué en {:.1f}ms : {}", name, elapsed, exc)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug("{} terminé en {:.1f}ms", name, elapsed)
                return result
            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error("{} échoué en {:.1f}ms : {}", name, elapsed, exc)
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator