"""
Déploiement automatisé de la plateforme UNICEF Madagascar.

Ce script orchestre le cycle complet de déploiement applicatif :
  1. Validation de l'environnement (.env, variables critiques)
  2. Vérification PostgreSQL / Redis / stockage ML
  3. Exécution des migrations Alembic
  4. Initialisation DB optionnelle (premier déploiement)
  5. Vérification des modèles ML
  6. Collecte des assets statiques
  7. Vérification de santé applicative
  8. Redémarrage contrôlé des services
  9. Smoke tests post-déploiement
 10. Rollback automatique si échec critique

Compatible :
  - Docker Compose
  - Kubernetes Jobs
  - GitHub Actions / GitLab CI
  - Déploiement manuel SSH

Usage :
    python scripts/deploy.py
    python scripts/deploy.py --env staging
    python scripts/deploy.py --skip-tests
    python scripts/deploy.py --init-db
    python scripts/deploy.py --rollback-on-failure
    python scripts/deploy.py --dry-run

Auteur : Équipe Data UNICEF Madagascar
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests

# ── Résolution racine projet ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.utils.logger import (
    get_logger,
    log_alert,
    setup_logging,
)

setup_logging()
log = get_logger("deploy")


# ─────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────

REQUIRED_ENV_VARS = [
    "DATABASE_URL",
    "SECRET_KEY",
]

OPTIONAL_SERVICES = [
    "REDIS_URL",
    "MLFLOW_TRACKING_URI",
]

HEALTH_ENDPOINTS = [
    "/health",
    "/health/live",
    "/health/ready",
]

DEPLOY_LOCK_FILE = ROOT / ".deploy.lock"

DEFAULT_TIMEOUT = 120


# ─────────────────────────────────────────────────────────────────
# Helpers shell
# ─────────────────────────────────────────────────────────────────

def run_command(
    command: List[str],
    *,
    cwd: Optional[Path] = None,
    timeout: int = DEFAULT_TIMEOUT,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Exécute une commande système avec logs structurés.
    """
    cmd_str = " ".join(command)

    log.info("Exécution : {}", cmd_str)

    try:
        result = subprocess.run(
            command,
            cwd=cwd or ROOT,
            timeout=timeout,
            check=False,
            capture_output=capture_output,
            text=True,
        )

        if result.stdout:
            log.debug("STDOUT:\n{}", result.stdout.strip())

        if result.stderr:
            log.debug("STDERR:\n{}", result.stderr.strip())

        if check and result.returncode != 0:
            raise RuntimeError(
                f"Commande échouée ({result.returncode}) : {cmd_str}"
            )

        return result

    except subprocess.TimeoutExpired:
        log.error("Timeout commande : {}", cmd_str)
        raise

    except Exception as exc:
        log.exception("Erreur commande '{}': {}", cmd_str, exc)
        raise


# ─────────────────────────────────────────────────────────────────
# Validation environnement
# ─────────────────────────────────────────────────────────────────

def validate_environment() -> None:
    """
    Vérifie les variables critiques.
    """
    log.info("Validation environnement...")

    missing = []

    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        raise RuntimeError(
            f"Variables d'environnement manquantes : {missing}"
        )

    for var in OPTIONAL_SERVICES:
        if os.getenv(var):
            log.info("  ✓ {} configuré", var)
        else:
            log.warning("  ⚠ {} absent", var)

    log.info("✓ Variables environnement validées")


def validate_directories() -> None:
    """
    Vérifie les répertoires critiques.
    """
    log.info("Validation structure fichiers...")

    required_dirs = [
        ROOT / "models",
        ROOT / "logs",
        ROOT / "data",
        ROOT / "scripts",
        ROOT / "src",
    ]

    for path in required_dirs:
        path.mkdir(parents=True, exist_ok=True)
        log.info("  ✓ {}", path.relative_to(ROOT))


# ─────────────────────────────────────────────────────────────────
# Base de données
# ─────────────────────────────────────────────────────────────────

def check_database_connection() -> None:
    """
    Vérifie PostgreSQL.
    """
    log.info("Vérification PostgreSQL...")

    run_command([
        "python",
        "-c",
        (
            "from sqlalchemy import create_engine,text;"
            "from config.settings import settings;"
            "e=create_engine(settings.database_url);"
            "c=e.connect();"
            "c.execute(text('SELECT 1'));"
            "print('DB OK');"
        )
    ])

    log.info("✓ PostgreSQL accessible")


def run_alembic_migrations() -> None:
    """
    Exécute Alembic upgrade head.
    """
    log.info("Migration Alembic...")

    run_command([
        "alembic",
        "upgrade",
        "head",
    ])

    log.info("✓ Migrations appliquées")


def initialize_database() -> None:
    """
    Initialise complètement la DB.
    """
    log.warning("Initialisation DB complète demandée")

    run_command([
        "python",
        "scripts/init_db.py",
    ], timeout=900)

    log.info("✓ Base initialisée")


# ─────────────────────────────────────────────────────────────────
# Vérification modèles ML
# ─────────────────────────────────────────────────────────────────

def verify_ml_models() -> None:
    """
    Vérifie la présence des artefacts ML.
    """
    log.info("Vérification modèles ML...")

    model_dir = ROOT / "models"

    expected = [
        "malaria",
        "nutrition",
    ]

    missing = []

    for model_name in expected:
        path = model_dir / model_name

        if not path.exists():
            missing.append(model_name)
            continue

        files = list(path.glob("*"))

        if not files:
            missing.append(model_name)

    if missing:
        raise RuntimeError(
            f"Modèles ML manquants : {missing}"
        )

    log.info("✓ Modèles ML validés")


# ─────────────────────────────────────────────────────────────────
# Collecte statiques
# ─────────────────────────────────────────────────────────────────

def collect_static_assets() -> None:
    """
    Prépare assets statiques frontend.
    """
    frontend_dir = ROOT / "frontend"

    if not frontend_dir.exists():
        log.warning("Frontend absent — skip assets")
        return

    package_json = frontend_dir / "package.json"

    if not package_json.exists():
        log.warning("package.json absent — skip frontend")
        return

    log.info("Build frontend...")

    run_command(
        ["npm", "install"],
        cwd=frontend_dir,
        timeout=900,
    )

    run_command(
        ["npm", "run", "build"],
        cwd=frontend_dir,
        timeout=1800,
    )

    log.info("✓ Frontend build terminé")


# ─────────────────────────────────────────────────────────────────
# Services
# ─────────────────────────────────────────────────────────────────

def restart_services(env: str) -> None:
    """
    Redémarre les services applicatifs.
    """
    log.info("Redémarrage services ({})...", env)

    docker_compose = shutil.which("docker-compose")

    if docker_compose:
        run_command([
            docker_compose,
            "restart",
        ])
        log.info("✓ Services Docker redémarrés")
        return

    systemctl = shutil.which("systemctl")

    if systemctl:
        services = [
            "unicef-api",
            "unicef-celery-worker",
            "unicef-celery-beat",
        ]

        for service in services:
            run_command([
                systemctl,
                "restart",
                service,
            ])

        log.info("✓ Services systemd redémarrés")
        return

    log.warning(
        "Aucun orchestrateur détecté — restart manuel requis"
    )


# ─────────────────────────────────────────────────────────────────
# Health checks
# ─────────────────────────────────────────────────────────────────

def check_health(base_url: str) -> None:
    """
    Vérifie endpoints de santé.
    """
    log.info("Health checks API...")

    for endpoint in HEALTH_ENDPOINTS:
        url = f"{base_url.rstrip('/')}{endpoint}"

        try:
            response = requests.get(url, timeout=10)

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Health check KO : {url}"
                )

            log.info(
                "  ✓ {} [{}]",
                endpoint,
                response.status_code,
            )

        except Exception as exc:
            log.error(
                "Health check échoué : {} ({})",
                endpoint,
                exc,
            )
            raise

    log.info("✓ API healthy")


# ─────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────

def run_smoke_tests() -> None:
    """
    Tests critiques post-déploiement.
    """
    log.info("Smoke tests...")

    tests = [
        "tests/smoke/test_api_health.py",
        "tests/smoke/test_prediction_flow.py",
    ]

    for test_file in tests:
        path = ROOT / test_file

        if not path.exists():
            log.warning("Test absent : {}", test_file)
            continue

        run_command([
            "pytest",
            str(path),
            "-q",
        ], timeout=600)

    log.info("✓ Smoke tests OK")


# ─────────────────────────────────────────────────────────────────
# Rollback
# ─────────────────────────────────────────────────────────────────

def rollback_deployment() -> None:
    """
    Rollback simplifié.
    """
    log.warning("Rollback déploiement...")

    try:
        run_command([
            "alembic",
            "downgrade",
            "-1",
        ])

        restart_services(settings.environment)

        log.warning("✓ Rollback effectué")

    except Exception as exc:
        log.critical("Rollback échoué : {}", exc)

        log_alert(
            alerte_id="deploy_rollback_failed",
            region_id="system",
            type_alerte="deployment",
            severite="crise",
            valeur=1.0,
        )


# ─────────────────────────────────────────────────────────────────
# Lock fichier
# ─────────────────────────────────────────────────────────────────

def acquire_lock() -> None:
    """
    Empêche plusieurs déploiements simultanés.
    """
    if DEPLOY_LOCK_FILE.exists():
        raise RuntimeError(
            "Déploiement déjà en cours (.deploy.lock)"
        )

    DEPLOY_LOCK_FILE.write_text(
        json.dumps({
            "pid": os.getpid(),
            "timestamp": time.time(),
        })
    )

    log.info("✓ Lock déploiement acquis")


def release_lock() -> None:
    """
    Supprime le lock.
    """
    if DEPLOY_LOCK_FILE.exists():
        DEPLOY_LOCK_FILE.unlink()

    log.info("✓ Lock libéré")


# ─────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Déploiement plateforme UNICEF Madagascar"
    )

    parser.add_argument(
        "--env",
        default="production",
        choices=["development", "staging", "production"],
    )

    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialise complètement la DB",
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Ignore les smoke tests",
    )

    parser.add_argument(
        "--rollback-on-failure",
        action="store_true",
        help="Rollback auto si échec",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les étapes sans exécution",
    )

    args = parser.parse_args()

    steps = [
        "validate_environment",
        "validate_directories",
        "check_database_connection",
        "run_alembic_migrations",
        "verify_ml_models",
        "collect_static_assets",
        "restart_services",
        "check_health",
        "run_smoke_tests",
    ]

    if args.init_db:
        steps.insert(3, "initialize_database")

    if args.dry_run:
        log.info("DRY RUN — étapes prévues :")
        for idx, step in enumerate(steps, start=1):
            log.info("  {}. {}", idx, step)
        return

    started = time.perf_counter()

    try:
        acquire_lock()

        validate_environment()

        validate_directories()

        check_database_connection()

        if args.init_db:
            initialize_database()

        run_alembic_migrations()

        verify_ml_models()

        collect_static_assets()

        restart_services(args.env)

        base_url = getattr(
            settings,
            "api_public_url",
            "http://localhost:8000",
        )

        check_health(base_url)

        if not args.skip_tests:
            run_smoke_tests()

        elapsed = time.perf_counter() - started

        log.info(
            "✓ Déploiement terminé avec succès en {:.1f}s",
            elapsed,
        )

    except Exception as exc:
        log.exception("Échec déploiement : {}", exc)

        if args.rollback_on_failure:
            rollback_deployment()

        sys.exit(1)

    finally:
        release_lock()


if __name__ == "__main__":
    main()