"""
src/api/main.py
================
Point d'entrée principal de l'application FastAPI.
Configure : lifespan, CORS, middleware, routers, gestion d'erreurs globale.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError

from config.settings import settings
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.rate_limiting import RateLimitMiddleware
from src.api.routers import malaria, nutrition, predictions, reports, weather
from src.database import engine, get_redis_pool
from src.utils.logger import setup_logging

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Gère le cycle de vie de l'application :
    - Connexions DB, Redis
    - Chargement des modèles ML en mémoire
    - Nettoyage à l'arrêt
    """
    setup_logging()
    logger.info("Démarrage Malaria-Nutrition Predictor v{}", settings.api_version)

    # ---- Startup ----
    # 1. Vérification connexion PostgreSQL
    try:
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        logger.info("PostgreSQL connecté")
    except Exception as exc:
        logger.error("Échec connexion PostgreSQL : {}", exc)
        raise

    # 2. Connexion Redis
    try:
        app.state.redis = await get_redis_pool()
        await app.state.redis.ping()
        logger.info("Redis connecté")
    except Exception as exc:
        logger.error("Échec connexion Redis : {}", exc)
        raise

    # 3. Pré-chargement modèles ML (lazy via ModelRegistry)
    try:
        from src.models.malaria_predictor import MalariaPredictor
        from src.models.nutrition_predictor import NutritionPredictor

        app.state.malaria_model = MalariaPredictor.load_latest()
        app.state.nutrition_model = NutritionPredictor.load_latest()
        logger.info("Modèles ML chargés en mémoire")
    except FileNotFoundError:
        logger.warning(
            "⚠️  Aucun modèle pré-entraîné trouvé — mode dégradé activé"
        )
        app.state.malaria_model = None
        app.state.nutrition_model = None

    logger.info("API prête — Environnement : {}", settings.app_env)

    yield  # ← Application en cours d'exécution

    # ---- Shutdown ----
    logger.info("Arrêt de l'application...")
    await app.state.redis.close()
    await engine.dispose()
    logger.info("Connexions fermées proprement")


# ---------------------------------------------------------------------------
# Création de l'application
# ---------------------------------------------------------------------------

def create_application() -> FastAPI:
    """Factory qui construit et configure l'instance FastAPI."""

    app = FastAPI(
        title=settings.app_name,
        description=(
            "API de prédiction du risque paludisme et malnutrition pour Madagascar. "
            "Développée pour UNICEF Madagascar — pipeline temps réel, "
            "22 régions, modèles ML explicables."
        ),
        version=settings.api_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url=f"/api/{settings.api_version}/openapi.json",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        contact={
            "name": "UNICEF Madagascar — Tech Team",
            "email": "tech@unicef-madagascar.org",
        },
        license_info={
            "name": "Propriétaire UNICEF",
            "url": "https://www.unicef.org",
        },
    )

    # ------------------------------------------------------------------
    # Middleware (ordre important : s'exécutent dans l'ordre inverse d'ajout)
    # ------------------------------------------------------------------

    # 1. Compression GZip pour les réponses volumineuses (cartes, rapports)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 2. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # 3. Rate Limiting (anti-abus)
    app.add_middleware(RateLimitMiddleware)

    # 4. Authentification JWT
    app.add_middleware(AuthMiddleware)

    # ------------------------------------------------------------------
    # Middleware de timing personnalisé (logging des performances)
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        response.headers["X-Process-Time"] = f"{elapsed:.2f}ms"
        if elapsed > 2000:
            logger.warning(
                "Requête lente [{} ms] : {} {}",
                int(elapsed),
                request.method,
                request.url.path,
            )
        return response

    # ------------------------------------------------------------------
    # Gestionnaires d'erreurs globaux
    # ------------------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Retourne des erreurs de validation lisibles."""
        errors = []
        for err in exc.errors():
            errors.append(
                {
                    "champ": " → ".join(str(loc) for loc in err["loc"]),
                    "message": err["msg"],
                    "type": err["type"],
                }
            )
        logger.warning(
            "Validation échouée sur {} {} : {}",
            request.method,
            request.url.path,
            errors,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "statut": "erreur",
                "code": "VALIDATION_ERROR",
                "erreurs": errors,
                "chemin": str(request.url.path),
            },
        )

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(
        request: Request, exc: SQLAlchemyError
    ) -> JSONResponse:
        logger.error("Erreur base de données : {}", str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "statut": "erreur",
                "code": "DATABASE_ERROR",
                "message": "Service base de données temporairement indisponible.",
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Erreur non gérée sur {} : {}", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "statut": "erreur",
                "code": "INTERNAL_ERROR",
                "message": "Une erreur interne est survenue. L'équipe technique a été notifiée.",
            },
        )

    # ------------------------------------------------------------------
    # Enregistrement des routers
    # ------------------------------------------------------------------
    API_PREFIX = f"/api/{settings.api_version}"

    app.include_router(
        weather.router,
        prefix=f"{API_PREFIX}/meteo",
        tags=["Météorologie"],
    )
    app.include_router(
        malaria.router,
        prefix=f"{API_PREFIX}/paludisme",
        tags=["Paludisme"],
    )
    app.include_router(
        nutrition.router,
        prefix=f"{API_PREFIX}/nutrition",
        tags=["Nutrition"],
    )
    app.include_router(
        predictions.router,
        prefix=f"{API_PREFIX}/predictions",
        tags=["Prédictions ML"],
    )
    app.include_router(
        reports.router,
        prefix=f"{API_PREFIX}/rapports",
        tags=["Rapports"],
    )

    # ------------------------------------------------------------------
    # Endpoints utilitaires
    # ------------------------------------------------------------------

    @app.get(
        "/health",
        summary="Health check",
        tags=["🔧 Système"],
        response_class=ORJSONResponse,
    )
    async def health_check(request: Request) -> dict:
        """Vérifie l'état de tous les services dépendants."""
        checks: dict[str, str] = {}

        # Redis
        try:
            await request.app.state.redis.ping()
            checks["redis"] = "ok"
        except Exception:
            checks["redis"] = "ko"

        # DB
        try:
            async with engine.begin() as conn:
                await conn.execute("SELECT 1")
            checks["postgresql"] = "ok"
        except Exception:
            checks["postgresql"] = "ko"

        # Modèles ML
        checks["modele_paludisme"] = (
            "chargé" if request.app.state.malaria_model else "non disponible"
        )
        checks["modele_nutrition"] = (
            "chargé" if request.app.state.nutrition_model else "non disponible"
        )

        global_status = (
            "sain"
            if all(v in ("ok", "chargé") for v in checks.values())
            else "dégradé"
        )

        return {
            "statut": global_status,
            "version": settings.api_version,
            "environnement": settings.app_env,
            "services": checks,
        }

    @app.get(
        "/",
        summary="Racine API",
        tags=["🔧 Système"],
        response_class=ORJSONResponse,
    )
    async def root() -> dict:
        return {
            "application": settings.app_name,
            "version": settings.api_version,
            "documentation": "/docs",
            "health": "/health",
            "description": "API UNICEF Madagascar — Prédiction Paludisme & Malnutrition",
        }

    return app


# ---------------------------------------------------------------------------
# Instance globale (importée par Uvicorn)
# ---------------------------------------------------------------------------
app = create_application()


# ---------------------------------------------------------------------------
# Démarrage direct (dev uniquement)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        workers=1 if settings.is_development else settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True,
    )