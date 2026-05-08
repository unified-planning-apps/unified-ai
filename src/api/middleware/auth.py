"""
src/api/middleware/auth.py
===========================
Middleware d'authentification JWT.
- Exclut les routes publiques (health, docs, openapi)
- Injecte les infos utilisateur dans request.state pour les logs
- N'effectue PAS la validation ici (déléguée à get_current_user dans dependencies.py)
  mais journalise les accès et marque les requêtes anonymes.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Routes accessibles sans authentification
PUBLIC_PATHS = frozenset(
    [
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/openapi.json",
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v1/auth/register",
        "/favicon.ico",
    ]
)

# Préfixes publics (health checks Kubernetes, etc.)
PUBLIC_PREFIXES = ("/docs", "/redoc", "/static", "/api/v1/auth")


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware léger d'authentification :
    1. Génère un X-Request-ID unique pour traçabilité
    2. Marque les requêtes non authentifiées (pour logs)
    3. Journalise chaque requête entrante avec son contexte utilisateur
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        # ── Génération ID de requête unique ────────────────────────────
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # ── Extraction token pour contexte de log ──────────────────────
        auth_header: Optional[str] = request.headers.get("Authorization")
        request.state.authenticated = False
        request.state.user_id = None

        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                from jose import jwt
                from config.settings import settings

                payload = jwt.decode(
                    token,
                    settings.jwt.secret_key,
                    algorithms=[settings.jwt.algorithm],
                    options={"verify_exp": True},
                )
                request.state.authenticated = True
                request.state.user_id = payload.get("sub")
                request.state.username = payload.get("username", "inconnu")
                request.state.user_role = payload.get("role", "viewer")

            except Exception:
                # On ne bloque pas ici — la dépendance `get_current_user`
                # renverra le 401 approprié si la route est protégée
                request.state.authenticated = False

        # ── Vérification routes publiques ──────────────────────────────
        path = request.url.path
        is_public = path in PUBLIC_PATHS or any(
            path.startswith(p) for p in PUBLIC_PREFIXES
        )

        if not is_public and not request.state.authenticated:
            # Loggue les tentatives d'accès non authentifiées à des routes protégées
            logger.debug(
                "Accès non authentifié à une route protégée : {} {} — IP: {}",
                request.method,
                path,
                request.client.host if request.client else "inconnu",
            )

        # ── Journalisation entrante ────────────────────────────────────
        start = time.perf_counter()
        logger.bind(
            request_id=request_id,
            user_id=getattr(request.state, "user_id", None),
            method=request.method,
            path=path,
        ).debug("→ Requête reçue")

        response = await call_next(request)

        # ── Journalisation sortante ────────────────────────────────────
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_level = "INFO" if response.status_code < 400 else "WARNING"
        if response.status_code >= 500:
            log_level = "ERROR"

        logger.log(
            log_level,
            "{} {} {} [{:.1f}ms] user={}",
            request.method,
            path,
            response.status_code,
            elapsed_ms,
            getattr(request.state, "user_id", "anonyme"),
        )

        # Propagation du X-Request-ID dans la réponse
        response.headers["X-Request-ID"] = request_id
        return response