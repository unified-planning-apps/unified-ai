"""
src/api/middleware/rate_limiting.py
=====================================
Rate limiting par fenêtre glissante (sliding window) stocké dans Redis.
- Limites différenciées par rôle utilisateur
- Limites par IP pour les routes publiques
- Headers standard RateLimit-* dans les réponses
- Bypass configurable pour les health checks
"""

from __future__ import annotations

import json
import time
from typing import Optional, Tuple

from fastapi import Request, Response, status
from fastapi.responses import ORJSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# ─────────────────────────────────────────────────────────────────
# Configuration des limites par contexte
# ─────────────────────────────────────────────────────────────────
RATE_LIMITS: dict[str, Tuple[int, int]] = {
    # (requêtes_max, fenêtre_secondes)
    "admin":      (1000, 60),   # 1000 req/min
    "national":   (500,  60),   # 500 req/min
    "regional":   (200,  60),   # 200 req/min
    "viewer":     (100,  60),   # 100 req/min
    "anonymous":  (30,   60),   # 30 req/min (IP-based)
    "predictions":(50,   60),   # 50 req/min pour endpoints ML (coûteux)
}

# Routes exemptées du rate limiting
EXEMPT_PATHS = frozenset(
    ["/health", "/", "/favicon.ico", "/docs", "/redoc", "/api/v1/openapi.json"]
)

# Routes soumises à une limite plus stricte (endpoints ML)
STRICT_PATH_PREFIXES = ("/api/v1/predictions",)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter Redis.
    Utilise la commande Redis ZADD + ZREMRANGEBYSCORE pour une fenêtre glissante précise.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # ── Bypass total pour les routes exemptées ─────────────────
        if path in EXEMPT_PATHS:
            return await call_next(request)

        # ── Récupération du client Redis depuis l'état app ──────────
        redis = getattr(request.app.state, "redis", None)
        if redis is None:
            # Redis non disponible → on laisse passer (mode dégradé)
            logger.warning("Rate limiter désactivé : Redis non disponible")
            return await call_next(request)

        # ── Détermination de la clé et des limites ──────────────────
        role = getattr(request.state, "user_role", "anonymous")
        user_id = getattr(request.state, "user_id", None)

        # Limite spéciale pour les endpoints de prédiction (coût ML)
        is_strict_path = any(path.startswith(p) for p in STRICT_PATH_PREFIXES)
        if is_strict_path:
            limit_key = "predictions"
        else:
            limit_key = role if role in RATE_LIMITS else "anonymous"

        max_requests, window_seconds = RATE_LIMITS[limit_key]

        # Clé Redis : par user_id si authentifié, sinon par IP
        if user_id:
            redis_key = f"rl:user:{user_id}:{limit_key}"
        else:
            client_ip = self._get_client_ip(request)
            redis_key = f"rl:ip:{client_ip}:{limit_key}"

        # ── Comptage sliding window ──────────────────────────────────
        allowed, current_count, ttl_remaining = await self._check_rate_limit(
            redis, redis_key, max_requests, window_seconds
        )

        # ── Construction headers standard ───────────────────────────
        headers = {
            "X-RateLimit-Limit": str(max_requests),
            "X-RateLimit-Remaining": str(max(0, max_requests - current_count)),
            "X-RateLimit-Reset": str(int(time.time()) + ttl_remaining),
            "X-RateLimit-Window": f"{window_seconds}s",
        }

        if not allowed:
            logger.warning(
                "Rate limit dépassé — clé={} count={}/{} IP={}",
                redis_key,
                current_count,
                max_requests,
                self._get_client_ip(request),
            )
            return ORJSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "statut": "erreur",
                    "code": "RATE_LIMIT_DEPASSE",
                    "message": (
                        f"Trop de requêtes. Limite : {max_requests} requêtes "
                        f"par {window_seconds} secondes. "
                        f"Réessayez dans {ttl_remaining} secondes."
                    ),
                    "retry_after": ttl_remaining,
                },
                headers={
                    **headers,
                    "Retry-After": str(ttl_remaining),
                },
            )

        response = await call_next(request)

        # Ajout des headers rate limit dans la réponse normale
        for key, value in headers.items():
            response.headers[key] = value

        return response

    @staticmethod
    async def _check_rate_limit(
        redis,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> Tuple[bool, int, int]:
        """
        Implémente le sliding window counter via Redis sorted sets.

        Retourne : (autorisé, compte_actuel, secondes_avant_reset)
        """
        now = time.time()
        window_start = now - window_seconds

        pipe = redis.pipeline()
        # 1. Supprimer les entrées hors fenêtre
        pipe.zremrangebyscore(key, 0, window_start)
        # 2. Ajouter la requête courante
        pipe.zadd(key, {str(now): now})
        # 3. Compter les requêtes dans la fenêtre
        pipe.zcard(key)
        # 4. Définir expiration de la clé
        pipe.expire(key, window_seconds + 1)

        results = await pipe.execute()
        current_count: int = results[2]

        allowed = current_count <= max_requests
        ttl_remaining = window_seconds  # approximatif

        return allowed, current_count, ttl_remaining

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extrait l'IP réelle du client (supporte X-Forwarded-For pour les proxies)."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Prend la première IP (client original)
            return forwarded_for.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        if request.client:
            return request.client.host
        return "unknown"