"""
src/api/dependencies.py
========================
Dépendances FastAPI injectables dans tous les routers.
Couvre : session DB, client Redis, utilisateur courant, modèles ML, pagination.
"""

from __future__ import annotations

from typing import Annotated, AsyncGenerator, Optional

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Query, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from src.database import AsyncSessionLocal
from src.utils.constants import UserRole

# ─────────────────────────────────────────────
# Schéma Bearer JWT
# ─────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)


# ─────────────────────────────────────────────
# Modèle utilisateur courant (extrait du JWT)
# ─────────────────────────────────────────────
class CurrentUser:
    """Représente l'utilisateur authentifié extrait du token JWT."""

    def __init__(
        self,
        user_id: int,
        username: str,
        email: str,
        role: UserRole,
        region: Optional[str] = None,
    ):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.region = region  # Si rôle REGIONAL, limite aux données de sa région

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN

    @property
    def is_national(self) -> bool:
        return self.role in (UserRole.ADMIN, UserRole.NATIONAL)

    def can_access_region(self, region_id: str) -> bool:
        """Vérifie si l'utilisateur peut accéder aux données d'une région."""
        if self.is_national:
            return True
        return self.region == region_id


# ─────────────────────────────────────────────
# Dépendance : Session base de données
# ─────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Fournit une session SQLAlchemy asynchrone par requête.
    La session est automatiquement fermée après la requête.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


DbSession = Annotated[AsyncSession, Depends(get_db)]


# ─────────────────────────────────────────────
# Dépendance : Client Redis
# ─────────────────────────────────────────────
async def get_redis(request: Request) -> aioredis.Redis:
    """Retourne le client Redis partagé depuis l'état de l'app."""
    return request.app.state.redis


RedisClient = Annotated[aioredis.Redis, Depends(get_redis)]


# ─────────────────────────────────────────────
# Dépendance : Authentification JWT
# ─────────────────────────────────────────────
async def get_current_user(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Security(bearer_scheme)
    ] = None,
) -> CurrentUser:
    """
    Décode le token Bearer JWT et retourne l'utilisateur courant.
    Lève HTTP 401 si le token est absent, invalide ou expiré.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "TOKEN_MANQUANT",
                "message": "Token d'authentification requis.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            settings.jwt.secret_key,
            algorithms=[settings.jwt.algorithm],
        )
        user_id_str = payload.get("sub")
        user_id = int(user_id_str)        
        username: str = payload.get("username")
        email: str = payload.get("email")
        role_str: str = payload.get("role", "viewer")
        region: Optional[str] = payload.get("region")

        if user_id is None or username is None:
            raise ValueError("Payload JWT incomplet")

        role = UserRole(role_str)

    except JWTError as exc:
        logger.warning("Token JWT invalide : {}", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "TOKEN_INVALIDE",
                "message": "Token expiré ou invalide. Veuillez vous reconnecter.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "ROLE_INCONNU",
                "message": f"Rôle utilisateur non reconnu : {exc}",
            },
        ) from exc

    return CurrentUser(
        user_id=user_id,
        username=username,
        email=email,
        role=role,
        region=region,
    )


AuthUser = Annotated[CurrentUser, Depends(get_current_user)]


# ─────────────────────────────────────────────
# Dépendance : Vérification de rôle
# ─────────────────────────────────────────────
def require_role(*roles: UserRole):
    """
    Factory de dépendance qui vérifie que l'utilisateur possède l'un des rôles requis.

    Exemple d'utilisation :
        @router.delete("/...", dependencies=[Depends(require_role(UserRole.ADMIN))])
    """
    async def _check(user: AuthUser) -> CurrentUser:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "ACCES_REFUSE",
                    "message": (
                        f"Cette action nécessite l'un des rôles : "
                        f"{[r.value for r in roles]}. "
                        f"Votre rôle actuel : {user.role.value}"
                    ),
                },
            )
        return user

    return _check


AdminOnly = Depends(require_role(UserRole.ADMIN))
NationalOrAdmin = Depends(require_role(UserRole.ADMIN, UserRole.NATIONAL))


# ─────────────────────────────────────────────
# Dépendance : Modèles ML
# ─────────────────────────────────────────────
async def get_malaria_model(request: Request):
    """Retourne le modèle de prédiction paludisme depuis l'état de l'app."""
    model = request.app.state.malaria_model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODELE_INDISPONIBLE",
                "message": "Le modèle de prédiction paludisme n'est pas disponible.",
            },
        )
    return model


async def get_nutrition_model(request: Request):
    """Retourne le modèle de prédiction nutrition depuis l'état de l'app."""
    model = request.app.state.nutrition_model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODELE_INDISPONIBLE",
                "message": "Le modèle de prédiction nutrition n'est pas disponible.",
            },
        )
    return model


MalariaModel = Annotated[object, Depends(get_malaria_model)]
NutritionModel = Annotated[object, Depends(get_nutrition_model)]


# ─────────────────────────────────────────────
# Dépendance : Pagination
# ─────────────────────────────────────────────
class PaginationParams:
    """Paramètres de pagination standardisés pour tous les endpoints de liste."""

    def __init__(
        self,
        page: int = Query(default=1, ge=1, description="Numéro de page (commence à 1)"),
        taille: int = Query(
            default=20,
            ge=1,
            le=200,
            description="Nombre d'éléments par page (max 200)",
        ),
    ):
        self.page = page
        self.taille = taille
        self.offset = (page - 1) * taille
        self.limit = taille


Pagination = Annotated[PaginationParams, Depends(PaginationParams)]


# ─────────────────────────────────────────────
# Dépendance : Validation région Madagascar
# ─────────────────────────────────────────────
def get_valid_region(
    region_id: str,
    user: AuthUser,
) -> str:
    """
    Vérifie que la région demandée est valide et accessible par l'utilisateur.
    Charge la liste des régions depuis le metadata JSON.
    """
    import json
    from pathlib import Path

    regions_file = Path("config/regions_metadata.json")
    with regions_file.open() as f:
        metadata = json.load(f)

    valid_ids = {r["id"] for r in metadata["regions"]}

    if region_id not in valid_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "REGION_INTROUVABLE",
                "message": f"Région '{region_id}' introuvable. "
                           f"Régions valides : {sorted(valid_ids)}",
            },
        )

    if not user.can_access_region(region_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "ACCES_REGION_REFUSE",
                "message": (
                    f"Vous n'êtes pas autorisé à accéder aux données de '{region_id}'. "
                    f"Votre région : {user.region}"
                ),
            },
        )

    return region_id


# ─────────────────────────────────────────────
# Dépendance : Cache Redis helper
# ─────────────────────────────────────────────
class CacheHelper:
    """
    Utilitaire d'accès au cache Redis.
    Gère la sérialisation JSON, le TTL et les clés préfixées.
    """

    def __init__(self, redis: aioredis.Redis):
        self.redis = redis
        self._prefix = "unicef:mdg:"

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Optional[str]:
        return await self.redis.get(self._key(key))

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        await self.redis.setex(self._key(key), ttl, value)

    async def delete(self, key: str) -> None:
        await self.redis.delete(self._key(key))

    async def exists(self, key: str) -> bool:
        return bool(await self.redis.exists(self._key(key)))

    async def invalidate_pattern(self, pattern: str) -> int:
        """Supprime toutes les clés correspondant au pattern."""
        keys = await self.redis.keys(self._key(pattern))
        if keys:
            return await self.redis.delete(*keys)
        return 0


async def get_cache(redis: RedisClient) -> CacheHelper:
    return CacheHelper(redis)


Cache = Annotated[CacheHelper, Depends(get_cache)]