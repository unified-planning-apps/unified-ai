"""
Logique métier pour l'authentification et la gestion des utilisateurs.

Inclut la création automatique du compte admin par défaut au premier démarrage.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from schema.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from security.jwt import create_access_token
from security.password import hash_password, verify_password
from src.database.models import User


class AuthService:
    """Service d'authentification — utilisé par le router auth.py"""

    def __init__(self, db: AsyncSession):
        self._db = db

    # ─────────────────────────────────────────────
    # Login
    # ─────────────────────────────────────────────

    async def login(self, data: LoginRequest) -> TokenResponse:
        """
        Authentifie un utilisateur et retourne un token JWT.
        Lève ValueError si identifiants incorrects.
        """
        # Cherche l'utilisateur
        user = await self._get_by_username(data.username)

        if user is None:
            raise ValueError("Nom d'utilisateur ou mot de passe incorrect.")

        if not user.is_active:
            raise ValueError("Compte désactivé. Contactez l'administrateur.")

        if not verify_password(data.password, user.hashed_password):
            raise ValueError("Nom d'utilisateur ou mot de passe incorrect.")

        # Génération token
        token, expires_in = create_access_token(
            user_id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            region_id=user.region_id,
        )

        logger.info("Login réussi — utilisateur={} rôle={}", user.username, user.role)

        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=expires_in,
            user=UserResponse(
                user_id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role,
                region_id=user.region_id,
                organisation=user.organisation,
                is_active=user.is_active,
            ),
        )

    # ─────────────────────────────────────────────
    # Register
    # ─────────────────────────────────────────────

    async def register(self, data: RegisterRequest) -> UserResponse:
        """
        Crée un nouveau compte utilisateur.
        Lève ValueError si username ou email déjà utilisés.
        """
        # Vérification unicité username
        existing_username = await self._get_by_username(data.username)
        if existing_username:
            raise ValueError(f"Le nom d'utilisateur '{data.username}' est déjà pris.")

        # Vérification unicité email
        existing_email = await self._get_by_email(data.email)
        if existing_email:
            raise ValueError(f"L'email '{data.email}' est déjà utilisé.")

        # Validation rôle
        valid_roles = ["admin", "national", "regional", "viewer"]
        if data.role not in valid_roles:
            data.role = "viewer"

        # Création
        user = User(
            username=data.username,
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
            role=data.role,
            region_id=data.region_id,
            organisation=data.organisation,
            is_active=True,
        )
        self._db.add(user)
        await self._db.flush()

        logger.info("Nouvel utilisateur créé — {} ({})", user.username, user.role)

        return UserResponse(
            user_id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            region_id=user.region_id,
            organisation=user.organisation,
            is_active=user.is_active,
        )

    # ─────────────────────────────────────────────
    # Profil utilisateur
    # ─────────────────────────────────────────────

    async def get_me(self, user_id: int) -> Optional[UserResponse]:
        """Retourne les infos du compte connecté."""
        stmt = select(User).where(User.id == user_id)
        result = await self._db.execute(stmt)
        user = result.scalar_one_or_none()
        if user is None:
            return None

        return UserResponse(
            user_id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            region_id=user.region_id,
            organisation=user.organisation,
            is_active=user.is_active,
        )

    async def change_password(
        self,
        user_id: int,
        old_password: str,
        new_password: str,
    ) -> bool:
        """Change le mot de passe d'un utilisateur."""
        stmt = select(User).where(User.id == user_id)
        result = await self._db.execute(stmt)
        user = result.scalar_one_or_none()

        if user is None:
            return False

        if not verify_password(old_password, user.hashed_password):
            raise ValueError("Ancien mot de passe incorrect.")

        user.hashed_password = hash_password(new_password)
        await self._db.flush()
        logger.info("Mot de passe changé pour {}", user.username)
        return True

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    async def _get_by_username(self, username: str) -> Optional[User]:
        result = await self._db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def _get_by_email(self, email: str) -> Optional[User]:
        result = await self._db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()


# ─────────────────────────────────────────────
# Création du compte admin par défaut
# ─────────────────────────────────────────────

async def create_default_admin(db: AsyncSession) -> None:
    """
    Crée le compte admin par défaut si aucun utilisateur n'existe.
    Appelé au démarrage de l'application (main.py lifespan).

    Identifiants par défaut :
        username : admin
        password : admin123
        → À CHANGER EN PRODUCTION !
    """
    result = await db.execute(select(User).limit(1))
    existing = result.scalar_one_or_none()

    if existing is not None:
        return  # Des utilisateurs existent déjà

    admin = User(
        username="admin",
        email="admin@unicef-madagascar.org",
        hashed_password=hash_password("admin123"),
        full_name="Administrateur UNICEF",
        role="admin",
        organisation="UNICEF Madagascar",
        is_active=True,
    )
    db.add(admin)

    # Compte de démonstration viewer
    viewer = User(
        username="demo",
        email="demo@unicef-madagascar.org",
        hashed_password=hash_password("demo123"),
        full_name="Compte Démonstration",
        role="viewer",
        organisation="UNICEF Madagascar",
        is_active=True,
    )
    db.add(viewer)

    await db.flush()
    logger.info("Comptes par défaut créés — admin/admin123 et demo/demo123")
    logger.warning(
        "CHANGEZ le mot de passe admin en production ! "
        "POST /api/v1/auth/change-password"
    )