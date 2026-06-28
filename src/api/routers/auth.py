"""
src/api/routers/auth.py
========================
Endpoints d'authentification et gestion des utilisateurs.

Routes :
  POST /auth/login          → obtenir un token JWT
  POST /auth/register       → créer un compte
  GET  /auth/me             → profil de l'utilisateur connecté
  POST /auth/change-password → changer son mot de passe
  GET  /auth/users          → liste des utilisateurs (admin seulement)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from schema.auth import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from services.auth_service import AuthService
from src.api.dependencies import AuthUser, DbSession

router = APIRouter(tags=["🔐 Authentification"])


# ─────────────────────────────────────────────
# POST /auth/login
# ─────────────────────────────────────────────

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Connexion — obtenir un token JWT",
    description="""
Connectez-vous avec vos identifiants pour obtenir un token Bearer JWT.

**Comptes par défaut (développement) :**
- `admin` / `admin123` → accès total
- `demo` / `demo123`  → lecture seule

**Utilisation du token :**
Cliquez sur **Authorize 🔒** en haut de la page Swagger et collez votre `access_token`.
    """,
)
async def login(data: LoginRequest, db: DbSession) -> TokenResponse:
    """Authentifie l'utilisateur et retourne un token JWT."""
    try:
        service = AuthService(db)
        return await service.login(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code":    "IDENTIFIANTS_INCORRECTS",
                "message": str(exc),
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─────────────────────────────────────────────
# POST /auth/register
# ─────────────────────────────────────────────

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Créer un compte utilisateur",
    description="""
Crée un nouveau compte utilisateur.

**Rôles disponibles :**
- `viewer`   → lecture seule (défaut)
- `regional` → accès limité à une région (`region_id` obligatoire)
- `national` → accès toutes les régions
- `admin`    → accès total + gestion utilisateurs

*Note : seul un admin peut créer des comptes avec rôle admin ou national.*
    """,
)
async def register(data: RegisterRequest, db: DbSession) -> UserResponse:
    """Crée un nouveau compte utilisateur."""
    try:
        service = AuthService(db)
        return await service.register(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code":    "UTILISATEUR_EXISTE",
                "message": str(exc),
            },
        )


# ─────────────────────────────────────────────
# GET /auth/me
# ─────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Profil de l'utilisateur connecté",
)
async def get_me(user: AuthUser, db: DbSession) -> UserResponse:
    """Retourne les informations du compte connecté."""
    service = AuthService(db)
    result  = await service.get_me(user.user_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "UTILISATEUR_INTROUVABLE", "message": "Compte introuvable."},
        )
    return result


# ─────────────────────────────────────────────
# POST /auth/change-password
# ─────────────────────────────────────────────

@router.post(
    "/change-password",
    summary="Changer son mot de passe",
    status_code=status.HTTP_200_OK,
)
async def change_password(
    data: ChangePasswordRequest,
    user: AuthUser,
    db: DbSession,
) -> dict:
    """Change le mot de passe de l'utilisateur connecté."""
    try:
        service = AuthService(db)
        await service.change_password(user.user_id, data.old_password, data.new_password)
        return {"message": "Mot de passe changé avec succès."}
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "MOT_DE_PASSE_INCORRECT", "message": str(exc)},
        )


# ─────────────────────────────────────────────
# GET /auth/users  (admin uniquement)
# ─────────────────────────────────────────────

@router.get(
    "/users",
    summary="Liste des utilisateurs (admin)",
    description="Accessible uniquement aux administrateurs.",
)
async def list_users(user: AuthUser, db: DbSession) -> list:
    """Liste tous les utilisateurs — admin seulement."""
    if user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code":    "ACCES_REFUSE",
                "message": "Seuls les administrateurs peuvent lister les utilisateurs.",
            },
        )
    from sqlalchemy import select
    from src.database.models import User as UserModel

    result = await db.execute(select(UserModel).order_by(UserModel.id))
    users  = result.scalars().all()

    return [
        {
            "user_id":      u.id,
            "username":     u.username,
            "email":        u.email,
            "full_name":    u.full_name,
            "role":         u.role,
            "region_id":    u.region_id,
            "organisation": u.organisation,
            "is_active":    u.is_active,
        }
        for u in users
    ]