"""
schema/auth.py
===============
Modèles Pydantic pour l'authentification et la gestion des utilisateurs.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, EmailStr, Field


# ─────────────────────────────────────────────
# Requêtes
# ─────────────────────────────────────────────

class LoginRequest(BaseModel):
    """Corps de la requête POST /auth/login"""
    username: str = Field(..., example="admin")
    password: str = Field(..., example="admin123")


class RegisterRequest(BaseModel):
    """Corps de la requête POST /auth/register"""
    username: str     = Field(..., min_length=3, max_length=50, example="agent_sante")
    email:    str     = Field(..., example="agent@sante.mg")
    password: str     = Field(..., min_length=6, example="motdepasse123")
    full_name: Optional[str]  = Field(None, example="Jean Rakoto")
    organisation: Optional[str] = Field(None, example="UNICEF Madagascar")
    role: str         = Field(default="viewer", example="viewer")
    region_id: Optional[str]  = Field(None, example="MDG-ANA")


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., example="ancien_mdp")
    new_password: str = Field(..., min_length=6, example="nouveau_mdp")


class ForgotPasswordRequest(BaseModel):
    """Corps de la requête POST /auth/forgot-password"""
    email: str = Field(..., example="agent@sante.mg")


class ResetPasswordRequest(BaseModel):
    """Corps de la requête POST /auth/reset-password"""
    token:        str = Field(..., description="Token de réinitialisation reçu par email")
    new_password: str = Field(..., min_length=6, example="nouveau_mdp")


# ─────────────────────────────────────────────
# Réponses
# ─────────────────────────────────────────────

class TokenResponse(BaseModel):
    """Réponse retournée après login réussi"""
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int         # secondes
    user: "UserResponse"


class UserResponse(BaseModel):
    """Informations de l'utilisateur connecté"""
    user_id:      int
    username:     str
    email:        str
    full_name:    Optional[str]
    role:         str
    region_id:    Optional[str]
    organisation: Optional[str]
    is_active:    bool

    class Config:
        from_attributes = True