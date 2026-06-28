"""
security/jwt.py
================
Création et vérification des tokens JWT.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from loguru import logger

from config.settings import settings


def create_access_token(
    user_id: int,
    username: str,
    email: str,
    role: str,
    region_id: Optional[str] = None,
) -> tuple[str, int]:
    """
    Crée un token JWT signé.

    Returns:
        (token_str, expires_in_seconds)
    """
    expires_delta = timedelta(minutes=settings.jwt.access_token_expire_minutes)
    expire        = datetime.now(timezone.utc) + expires_delta

    payload: Dict[str, Any] = {
        "sub":      str(user_id),
        "username": username,
        "email":    email,
        "role":     role,
        "region":   region_id,
        "exp":      expire,
        "iat":      datetime.now(timezone.utc),
    }

    token = jwt.encode(
        payload,
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm,
    )

    expires_in = int(expires_delta.total_seconds())
    return token, expires_in


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Décode et valide un token JWT.
    Retourne le payload ou None si invalide/expiré.
    """
    try:
        return jwt.decode(
            token,
            settings.jwt.secret_key,
            algorithms=[settings.jwt.algorithm],
        )
    except JWTError as exc:
        logger.debug("Token JWT invalide : {}", exc)
        return None