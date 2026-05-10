"""
Initialisation de la couche base de données.

Exporte :
  - engine           : AsyncEngine SQLAlchemy (utilisé par main.py au startup)
  - AsyncSessionLocal: SessionFactory pour get_db()
  - get_redis_pool() : Pool Redis async (utilisé par main.py au startup)
  - Base             : DeclarativeBase pour tous les modèles ORM
"""

from __future__ import annotations

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config.settings import settings

# ─────────────────────────────────────────────────────────────────
# SQLAlchemy Async Engine
# ─────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_pre_ping=True,          # Vérifie connexion avant utilisation
    pool_recycle=3600,           # Recycle toutes les heures (évite timeout PostgreSQL)
    echo=settings.database.echo, # Logs SQL en dev uniquement
    future=True,
)

# ─────────────────────────────────────────────────────────────────
# Session Factory
# ─────────────────────────────────────────────────────────────────

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # Pas de rechargement auto après commit
    autocommit=False,
    autoflush=False,
)

# ─────────────────────────────────────────────────────────────────
# Base ORM
# ─────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Base déclarative pour tous les modèles SQLAlchemy du projet."""
    pass

# ─────────────────────────────────────────────────────────────────
# Redis Pool
# ─────────────────────────────────────────────────────────────────

async def get_redis_pool() -> aioredis.Redis:
    """
    Crée et retourne un pool de connexions Redis async.
    Appelé par main.py au démarrage de l'application.
    """
    return await aioredis.from_url(
        settings.redis.url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=50,
    )


__all__ = [
    "engine",
    "AsyncSessionLocal",
    "Base",
    "get_redis_pool",
]