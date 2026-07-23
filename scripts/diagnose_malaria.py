"""
Diagnostic ponctuel : appelle FeatureEngineer directement, SANS avaler les
erreurs, pour voir pourquoi build_training_dataset produit 0 samples.

Usage :
    python scripts/diagnose_feature_engineering.py
"""
from __future__ import annotations

import asyncio
import sys
import traceback
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def main():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import text
    from src.preprocessing.feature_engineering import FeatureEngineer
    from config.settings import settings

    async_url = settings.database.sync_url.replace(
        "postgresql://", "postgresql+asyncpg://", 1
    )
    print(f"URL async utilisée : {async_url}")

    engine = create_async_engine(async_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    region_id = "MDG-ANA"
    target_date = date(2024, 3, 4)  # doit exister dans malaria_observations

    try:
        async with SessionLocal() as session:
            # 0. La session arrive-t-elle même à parler à la DB ?
            print("\n--- Test 0 : connexion brute ---")
            result = await session.execute(text("SELECT count(*) FROM malaria_observations"))
            print("count(*) malaria_observations :", result.scalar())

            engineer = FeatureEngineer(db=session, training_mode=True)
            print("type(engineer._db) :", type(engineer._db).__name__)
            print("_is_real_db() :", engineer._is_real_db())

            # 1. La requête DB malaria brute fonctionne-t-elle ?
            print("\n--- Test 1 : _fetch_malaria_from_db ---")
            records = await engineer._fetch_malaria_from_db(region_id, target_date, weeks=6)
            print(f"{len(records)} enregistrements trouvés")
            if records:
                print("Premier enregistrement :", records[0])

            # 2. build_malaria_features plante-t-il ?
            print("\n--- Test 2 : build_malaria_features (SANS try/except) ---")
            features = await engineer.build_malaria_features(region_id, target_date)
            print(f"OK — {len(features)} clés retournées")

            # 3. _get_label plante-t-il ou retourne-t-il None ?
            print("\n--- Test 3 : _get_label ---")
            label = await engineer._get_label(region_id, target_date, "malaria")
            print("Label obtenu :", label)

    except Exception:
        print("\n!!! EXCEPTION NON ATTRAPÉE — voici la trace complète !!!")
        traceback.print_exc()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())