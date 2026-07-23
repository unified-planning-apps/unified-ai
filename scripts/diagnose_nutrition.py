"""
Diagnostic ponctuel nutrition : appelle FeatureEngineer directement pour une
date proche d'une vraie donnée MICS6 (2018-06-01, région MDG-ANA), SANS
avaler les erreurs.

Usage :
    python scripts/diagnose_nutrition.py
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
    target_date = date(2018, 6, 15)  # 2 semaines après l'enquête MICS6 (2018-06-01)

    try:
        async with SessionLocal() as session:
            print("\n--- Test 0 : contenu réel autour de la date cible ---")
            result = await session.execute(text("""
                SELECT region_code, date_enquete, gam_pct, sam_pct, mam_pct, source
                FROM nutrition_observations
                WHERE region_code = :r
                ORDER BY date_enquete
            """), {"r": region_id})
            rows = result.fetchall()
            print(f"{len(rows)} lignes nutrition_observations pour {region_id} :")
            for r in rows:
                print(" ", dict(r._mapping))

            engineer = FeatureEngineer(db=session, training_mode=True)

            print("\n--- Test 1 : _fetch_nutrition_from_db (label, ref=target_date) ---")
            records = await engineer._fetch_nutrition_from_db(region_id, target_date, months=1)
            print(f"{len(records)} enregistrements trouvés")
            if records:
                print("Dernier enregistrement :", records[-1])

            print("\n--- Test 2 : build_nutrition_features (SANS try/except) ---")
            features = await engineer.build_nutrition_features(region_id, target_date)
            print(f"OK — {len(features)} clés retournées")

            print("\n--- Test 3 : _get_label ---")
            label = await engineer._get_label(region_id, target_date, "nutrition")
            print("Label obtenu :", label)

    except Exception:
        print("\n!!! EXCEPTION NON ATTRAPÉE — voici la trace complète !!!")
        traceback.print_exc()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())