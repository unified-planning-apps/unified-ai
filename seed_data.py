"""
scripts/seed_data.py  (placed alongside docker-compose.yml, mounted at runtime)
=====================
Seeds initial data using the schema from src/database/models.py.
Does NOT use seed_regions.py which targets an incompatible older schema.
Users are created automatically by create_default_admin() at API startup.

Usage:
    python /docker-scripts/seed_data.py
    python /docker-scripts/seed_data.py --dry-run
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────
# When this script is bind-mounted at /docker-scripts/seed_data.py, the app
# lives at /app (set by PYTHONPATH env var in docker-compose). We still need
# the /app root on sys.path for local imports (config.settings, etc.).
APP_ROOT = Path(os.environ.get("APP_ROOT", "/app"))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# ── Main ───────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    import sqlalchemy as sa
    from sqlalchemy import text
    from loguru import logger

    from config.settings import settings

    sync_url = settings.database.sync_url
    logger.info("seed_data — connecting to: {}", sync_url.split("@")[-1])

    engine = sa.create_engine(sync_url, pool_pre_ping=True)

    # ── 1. Regions ─────────────────────────────────────────────────────────
    meta_file = APP_ROOT / "config" / "regions_metadata.json"
    if not meta_file.exists():
        logger.warning("regions_metadata.json not found at {}, skipping", meta_file)
    else:
        raw = json.loads(meta_file.read_text(encoding="utf-8"))
        regions_data = raw.get("regions", [])
        logger.info("Seeding {} regions…", len(regions_data))

        if not dry_run:
            upsert_sql = text("""
                INSERT INTO regions (
                    region_id, name, chef_lieu,
                    latitude, longitude,
                    altitude_mean_m, area_km2, population_2023,
                    climate_zone, malaria_endemicity, metadata_json
                ) VALUES (
                    :region_id, :name, :chef_lieu,
                    :latitude, :longitude,
                    :altitude_mean_m, :area_km2, :population_2023,
                    :climate_zone, :malaria_endemicity, CAST(:metadata_json AS jsonb)
                )
                ON CONFLICT (region_id) DO UPDATE SET
                    name               = EXCLUDED.name,
                    chef_lieu          = EXCLUDED.chef_lieu,
                    latitude           = EXCLUDED.latitude,
                    longitude          = EXCLUDED.longitude,
                    altitude_mean_m    = EXCLUDED.altitude_mean_m,
                    area_km2           = EXCLUDED.area_km2,
                    population_2023    = EXCLUDED.population_2023,
                    climate_zone       = EXCLUDED.climate_zone,
                    malaria_endemicity = EXCLUDED.malaria_endemicity,
                    metadata_json      = EXCLUDED.metadata_json
            """)

            MODEL_KEYS = {
                "id", "name", "chef_lieu", "latitude", "longitude",
                "altitude_mean_m", "area_km2", "population_2023",
                "climate_zone", "malaria_endemicity",
            }

            rows = []
            for r in regions_data:
                extra = {k: v for k, v in r.items() if k not in MODEL_KEYS}
                rows.append({
                    "region_id":          r["id"],
                    "name":               r.get("name", ""),
                    "chef_lieu":          r.get("chef_lieu"),
                    "latitude":           r.get("latitude"),
                    "longitude":          r.get("longitude"),
                    "altitude_mean_m":    r.get("altitude_mean_m"),
                    "area_km2":           r.get("area_km2"),
                    "population_2023":    r.get("population_2023"),
                    "climate_zone":       r.get("climate_zone"),
                    "malaria_endemicity": r.get("malaria_endemicity"),
                    "metadata_json":      json.dumps(extra, ensure_ascii=False),
                })

            with engine.begin() as conn:
                for row in rows:
                    conn.execute(upsert_sql, row)

            logger.info("  ✓ {} regions upserted", len(rows))
        else:
            logger.info("  [dry-run] {} regions would be inserted", len(regions_data))

    # ── 2. Sample recipes ──────────────────────────────────────────────────
    SAMPLE_RECIPES = [
        {
            "recette_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "ravitoto-kitoza")),
            "nom": "Ravitoto sy Kitoza",
            "nom_malgache": "Ravitoto sy Kitoza",
            "regions_adaptees": ["MDG-ANA", "MDG-VAK", "MDG-ITM"],
            "saison": ["saison_pluies", "saison_seche"],
            "cible": ["famille", "femmes_enceintes"],
            "calories_kcal": 420,
            "proteines_g": 22,
            "glucides_g": 48,
            "lipides_g": 15,
            "fer_mg": 6.5,
            "vitamine_a_ug": 120,
            "zinc_mg": 3.2,
            "score_nutritionnel": 78,
            "ingredients": [
                {"nom": "Feuilles de manioc pilées", "quantite_g": 200, "disponible_localement": True},
                {"nom": "Kitoza (viande séchée)", "quantite_g": 100, "disponible_localement": True},
                {"nom": "Ail", "quantite_g": 10, "disponible_localement": True},
                {"nom": "Huile", "quantite_g": 15, "disponible_localement": True},
            ],
            "instructions": "Faire revenir l'ail dans l'huile. Ajouter le kitoza et les feuilles de manioc. Cuire 20 min à feu doux. Servir avec du riz.",
            "temps_preparation_min": 30,
            "cout_estime_ariary": 3500,
        },
        {
            "recette_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "bouillie-enrichie-banane")),
            "nom": "Bouillie enrichie à la banane",
            "nom_malgache": "Vary amin'ny akondro",
            "regions_adaptees": None,
            "saison": ["saison_pluies", "saison_seche"],
            "cible": ["enfants_6_23m", "enfants_2_5ans"],
            "calories_kcal": 180,
            "proteines_g": 4,
            "glucides_g": 35,
            "lipides_g": 3,
            "fer_mg": 2.1,
            "vitamine_a_ug": 45,
            "zinc_mg": 0.8,
            "score_nutritionnel": 82,
            "ingredients": [
                {"nom": "Farine de riz", "quantite_g": 50, "disponible_localement": True},
                {"nom": "Banane mûre", "quantite_g": 80, "disponible_localement": True},
                {"nom": "Lait en poudre", "quantite_g": 10, "disponible_localement": True},
                {"nom": "Sucre", "quantite_g": 5, "disponible_localement": True},
            ],
            "instructions": "Délayer la farine dans un peu d'eau froide. Porter 250 ml d'eau à ébullition. Verser la farine en remuant. Cuire 10 min. Écraser la banane et incorporer avec le lait en poudre.",
            "temps_preparation_min": 15,
            "cout_estime_ariary": 1200,
        },
        {
            "recette_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "soupe-spiruline-legumes")),
            "nom": "Soupe de légumes à la spiruline",
            "nom_malgache": "Haza miaraka amin'ny spiruline",
            "regions_adaptees": None,
            "saison": ["saison_seche"],
            "cible": ["enfants_2_5ans", "femmes_enceintes", "femmes_allaitantes"],
            "calories_kcal": 120,
            "proteines_g": 8,
            "glucides_g": 16,
            "lipides_g": 3,
            "fer_mg": 4.8,
            "vitamine_a_ug": 280,
            "zinc_mg": 1.5,
            "score_nutritionnel": 91,
            "ingredients": [
                {"nom": "Poudre de spiruline", "quantite_g": 5, "disponible_localement": False},
                {"nom": "Carottes", "quantite_g": 80, "disponible_localement": True},
                {"nom": "Patates douces", "quantite_g": 100, "disponible_localement": True},
                {"nom": "Oignons", "quantite_g": 30, "disponible_localement": True},
            ],
            "instructions": "Cuire les légumes en dés dans 500 ml d'eau pendant 15 min. Mixer. Incorporer la spiruline hors du feu pour conserver ses nutriments.",
            "temps_preparation_min": 20,
            "cout_estime_ariary": 1800,
        },
    ]

    logger.info("Seeding {} sample recipes…", len(SAMPLE_RECIPES))

    if not dry_run:
        recipe_sql = text("""
            INSERT INTO recipes (
                recette_id, nom, nom_malgache,
                regions_adaptees, saison, cible,
                calories_kcal, proteines_g, glucides_g, lipides_g,
                fer_mg, vitamine_a_ug, zinc_mg, score_nutritionnel,
                ingredients, instructions,
                temps_preparation_min, cout_estime_ariary, actif
            ) VALUES (
                :recette_id, :nom, :nom_malgache,
                CAST(:regions_adaptees AS jsonb),
                CAST(:saison AS jsonb),
                CAST(:cible AS jsonb),
                :calories_kcal, :proteines_g, :glucides_g, :lipides_g,
                :fer_mg, :vitamine_a_ug, :zinc_mg, :score_nutritionnel,
                CAST(:ingredients AS jsonb),
                :instructions,
                :temps_preparation_min, :cout_estime_ariary, true
            )
            ON CONFLICT (recette_id) DO NOTHING
        """)

        with engine.begin() as conn:
            for recipe in SAMPLE_RECIPES:
                conn.execute(recipe_sql, {
                    **recipe,
                    "regions_adaptees": json.dumps(recipe["regions_adaptees"], ensure_ascii=False),
                    "saison":           json.dumps(recipe["saison"],           ensure_ascii=False),
                    "cible":            json.dumps(recipe["cible"],            ensure_ascii=False),
                    "ingredients":      json.dumps(recipe["ingredients"],      ensure_ascii=False),
                })

        logger.info("  ✓ {} recipes seeded", len(SAMPLE_RECIPES))
    else:
        logger.info("  [dry-run] {} recipes would be inserted", len(SAMPLE_RECIPES))

    engine.dispose()
    logger.info("✅ seed_data.py done.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
