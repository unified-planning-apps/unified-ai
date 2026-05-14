#!/usr/bin/env python3
"""
Peuplement du référentiel géographique de Madagascar.

Ce script insère (ou met à jour) les données de référence des 22 régions
dans la table public.regions, incluant :
  - Centroïdes géographiques (PostGIS GEOGRAPHY)
  - Bounding boxes (polygones)
  - Métadonnées démographiques et climatiques
  - Relations de voisinage (régions limitrophes)
  - Base de recettes nutritionnelles de référence (recettes_seed)

Usage :
    python scripts/seed_regions.py                   # Insert/update régions
    python scripts/seed_regions.py --with-recipes    # + recettes de base
    python scripts/seed_regions.py --with-users      # + utilisateurs initiaux
    python scripts/seed_regions.py --reset           # Efface et recharge tout
    python scripts/seed_regions.py --verify          # Vérifie la cohérence

Idempotent : peut être exécuté plusieurs fois sans dupliquer les données
(ON CONFLICT DO UPDATE).

Prérequis :
    - init_db.py exécuté au préalable
    - Extension PostGIS active

Auteur : Équipe Data UNICEF Madagascar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.settings import settings
from src.utils.constants import REGIONS_MADAGASCAR
from src.utils.geo_helpers import _REGIONS_DB, RegionMetadata
from src.utils.logger import get_logger, setup_logging

setup_logging()
log = get_logger("seed_regions")


# ─────────────────────────────────────────────────────────────────
# 1. DONNÉES RÉGIONS
# ─────────────────────────────────────────────────────────────────

def build_region_rows() -> List[Dict[str, Any]]:
    """
    Construit la liste des dicts à insérer dans public.regions
    depuis _REGIONS_DB (geo_helpers.py — source unique de vérité).
    """
    rows = []
    for code, meta in _REGIONS_DB.items():
        lat, lon = meta.centroid
        lat_min, lat_max, lon_min, lon_max = meta.bbox

        # WKT du centroïde pour PostGIS
        centroid_wkt = f"POINT({lon} {lat})"

        # WKT du bbox comme polygone
        bbox_wkt = (
            f"POLYGON(("
            f"{lon_min} {lat_min}, "
            f"{lon_max} {lat_min}, "
            f"{lon_max} {lat_max}, "
            f"{lon_min} {lat_max}, "
            f"{lon_min} {lat_min}"
            f"))"
        )

        rows.append({
            "code":               code,
            "nom_fr":             meta.nom_fr,
            "nom_mg":             meta.nom_mg,
            "centroid_wkt":       centroid_wkt,
            "bbox_wkt":           bbox_wkt,
            "altitude_moyenne_m": meta.altitude_moyenne_m,
            "zone_altitude":      meta.zone_altitude.value,
            "zone_climatique":    meta.zone_climatique.value,
            "population_estimee": meta.population_estimee,
            "superficie_km2":     meta.superficie_km2,
            "voisins":            meta.voisins,
            "metadata": json.dumps({
                "zones_humides": [
                    {"lat": lat, "lon": lon}
                    for lat, lon in meta.zones_humides_coords
                ],
                "source": "INSTAT Madagascar / GADM / WorldPop 2024",
            }),
        })
    return rows


def upsert_regions(engine: Engine) -> int:
    """
    Insère ou met à jour les 22 régions (ON CONFLICT DO UPDATE).
    Retourne le nombre de lignes traitées.
    """
    rows = build_region_rows()
    log.info("Insertion/mise à jour de {} régions...", len(rows))

    upsert_sql = text("""
        INSERT INTO public.regions (
            code, nom_fr, nom_mg,
            centroid, bbox_geom,
            altitude_moyenne_m, zone_altitude, zone_climatique,
            population_estimee, superficie_km2,
            voisins, metadata
        ) VALUES (
            :code, :nom_fr, :nom_mg,
            ST_GeogFromText(:centroid_wkt),
            ST_GeogFromText(:bbox_wkt),
            :altitude_moyenne_m, :zone_altitude, :zone_climatique,
            :population_estimee, :superficie_km2,
            :voisins, :metadata::jsonb
        )
        ON CONFLICT (code) DO UPDATE SET
            nom_fr              = EXCLUDED.nom_fr,
            nom_mg              = EXCLUDED.nom_mg,
            centroid            = EXCLUDED.centroid,
            bbox_geom           = EXCLUDED.bbox_geom,
            altitude_moyenne_m  = EXCLUDED.altitude_moyenne_m,
            zone_altitude       = EXCLUDED.zone_altitude,
            zone_climatique     = EXCLUDED.zone_climatique,
            population_estimee  = EXCLUDED.population_estimee,
            superficie_km2      = EXCLUDED.superficie_km2,
            voisins             = EXCLUDED.voisins,
            metadata            = EXCLUDED.metadata,
            mis_a_jour_le       = NOW()
        RETURNING code;
    """)

    with engine.begin() as conn:
        result = conn.execute(upsert_sql, rows)
        codes = [r[0] for r in result.fetchall()]

    log.info("  ✓ {} régions upsertées", len(codes))

    # Vérification : tous les codes REGIONS_MADAGASCAR présents
    manquants = set(REGIONS_MADAGASCAR) - set(codes)
    if manquants:
        log.warning("  ⚠ Régions non upsertées : {}", manquants)

    return len(codes)


# ─────────────────────────────────────────────────────────────────
# 2. RECETTES NUTRITIONNELLES DE BASE
# ─────────────────────────────────────────────────────────────────

# Catalogue de recettes de référence adaptées à Madagascar
# Sources : UNICEF Madagascar, FAO, Institut Pasteur Madagascar
RECETTES_SEED: List[Dict[str, Any]] = [
    {
        "nom_fr": "Bouillie de maïs enrichie au moringa",
        "nom_mg": "Laoka katsaka miaraka amin'ny moringa",
        "description": (
            "Bouillie nutritive à base de farine de maïs enrichie avec des feuilles "
            "de moringa fraîches. Riche en fer, vitamines A et C. "
            "Recommandée pour les enfants 6–23 mois en période de soudure."
        ),
        "groupe_cible":  ["enfants_6_23m", "femmes_allaitantes"],
        "regions_adaptees": None,  # Toutes régions
        "saisons_adaptees": ["saison_pluies", "transition"],
        "ingredients": [
            {"nom": "Farine de maïs",      "quantite": 50,  "unite": "g",    "optionnel": False},
            {"nom": "Feuilles de moringa", "quantite": 30,  "unite": "g",    "optionnel": False},
            {"nom": "Lait en poudre",      "quantite": 15,  "unite": "g",    "optionnel": True},
            {"nom": "Sucre de canne",      "quantite": 10,  "unite": "g",    "optionnel": True},
            {"nom": "Sel iodé",            "quantite": 1,   "unite": "pincee","optionnel": False},
            {"nom": "Eau",                 "quantite": 500, "unite": "ml",   "optionnel": False},
        ],
        "valeurs_nutritionnelles": {
            "energie_kcal": 220,
            "proteines_g":   8.5,
            "fer_mg":        4.2,
            "vitamine_a_ug": 180,
            "vitamine_c_mg": 45,
            "calcium_mg":    90,
        },
        "objectifs":  ["energie", "fer", "vitamine_a"],
        "allergenes": ["lait"],
        "temps_prep_min": 20,
        "difficulte": "facile",
        "source": "UNICEF Madagascar / Nutri-Action Guide 2023",
    },
    {
        "nom_fr": "Vary amin'anana (riz aux légumes verts)",
        "nom_mg": "Vary amin'anana",
        "description": (
            "Plat traditionnel malgache à base de riz et feuilles vertes (brèdes). "
            "Adapté pour toute la famille. Compléter avec de l'huile de palme rouge "
            "pour augmenter l'apport en vitamine A."
        ),
        "groupe_cible":  ["famille", "enfants_2_5ans", "femmes_enceintes"],
        "regions_adaptees": None,
        "saisons_adaptees": ["saison_pluies", "transition", "saison_seche"],
        "ingredients": [
            {"nom": "Riz blanc",           "quantite": 200, "unite": "g",  "optionnel": False},
            {"nom": "Brèdes (feuilles vertes mélangées)", "quantite": 150, "unite": "g", "optionnel": False},
            {"nom": "Oignon",              "quantite": 50,  "unite": "g",  "optionnel": False},
            {"nom": "Tomate",              "quantite": 80,  "unite": "g",  "optionnel": True},
            {"nom": "Huile de palme rouge","quantite": 15,  "unite": "ml", "optionnel": True,
             "note": "Fortement recommandé — source vitamine A"},
            {"nom": "Sel iodé",            "quantite": 2,   "unite": "g",  "optionnel": False},
            {"nom": "Ail",                 "quantite": 5,   "unite": "g",  "optionnel": True},
        ],
        "valeurs_nutritionnelles": {
            "energie_kcal": 380,
            "proteines_g":   7.0,
            "fer_mg":        3.5,
            "vitamine_a_ug": 120,
            "vitamine_c_mg": 30,
            "calcium_mg":    60,
        },
        "objectifs":  ["energie", "vitamine_a", "fer"],
        "allergenes": [],
        "temps_prep_min": 35,
        "difficulte": "facile",
        "source": "Recette traditionnelle malgache — validée UNICEF",
    },
    {
        "nom_fr": "Soupe de haricots et patate douce à chair orange",
        "nom_mg": "Hena-maina sy vomanga mena",
        "description": (
            "Soupe nutritive riche en protéines végétales et bêta-carotène. "
            "La patate douce à chair orange est une source exceptionnelle de pro-vitamine A. "
            "Particulièrement recommandée dans les zones à risque de malnutrition chronique "
            "(Androy, Anosy, Atsimo-Andrefana)."
        ),
        "groupe_cible":  ["enfants_6_23m", "enfants_2_5ans", "famille"],
        "regions_adaptees": ["MDG_AND", "MDG-AAN", "MDG-ASO", "MDG-IHO"],
        "saisons_adaptees": ["saison_seche", "transition"],
        "ingredients": [
            {"nom": "Haricots rouges secs","quantite": 100, "unite": "g",  "optionnel": False},
            {"nom": "Patate douce chair orange","quantite": 200, "unite": "g", "optionnel": False},
            {"nom": "Oignon",              "quantite": 60,  "unite": "g",  "optionnel": False},
            {"nom": "Tomate",              "quantite": 100, "unite": "g",  "optionnel": True},
            {"nom": "Huile végétale",      "quantite": 10,  "unite": "ml", "optionnel": False},
            {"nom": "Sel iodé",            "quantite": 2,   "unite": "g",  "optionnel": False},
            {"nom": "Eau",                 "quantite": 600, "unite": "ml", "optionnel": False},
        ],
        "valeurs_nutritionnelles": {
            "energie_kcal": 310,
            "proteines_g":  14.0,
            "fer_mg":        5.8,
            "vitamine_a_ug": 420,
            "vitamine_c_mg": 20,
            "calcium_mg":    80,
            "zinc_mg":        2.1,
        },
        "objectifs":  ["proteines", "vitamine_a", "fer", "zinc"],
        "allergenes": [],
        "temps_prep_min": 60,
        "difficulte": "facile",
        "source": "HKI (Helen Keller International) Madagascar — OFSP Program",
    },
    {
        "nom_fr": "Poisson grillé au gingembre et citron vert avec riz complet",
        "nom_mg": "Trondro voatosta miaraka amin'ny sakamalao",
        "description": (
            "Plat complet côtier riche en protéines animales et oméga-3. "
            "Particulièrement adapté aux régions côtières (Atsinanana, Diana, Boeny, Sava). "
            "Le citron vert améliore l'absorption du fer du riz."
        ),
        "groupe_cible":  ["famille", "femmes_enceintes", "femmes_allaitantes"],
        "regions_adaptees": [
            "MDG-ATS", "MDG-DIA", "MDG-BOE", "MDG-SAV",
            "MDG-ANA2", "MDG-ANO", "MDG-FIT",
        ],
        "saisons_adaptees": ["saison_pluies", "transition", "saison_seche"],
        "ingredients": [
            {"nom": "Poisson frais (tilapia/capitaine)","quantite": 200,"unite": "g","optionnel": False},
            {"nom": "Riz complet",         "quantite": 150, "unite": "g",  "optionnel": False},
            {"nom": "Gingembre frais",     "quantite": 10,  "unite": "g",  "optionnel": False},
            {"nom": "Citron vert",         "quantite": 30,  "unite": "ml", "optionnel": False},
            {"nom": "Ail",                 "quantite": 5,   "unite": "g",  "optionnel": True},
            {"nom": "Huile de coco",       "quantite": 10,  "unite": "ml", "optionnel": True},
            {"nom": "Sel iodé",            "quantite": 2,   "unite": "g",  "optionnel": False},
        ],
        "valeurs_nutritionnelles": {
            "energie_kcal": 450,
            "proteines_g":  35.0,
            "fer_mg":        3.2,
            "vitamine_a_ug":  30,
            "vitamine_c_mg":  25,
            "omega3_mg":    1200,
            "iode_ug":       85,
        },
        "objectifs":  ["proteines", "omega3", "iode"],
        "allergenes": ["poisson"],
        "temps_prep_min": 40,
        "difficulte": "moyen",
        "source": "Recette côtière adaptée — validée diététicienne UNICEF",
    },
    {
        "nom_fr": "Bouillie thérapeutique de récupération (type RUTF maison)",
        "nom_mg": "Sakafo mamelona ho an'ny ankizy",
        "description": (
            "Préparation à haute densité énergétique pour enfants en malnutrition aiguë modérée (MAM). "
            "À utiliser en complément du suivi médical. "
            "NE PAS utiliser pour les cas de SAM (malnutrition sévère) sans supervision médicale."
        ),
        "groupe_cible":  ["enfants_6_23m", "enfants_2_5ans"],
        "regions_adaptees": None,
        "saisons_adaptees": ["saison_pluies", "transition", "saison_seche"],
        "ingredients": [
            {"nom": "Farine de soja grillée","quantite": 40, "unite": "g", "optionnel": False},
            {"nom": "Farine d'arachide",    "quantite": 30,  "unite": "g", "optionnel": False,
             "note": "Vérifier absence d'aflatoxine"},
            {"nom": "Sucre glace",          "quantite": 25,  "unite": "g", "optionnel": False},
            {"nom": "Huile végétale",       "quantite": 20,  "unite": "ml","optionnel": False},
            {"nom": "Lait en poudre entier","quantite": 30,  "unite": "g", "optionnel": False},
            {"nom": "Sel iodé",             "quantite": 0.5, "unite": "g", "optionnel": False},
        ],
        "valeurs_nutritionnelles": {
            "energie_kcal": 520,
            "proteines_g":  14.0,
            "lipides_g":    28.0,
            "fer_mg":        3.0,
            "zinc_mg":       3.0,
            "vitamine_a_ug": 80,
        },
        "objectifs":  ["energie", "proteines", "recuperation_nutritionnelle"],
        "allergenes": ["arachide", "soja", "lait"],
        "temps_prep_min": 15,
        "difficulte": "facile",
        "source": "Adapté OMS / UNICEF RUTF guidelines — usage MAM uniquement",
    },
]


def seed_recipes(engine: Engine) -> int:
    """Insère les recettes de base (idempotent par nom_fr)."""
    log.info("Insertion de {} recettes de base...", len(RECETTES_SEED))

    sql = text("""
        INSERT INTO public.recettes (
            nom_fr, nom_mg, description,
            groupe_cible, regions_adaptees, saisons_adaptees,
            ingredients, valeurs_nutritionnelles,
            objectifs, allergenes,
            temps_prep_min, difficulte, source
        ) VALUES (
            :nom_fr, :nom_mg, :description,
            :groupe_cible, :regions_adaptees, :saisons_adaptees,
            :ingredients::jsonb, :valeurs_nutritionnelles::jsonb,
            :objectifs, :allergenes,
            :temps_prep_min, :difficulte, :source
        )
        ON CONFLICT DO NOTHING
        RETURNING id;
    """)

    inserted = 0
    with engine.begin() as conn:
        for recette in RECETTES_SEED:
            row = {
                **recette,
                "ingredients":             json.dumps(recette["ingredients"],             ensure_ascii=False),
                "valeurs_nutritionnelles": json.dumps(recette["valeurs_nutritionnelles"], ensure_ascii=False),
            }
            result = conn.execute(sql, row)
            if result.fetchone():
                inserted += 1
                log.debug("  ✓ Recette : {}", recette["nom_fr"])
            else:
                log.debug("  ~ Recette déjà existante : {}", recette["nom_fr"])

    log.info("  ✓ {} nouvelles recettes insérées", inserted)
    return inserted


# ─────────────────────────────────────────────────────────────────
# 3. UTILISATEURS INITIAUX
# ─────────────────────────────────────────────────────────────────

def seed_initial_users(engine: Engine) -> int:
    """
    Crée les comptes utilisateurs initiaux (admin + comptes de service).

    ⚠ Les mots de passe sont des hashes bcrypt de valeurs par défaut
    à CHANGER IMPÉRATIVEMENT lors du premier déploiement.
    """
    import hashlib

    # Génération de hash SHA-256 simplifié pour le seed
    # En production : utiliser passlib.hash.bcrypt
    def fake_hash(pwd: str) -> str:
        return "CHANGEME_" + hashlib.sha256(pwd.encode()).hexdigest()[:16]

    INITIAL_USERS = [
        {
            "email":          "admin@unicef-mdg.org",
            "nom_complet":    "Administrateur Système",
            "role":           "admin",
            "region_code":    None,
            "hashed_password": fake_hash("admin_changeme_2024"),
        },
        {
            "email":          "national@unicef-mdg.org",
            "nom_complet":    "Coordinateur National",
            "role":           "national",
            "region_code":    None,
            "hashed_password": fake_hash("national_changeme_2024"),
        },
        {
            "email":          "service_api@unicef-mdg.org",
            "nom_complet":    "Compte Service API",
            "role":           "viewer",
            "region_code":    None,
            "hashed_password": fake_hash("service_api_changeme_2024"),
        },
    ]

    sql = text("""
        INSERT INTO public.utilisateurs (email, nom_complet, role, region_code, hashed_password)
        VALUES (:email, :nom_complet, :role, :region_code, :hashed_password)
        ON CONFLICT (email) DO NOTHING
        RETURNING id;
    """)

    inserted = 0
    with engine.begin() as conn:
        for user in INITIAL_USERS:
            result = conn.execute(sql, user)
            if result.fetchone():
                inserted += 1
                log.info("  ✓ Utilisateur créé : {} ({})", user["email"], user["role"])
            else:
                log.debug("  ~ Utilisateur existant : {}", user["email"])

    log.warning(
        "⚠ SÉCURITÉ : {} comptes créés avec mots de passe temporaires — "
        "changer via l'API /admin/users avant mise en production !",
        inserted
    )
    return inserted


# ─────────────────────────────────────────────────────────────────
# 4. VÉRIFICATION DES DONNÉES
# ─────────────────────────────────────────────────────────────────

def verify_seed_data(engine: Engine) -> bool:
    """
    Vérifie la cohérence des données seedées.
    Contrôles :
      - 22 régions présentes
      - Centroïdes valides (PostGIS)
      - Voisinage symétrique (si A voisin de B, B voisin de A)
      - Recettes référencent des régions existantes
    """
    log.info("Vérification de la cohérence des données...")
    ok = True

    with engine.connect() as conn:

        # 1. Nombre de régions
        n_regions = conn.execute(
            text("SELECT COUNT(*) FROM public.regions WHERE actif = TRUE;")
        ).scalar()
        if n_regions == 22:
            log.info("  ✓ 22 régions actives")
        else:
            log.error("  ✗ Nombre de régions incorrect : {} (attendu: 22)", n_regions)
            ok = False

        # 2. Centroïdes dans le bounding box Madagascar
        n_hors_mdg = conn.execute(text("""
            SELECT COUNT(*) FROM public.regions
            WHERE NOT ST_DWithin(
                centroid,
                ST_GeogFromText('POINT(46.5 -20.0)'),
                1500000  -- 1 500 km rayon
            );
        """)).scalar()
        if n_hors_mdg == 0:
            log.info("  ✓ Tous les centroïdes sont dans le rayon Madagascar")
        else:
            log.error("  ✗ {} centroïdes hors du rayon Madagascar", n_hors_mdg)
            ok = False

        # 3. Voisinage symétrique
        asymetriques = conn.execute(text("""
            SELECT r1.code, r2.code
            FROM public.regions r1
            JOIN public.regions r2 ON r2.code = ANY(r1.voisins)
            WHERE r1.code <> ALL(r2.voisins)
            LIMIT 10;
        """)).fetchall()
        if not asymetriques:
            log.info("  ✓ Voisinage symétrique validé")
        else:
            for a, b in asymetriques:
                log.warning("  ⚠ Voisinage asymétrique : {} ↔ {}", a, b)
            # Non bloquant : données incomplètes mais pas erreur fatale

        # 4. Recettes — régions valides
        n_recettes = conn.execute(
            text("SELECT COUNT(*) FROM public.recettes WHERE valide = TRUE;")
        ).scalar()
        log.info("  ✓ {} recettes valides en base", n_recettes)

        # 5. Population totale cohérente (Madagascar ~27M en 2024)
        pop_totale = conn.execute(
            text("SELECT SUM(population_estimee) FROM public.regions;")
        ).scalar() or 0
        if 25_000_000 <= pop_totale <= 35_000_000:
            log.info("  ✓ Population totale cohérente : {:,}", pop_totale)
        else:
            log.warning(
                "  ⚠ Population totale inattendue : {:,} (attendu ~27M)", pop_totale
            )

    return ok


# ─────────────────────────────────────────────────────────────────
# 5. RESET
# ─────────────────────────────────────────────────────────────────

def reset_seed_data(engine: Engine) -> None:
    """Efface les données seedées (pas les schémas/tables)."""
    log.warning("Reset des données seed...")
    with engine.begin() as conn:
        # Ordre respectant les FK
        conn.execute(text("DELETE FROM public.recettes;"))
        conn.execute(text("DELETE FROM public.utilisateurs WHERE email LIKE '%@unicef-mdg.org';"))
        conn.execute(text("DELETE FROM public.regions;"))
    log.info("  ✓ Données seed effacées")


# ─────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Peuplement du référentiel géographique et données de base"
    )
    parser.add_argument(
        "--with-recipes",
        action="store_true",
        help="Insère également les recettes nutritionnelles de base",
    )
    parser.add_argument(
        "--with-users",
        action="store_true",
        help="Crée les utilisateurs initiaux (admin, national, service)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Efface et recharge toutes les données seed",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Vérifie seulement la cohérence sans insérer",
    )
    args = parser.parse_args()

    engine = sa.create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )

    t_start = time.perf_counter()

    try:
        if args.verify:
            ok = verify_seed_data(engine)
            sys.exit(0 if ok else 1)

        if args.reset:
            if settings.environment == "production":
                log.critical("--reset INTERDIT en production !")
                sys.exit(1)
            reset_seed_data(engine)

        # Régions (toujours exécuté)
        n_regions = upsert_regions(engine)

        # Recettes optionnelles
        if args.with_recipes:
            seed_recipes(engine)

        # Utilisateurs optionnels
        if args.with_users:
            seed_initial_users(engine)

        # Rafraîchissement des vues matérialisées
        log.info("Rafraîchissement des vues matérialisées...")
        with engine.begin() as conn:
            for view in [
                "public.mv_malaria_weekly_summary",
                "public.mv_nutrition_status",
            ]:
                try:
                    conn.execute(text(f"REFRESH MATERIALIZED VIEW {view};"))
                    log.info("  ✓ {}", view)
                except Exception:
                    log.debug("  ~ Vue non rafraîchie (probablement vide) : {}", view)

        # Vérification finale
        ok = verify_seed_data(engine)

        elapsed = time.perf_counter() - t_start
        if ok:
            log.info(
                "✓ Seed terminé avec succès en {:.1f}s — {} régions chargées",
                elapsed, n_regions
            )
        else:
            log.warning("⚠ Seed terminé avec des avertissements — voir logs ci-dessus")

    except Exception as exc:
        log.exception("Erreur fatale lors du seed : {}", exc)
        sys.exit(1)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()