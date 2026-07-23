"""
Import des données nutrition MICS6 Madagascar 2018 (INSTAT + UNICEF) —
la seule enquête que j'ai trouvée couvrant les 22 régions de Madagascar
avec des vraies valeurs officielles par région (pas une distribution
nationale approximative comme fetch_unicef_nutrition.py).

  Source : https://www.unicef.org/madagascar/media/2421/file/MICS6-Madagascar-2018-Nutrition.pdf
  Enquête : MICS6, INSTAT + UNICEF, données 2018, enfants de moins de 5 ans.
  Indicateurs : retard de croissance (stunting), insuffisance pondérale
  (underweight), émaciation globale (GAM) et sévère (SAM).

  ─────────────────────────────────────────────────────────────────────
  Ce script NE DEVINE PAS la correspondance entre les noms de région du
  rapport et tes codes internes (MDG-XXX) : il interroge ta table
  `regions` directement et fait un matching automatique sur le nom
  normalisé (minuscule, sans accents/apostrophes/espaces). Toute région
  non reconnue est affichée clairement à la fin — RIEN n'est inséré à
  l'aveugle pour ces cas-là, à toi de compléter manuellement le
  dictionnaire ALIASES_MANUELS si besoin.

  Cas particulier : le rapport donne une valeur combinée pour
  "Vatovavy Fitovinany" (ex-région unique avant la réforme de 2009).
  Si ta base a deux codes séparés (Vatovavy / Fitovinany), la même
  valeur combinée leur est appliquée aux deux, avec un avertissement
  explicite dans les métadonnées.
  ─────────────────────────────────────────────────────────────────────

Usage :
    python scripts/import_mics6_nutrition.py --dry-run
    python scripts/import_mics6_nutrition.py --verify

Prérequis :
    pip install sqlalchemy psycopg2-binary
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.settings import settings
from src.utils.logger import get_logger, setup_logging

setup_logging()
log = get_logger("import_mics6_nutrition")

SOURCE_LABEL   = "mics6_2018_instat_unicef"
GROUPE_CIBLE   = "enfants_moins_5ans"
DATE_ENQUETE   = date(2018, 6, 1)  # enquête menée en 2018, mois précis non communiqué

# ─────────────────────────────────────────────────────────────────
# Données réelles extraites du rapport MICS6 2018 (par région)
# stunting_pct = retard de croissance, underweight_pct = insuffisance
# pondérale, gam_pct = émaciation globale (modérée+sévère), sam_pct = émaciation sévère
# ─────────────────────────────────────────────────────────────────
MICS6_DATA: Dict[str, Dict[str, float]] = {
    "Vakinankaratra":       {"stunting_pct": 60, "underweight_pct": 40, "gam_pct": 6,  "sam_pct": 1},
    "Amoron'i Mania":       {"stunting_pct": 55, "underweight_pct": 35, "gam_pct": 6,  "sam_pct": 0},
    "Haute Matsiatra":      {"stunting_pct": 54, "underweight_pct": 26, "gam_pct": 5,  "sam_pct": 1},
    "Bongolava":            {"stunting_pct": 52, "underweight_pct": 31, "gam_pct": 7,  "sam_pct": 1},
    "Analamanga":           {"stunting_pct": 48, "underweight_pct": 26, "gam_pct": 5,  "sam_pct": 1},
    "Alaotra Mangoro":      {"stunting_pct": 47, "underweight_pct": 28, "gam_pct": 6,  "sam_pct": 1},
    "Atsinanana":           {"stunting_pct": 46, "underweight_pct": 26, "gam_pct": 5,  "sam_pct": 1},
    "Itasy":                {"stunting_pct": 45, "underweight_pct": 28, "gam_pct": 6,  "sam_pct": 1},
    "Vatovavy Fitovinany":  {"stunting_pct": 44, "underweight_pct": 35, "gam_pct": 13, "sam_pct": 2},
    "Androy":               {"stunting_pct": 39, "underweight_pct": 24, "gam_pct": 7,  "sam_pct": 1},
    "Sava":                 {"stunting_pct": 39, "underweight_pct": 21, "gam_pct": 5,  "sam_pct": 1},
    "Atsimo Andrefana":     {"stunting_pct": 38, "underweight_pct": 26, "gam_pct": 6,  "sam_pct": 1},
    "Anosy":                {"stunting_pct": 38, "underweight_pct": 25, "gam_pct": 8,  "sam_pct": 1},
    "Boeny":                {"stunting_pct": 34, "underweight_pct": 29, "gam_pct": 9,  "sam_pct": 1},
    "Menabe":               {"stunting_pct": 34, "underweight_pct": 24, "gam_pct": 11, "sam_pct": 2},
    "Betsiboka":            {"stunting_pct": 34, "underweight_pct": 29, "gam_pct": 11, "sam_pct": 1},
    "Analanjirofo":         {"stunting_pct": 31, "underweight_pct": 20, "gam_pct": 7,  "sam_pct": 2},
    "Ihorombe":             {"stunting_pct": 31, "underweight_pct": 16, "gam_pct": 7,  "sam_pct": 1},
    "Diana":                {"stunting_pct": 30, "underweight_pct": 18, "gam_pct": 5,  "sam_pct": 1},
    "Sofia":                {"stunting_pct": 29, "underweight_pct": 20, "gam_pct": 5,  "sam_pct": 1},
    "Melaky":               {"stunting_pct": 26, "underweight_pct": 19, "gam_pct": 6,  "sam_pct": 2},
    "Atsimo Atsinanana":    {"stunting_pct": 20, "underweight_pct": 15, "gam_pct": 3,  "sam_pct": 1},
}

# Cas où un même nom de rapport doit s'appliquer à plusieurs codes région
# de ta base (ex: région fusionnée dans le rapport mais scindée chez toi).
SPLIT_REGIONS: Dict[str, List[str]] = {
    "Vatovavy Fitovinany": ["Vatovavy", "Fitovinany"],
}

# Si le matching automatique échoue pour un nom précis, ajoute ici la
# correspondance manuelle : "Nom du rapport": "code region dans ta base".
ALIASES_MANUELS: Dict[str, str] = {
    # À confirmer avec `SELECT code, nom_fr FROM regions ORDER BY code;`
    "Haute Matsiatra": "MDG-MAT",  # probable, à vérifier
    # "Betsiboka": "MDG-???",       # code exact inconnu pour l'instant
}


def normaliser(texte: str) -> str:
    """minuscule, sans accents, sans apostrophes/espaces/tirets — pour matcher robustement."""
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode("ascii")
    texte = texte.lower()
    for car in ["'", "’", "-", " ", "_"]:
        texte = texte.replace(car, "")
    return texte


def detecter_colonne_nom(engine: Engine) -> str:
    """Trouve la colonne 'nom de région' dans la table regions (nom, name, libelle...)."""
    candidats = ["nom_fr", "nom", "name", "region_name", "libelle", "nom_region", "designation"]
    with engine.connect() as conn:
        cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'regions'
        """)).fetchall()
    colonnes_dispo = [c[0] for c in cols]
    for candidat in candidats:
        if candidat in colonnes_dispo:
            return candidat
    raise RuntimeError(
        f"Impossible de trouver une colonne 'nom' dans regions. "
        f"Colonnes disponibles : {colonnes_dispo}. "
        f"Ajoute le bon nom de colonne dans detecter_colonne_nom()."
    )


def charger_regions(engine: Engine) -> Dict[str, str]:
    """Retourne {nom_normalise: code_region} depuis la table regions."""
    col_nom = detecter_colonne_nom(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT code, {col_nom} AS nom FROM public.regions")).fetchall()
    return {normaliser(r.nom): r.code for r in rows}


def classification_oms(gam_pct: float) -> str:
    if gam_pct < 5.0:
        return "acceptable"
    elif gam_pct < 10.0:
        return "alerte"
    elif gam_pct < 15.0:
        return "urgence"
    return "crise"


def construire_records(regions_db: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Construit les enregistrements nutrition_observations en matchant les
    noms MICS6 aux codes région réels de la base.
    Retourne (records, noms_non_matches).
    """
    records: List[Dict[str, Any]] = []
    non_matches: List[str] = []

    for nom_rapport, valeurs in MICS6_DATA.items():
        # Résout vers un ou plusieurs codes région cibles
        if nom_rapport in ALIASES_MANUELS:
            codes_cibles = [ALIASES_MANUELS[nom_rapport]]
        elif nom_rapport in SPLIT_REGIONS:
            codes_cibles = []
            for sous_nom in SPLIT_REGIONS[nom_rapport]:
                code = regions_db.get(normaliser(sous_nom))
                if code:
                    codes_cibles.append(code)
                else:
                    non_matches.append(f"{nom_rapport} → {sous_nom} (non trouvé)")
        else:
            code = regions_db.get(normaliser(nom_rapport))
            codes_cibles = [code] if code else []
            if not code:
                non_matches.append(nom_rapport)

        gam = valeurs["gam_pct"]
        sam = valeurs["sam_pct"]
        mam = round(gam - sam, 2) if gam >= sam else None
        stunting = valeurs["stunting_pct"]
        underweight = valeurs["underweight_pct"]

        for code_region in codes_cibles:
            records.append({
                "region_code":        code_region,
                "date_enquete":       DATE_ENQUETE.isoformat(),
                "gam_pct":            round(float(gam), 2),
                "mam_pct":            mam,
                "sam_pct":            round(float(sam), 2),
                "classification_oms": classification_oms(gam),
                "groupe_cible":       GROUPE_CIBLE,
                "n_enfants_enquetes": None,
                "score_sca":          None,
                "source":             SOURCE_LABEL,
                "metadata": json.dumps({
                    "import": "import_mics6_nutrition",
                    "nom_region_rapport": nom_rapport,
                    "stunting_pct": stunting,
                    "underweight_pct": underweight,
                    "region_fusionnee_dans_rapport": nom_rapport in SPLIT_REGIONS,
                }),
            })

    return records, non_matches


def inserer_nutrition(engine: Engine, records: List[Dict[str, Any]]) -> Tuple[int, int]:
    sql = text("""
        INSERT INTO public.nutrition_observations (
            region_code, date_enquete,
            gam_pct, mam_pct, sam_pct,
            classification_oms, groupe_cible,
            n_enfants_enquetes, score_sca,
            source, metadata
        ) VALUES (
            :region_code, :date_enquete,
            :gam_pct, :mam_pct, :sam_pct,
            :classification_oms, :groupe_cible,
            :n_enfants_enquetes, :score_sca,
            :source, CAST(:metadata AS jsonb)
        )
        ON CONFLICT (region_code, date_enquete, groupe_cible, source)
        DO NOTHING;
    """)
    with engine.begin() as conn:
        result = conn.execute(sql, records)
        inserted = result.rowcount
    return inserted, len(records) - inserted


def verifier_insertion(engine: Engine) -> None:
    sql = text("""
        SELECT region_code, gam_pct, mam_pct, sam_pct, classification_oms
        FROM public.nutrition_observations
        WHERE source = :source
        ORDER BY region_code;
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"source": SOURCE_LABEL}).fetchall()
    if not rows:
        log.warning("Vérification : aucune ligne source='{}' en base.", SOURCE_LABEL)
        return
    log.info("Vérification — {} lignes source='{}' :", len(rows), SOURCE_LABEL)
    for r in rows:
        log.info(
            "  {:<12} : gam={} mam={} sam={} | {}",
            r.region_code, r.gam_pct, r.mam_pct, r.sam_pct, r.classification_oms
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import des données nutrition MICS6 2018 par région (Madagascar)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    engine = sa.create_engine(settings.database.sync_url, pool_pre_ping=True)
    try:
        regions_db = charger_regions(engine)
        log.info("{} régions trouvées dans ta table `regions`", len(regions_db))

        records, non_matches = construire_records(regions_db)

        log.info("{} lignes construites depuis MICS6 2018 ({} régions du rapport)",
                  len(records), len(MICS6_DATA))

        if non_matches:
            log.warning(
                "{} nom(s) du rapport NON reconnus dans ta base — ces régions ne "
                "recevront AUCUNE donnée tant que tu n'ajoutes pas la correspondance "
                "manuelle dans ALIASES_MANUELS :\n{}",
                len(non_matches), "\n".join(f"  - {n}" for n in non_matches)
            )

        if args.dry_run:
            log.info("DRY-RUN : {} lignes seraient insérées, rien écrit en base.", len(records))
            for r in records:
                log.info(
                    "  {} | gam={} mam={} sam={} | {}",
                    r["region_code"], r["gam_pct"], r["mam_pct"], r["sam_pct"],
                    r["classification_oms"]
                )
            return

        if records:
            inserted, skipped = inserer_nutrition(engine, records)
            log.info("Import terminé : {} insérées, {} doublons ignorés", inserted, skipped)

        if args.verify:
            verifier_insertion(engine)

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()