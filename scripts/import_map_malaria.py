"""
Import des données Malaria Atlas Project (MAP) — risque paludisme annuel
par région — dans malaria_risk_annual.

  Source : Malaria Atlas Project, export "Subnational Unit data"
    https://malariaatlas.org/ (Admin1, Madagascar, 2000-2024)
    Colonnes attendues : ISO3, National Unit, Name, Admin Level, Metric,
    Units, Year, Value
    Metric ∈ {"Incidence Rate", "Mortality Rate", "Infection Prevalence"}

  Comme pour import_mics6_nutrition.py : PAS de correspondance nom→code
  devinée à l'aveugle. Le script interroge directement ta table
  `regions` (colonne nom_fr) et fait un matching automatique normalisé
  (minuscule, sans accents/apostrophes/espaces). Tout nom non reconnu
  est listé clairement à la fin, rien n'est inséré à l'aveugle pour ces
  cas-là.

Usage :
    python scripts/import_map_malaria.py --csv data/raw/Subnational_Unit-data.csv --dry-run
    python scripts/import_map_malaria.py --csv data/raw/Subnational_Unit-data.csv --verify
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.settings import settings
from src.utils.logger import get_logger, setup_logging

setup_logging()
log = get_logger("import_map_malaria")

SOURCE_LABEL = "malaria_atlas_project"

METRIC_TO_COLUMN = {
    "Incidence Rate":       "incidence_pour_mille",
    "Mortality Rate":       "mortalite_pour_100k",
    "Infection Prevalence": "prevalence_pct",
}

# Cas où un même nom du CSV doit s'appliquer à plusieurs codes région de
# ta base (région fusionnée dans le CSV mais scindée chez toi) — même cas
# que rencontré avec le MICS6.
SPLIT_REGIONS: Dict[str, List[str]] = {
    "Vatovavy Fitovinany": ["Vatovavy", "Fitovinany"],
}

# Alias manuels confirmés lors de l'import MICS6 (nom_fr réel en base
# différent du nom "officiel" utilisé dans ce CSV).
ALIASES_MANUELS: Dict[str, str] = {
    "Haute Matsiatra": "MDG-MAT",  # nom_fr en base : "Matsiatra Ambony"
    # "Betsiboka" : PAS de correspondance — cette région n'existe pas dans
    # ta table `regions` (confirmé lors de l'import MICS6). Les lignes
    # Betsiboka seront ignorées et listées en non-matchées.
}


def normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode("ascii")
    texte = texte.lower()
    for car in ["'", "’", "-", " ", "_"]:
        texte = texte.replace(car, "")
    return texte


def detecter_colonne_nom(engine: Engine) -> str:
    candidats = ["nom_fr", "nom", "name", "region_name", "libelle"]
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
        f"Colonnes disponibles : {colonnes_dispo}"
    )


def charger_regions(engine: Engine) -> Dict[str, str]:
    col_nom = detecter_colonne_nom(engine)
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT code, {col_nom} AS nom FROM public.regions")).fetchall()
    return {normaliser(r.nom): r.code for r in rows}


def lire_csv_map(csv_path: Path, regions_db: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Lit le CSV MAP, pivote les 3 métriques par (région, année), et
    convertit les noms de région en codes réels de la base.
    Retourne (records, noms_non_matches).
    """
    par_region_annee: Dict[Tuple[str, int], Dict[str, float]] = {}
    non_matches: set = set()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Admin Level", "").strip() != "admin1":
                continue

            metric = (row.get("Metric") or "").strip()
            colonne = METRIC_TO_COLUMN.get(metric)
            if colonne is None:
                continue  # métrique non pertinente pour ce modèle (on ignore silencieusement)

            nom_csv = (row.get("Name") or "").strip()

            # Résout vers un ou plusieurs codes région cibles
            if nom_csv in SPLIT_REGIONS:
                codes_cibles = []
                for sous_nom in SPLIT_REGIONS[nom_csv]:
                    c = regions_db.get(normaliser(sous_nom))
                    if c:
                        codes_cibles.append(c)
                    else:
                        non_matches.add(f"{nom_csv} → {sous_nom}")
            else:
                c = ALIASES_MANUELS.get(nom_csv) or regions_db.get(normaliser(nom_csv))
                if c:
                    codes_cibles = [c]
                else:
                    codes_cibles = []
                    non_matches.add(nom_csv)

            if not codes_cibles:
                continue

            try:
                annee = int(row.get("Year", "").strip())
                valeur = float(row.get("Value", "").strip())
            except ValueError:
                continue

            for code in codes_cibles:
                key = (code, annee)
                par_region_annee.setdefault(key, {})[colonne] = valeur

    records = [
        {
            "region_code": code,
            "annee": annee,
            "incidence_pour_mille": valeurs.get("incidence_pour_mille"),
            "mortalite_pour_100k":  valeurs.get("mortalite_pour_100k"),
            "prevalence_pct":       valeurs.get("prevalence_pct"),
            "source": SOURCE_LABEL,
        }
        for (code, annee), valeurs in sorted(par_region_annee.items())
    ]

    return records, sorted(non_matches)


def inserer(engine: Engine, records: List[Dict[str, Any]]) -> Tuple[int, int]:
    sql = text("""
        INSERT INTO public.malaria_risk_annual (
            region_code, annee, incidence_pour_mille,
            mortalite_pour_100k, prevalence_pct, source
        ) VALUES (
            :region_code, :annee, :incidence_pour_mille,
            :mortalite_pour_100k, :prevalence_pct, :source
        )
        ON CONFLICT (region_code, annee, source) DO UPDATE SET
            incidence_pour_mille = EXCLUDED.incidence_pour_mille,
            mortalite_pour_100k  = EXCLUDED.mortalite_pour_100k,
            prevalence_pct       = EXCLUDED.prevalence_pct;
    """)
    with engine.begin() as conn:
        result = conn.execute(sql, records)
        # rowcount avec ON CONFLICT DO UPDATE compte aussi les mises à jour
        return result.rowcount, 0


def verifier(engine: Engine) -> None:
    sql = text("""
        SELECT count(*) AS n, count(DISTINCT region_code) AS n_regions,
               min(annee) AS annee_min, max(annee) AS annee_max
        FROM public.malaria_risk_annual
        WHERE source = :source
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"source": SOURCE_LABEL}).fetchone()
    if not row or row.n == 0:
        log.warning("Vérification : aucune ligne source='{}' en base.", SOURCE_LABEL)
        return
    log.info(
        "Vérification — {} lignes, {} régions, années {}-{}",
        row.n, row.n_regions, row.annee_min, row.annee_max
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Malaria Atlas Project → malaria_risk_annual")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    engine = sa.create_engine(settings.database.sync_url, pool_pre_ping=True)
    try:
        regions_db = charger_regions(engine)
        log.info("{} régions trouvées dans ta table `regions`", len(regions_db))

        records, non_matches = lire_csv_map(args.csv, regions_db)
        log.info("{} lignes (région × année) construites depuis {}", len(records), args.csv.name)

        if non_matches:
            log.warning(
                "{} nom(s) de région NON reconnus — lignes ignorées : {}",
                len(non_matches), non_matches
            )

        if args.dry_run:
            log.info("DRY-RUN : {} lignes seraient insérées/mises à jour, rien écrit en base.", len(records))
            for r in records[:10]:
                log.info("  {}", r)
            return

        if records:
            n, _ = inserer(engine, records)
            log.info("Import terminé : {} lignes insérées/mises à jour", n)

        if args.verify:
            verifier(engine)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()