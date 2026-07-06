"""
Récupération des indicateurs nutrition UNICEF (Data Warehouse SDMX, API publique,
pas de clé requise) pour Madagascar, et import dans nutrition_observations.

Script autonome, indépendant de backfill_historical.py et de import_nutrition_csv.py.

  Source : UNICEF Indicator Data Warehouse
    Endpoint : https://sdmx.data.unicef.org/ws/public/sdmxapi/rest/
    Dataflow : UNICEF,GLOBAL_DATAFLOW,1.0
    Doc      : https://data.unicef.org/sdmx-api-documentation/

  Indicateurs récupérés (estimations UNICEF/OMS/Banque Mondiale — Joint
  Malnutrition Estimates, agrégées depuis les enquêtes ENSOMD/DHS/MICS) :
    - NT_ANT_WHZ_NE2 : émaciation modérée+sévère (poids/taille <-2 ET), = GAM
    - NT_ANT_WHZ_NE3 : émaciation sévère (poids/taille <-3 ET), = SAM
    - NT_ANT_HAZ_NE2 : retard de croissance (taille/âge <-2 ET), = stunting

  ─────────────────────────────────────────────────────────────────────
  LIMITE IMPORTANTE, À LIRE AVANT D'UTILISER CE SCRIPT :

  Ces données sont disponibles UNIQUEMENT au niveau national (une valeur
  par an pour tout Madagascar). Il n'existe aucune API publique donnant
  ces indicateurs par région administrative pour Madagascar — les vraies
  données régionales existent dans les rapports d'enquêtes SMART/ENSOMD
  de l'ONN/INSTAT, publiés en PDF/Excel, pas via API.

  Ce script applique donc la valeur nationale identiquement aux 22 régions,
  avec source = "unicef_national_distribue" et groupe_cible = "estimation_nationale".
  CE N'EST PAS UNE DONNÉE RÉGIONALE RÉELLE — c'est un fond de carte national
  à utiliser en dernier recours (ex: valeur par défaut pour une région sans
  enquête SMART locale), jamais à présenter comme une mesure région par région.
  Pour de vraies données régionales, utilise import_nutrition_csv.py avec les
  chiffres extraits des rapports SMART/ENSOMD.
  ─────────────────────────────────────────────────────────────────────

Usage :
    python scripts/fetch_unicef_nutrition.py
    python scripts/fetch_unicef_nutrition.py --start-year 2015 --end-year 2026
    python scripts/fetch_unicef_nutrition.py --dry-run
    python scripts/fetch_unicef_nutrition.py --verify

Prérequis :
    pip install httpx sqlalchemy psycopg2-binary
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.settings import settings
from src.utils.constants import REGIONS_MADAGASCAR
from src.utils.logger import get_logger, setup_logging

setup_logging()
log = get_logger("fetch_unicef_nutrition")

UNICEF_SDMX_BASE = "https://sdmx.data.unicef.org/ws/public/sdmxapi/rest"
UNICEF_DATAFLOW  = "UNICEF,NUTRITION,1.0"
UNICEF_COUNTRY   = "MDG"  # code ISO3 Madagascar

# Indicateurs : GAM (wasting), SAM (wasting sévère), stunting
INDICATOR_GAM      = "NT_ANT_WHZ_NE2"
INDICATOR_SAM      = "NT_ANT_WHZ_NE3"
INDICATOR_STUNTING = "NT_ANT_HAZ_NE2"

SOURCE_LABEL   = "unicef_national_distribue"
GROUPE_NATIONAL = "estimation_nationale"


@dataclass
class FetchStats:
    n_annees_recuperees: int = 0
    n_records_generes:   int = 0
    n_inserees:          int = 0
    n_doublons:          int = 0
    n_erreurs:           int = 0
    erreurs_detail:      List[str] = field(default_factory=list)


def _dq_key_nutrition() -> str:
    """
    Clé SDMX pour le dataflow UNICEF,NUTRITION,1.0 (8 dimensions), confirmée
    par l'URL réelle du databrowser de l'utilisateur :
        q=UNICEF:NUTRITION(1.0);MDG.NT_ANT_WHZ_NE2......
    → REF_AREA=MDG . INDICATOR . (6 dimensions restantes en wildcard) :
      SEX, AGE, WEALTH_QUINTILE, RESIDENCE, MATERNAL_EDU_LVL, HEAD_OF_HOUSE
    On filtre SEX=_T et AGE=Y0T4 après coup dans le CSV plutôt que dans la
    clé (une valeur de dimension explicite mais mal orthographiée renvoyait
    une 404, le wildcard est plus sûr).
    """
    indicators = "+".join([INDICATOR_GAM, INDICATOR_SAM, INDICATOR_STUNTING])
    return f"{UNICEF_COUNTRY}.{indicators}......"


def _dq_key_global() -> str:
    """
    Clé SDMX pour le dataflow UNICEF,GLOBAL_DATAFLOW,1.0 (fallback), calquée
    sur un exemple confirmé fonctionnel : REF_AREA.INDICATOR.SEX.AGE(vide)
    — l'âge est laissé en wildcard plutôt que fixé à un code qui pourrait
    ne pas exister dans ce dataflow ; on filtrera Y0T4 côté client au parsing.
    """
    indicators = "+".join([INDICATOR_GAM, INDICATOR_SAM, INDICATOR_STUNTING])
    return f"{UNICEF_COUNTRY}.{indicators}._T."


# Candidats essayés dans l'ordre — le premier qui répond 200 est utilisé.
_DATAFLOW_CANDIDATES = [
    ("UNICEF,NUTRITION,1.0", _dq_key_nutrition),
    ("UNICEF,GLOBAL_DATAFLOW,1.0", _dq_key_global),
]


def fetch_unicef_csv(timeout: float = 30.0) -> str:
    """
    Appelle l'API SDMX UNICEF et retourne le CSV brut (texte).
    Essaie plusieurs dataflows/clés à la suite (la structure exacte des
    dimensions peut varier selon le dataflow) ; s'arrête au premier succès.

    NB : pas de segment "/all" (providerRef) dans l'URL — les exemples
    UNICEF confirmés vont directement de la clé à la query string.
    """
    dernieres_erreurs = []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for dataflow, dq_builder in _DATAFLOW_CANDIDATES:
            url = f"{UNICEF_SDMX_BASE}/data/{dataflow}/{dq_builder()}"
            params = {"format": "csv"}
            log.info("Requête UNICEF SDMX : {} params={}", url, params)
            try:
                response = client.get(url, params=params)
                response.raise_for_status()
                log.info("Succès avec dataflow={}", dataflow)
                return response.text
            except httpx.HTTPStatusError as exc:
                log.warning(
                    "Échec dataflow={} : HTTP {} — tentative suivante",
                    dataflow, exc.response.status_code
                )
                dernieres_erreurs.append(str(exc))
                continue

    raise RuntimeError(
        "Tous les dataflows UNICEF testés ont échoué : " + " | ".join(dernieres_erreurs)
    )


def parse_unicef_csv(raw_csv: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse le CSV UNICEF vers {periode: {"annee":.., "date_obs":.., "gam_pct":.., ...}}.
    `periode` est la valeur brute de TIME_PERIOD (ex: "2011" ou "2004-02") —
    utilisée comme clé pour ne pas fusionner deux enquêtes de la même année
    (ex: 2004-02 et 2004-06 sont deux enquêtes distinctes la même année).
    Ne garde que REF_AREA == MDG et les lignes agrégées (Total) sur toutes
    les dimensions de désagrégation disponibles dans ce dataset.
    """
    reader = csv.DictReader(io.StringIO(raw_csv))
    if reader.fieldnames is None:
        log.error("Réponse UNICEF vide ou illisible (pas d'en-tête CSV)")
        return {}

    par_periode: Dict[str, Dict[str, Any]] = {}

    for row in reader:
        if row.get("REF_AREA", "").strip().upper() != UNICEF_COUNTRY:
            continue

        # SEX et AGE sont laissés en wildcard dans la requête (voir
        # _dq_key_nutrition) : on ne garde ici que le total (_T) pour le
        # sexe ET pour l'âge (ce dataset découpe l'âge en tranches
        # mensuelles fines — M0T11, M12T23, etc. — "_T" est le seul code
        # qui correspond à l'agrégat "0-5 ans" attendu par
        # nutrition_observations ; confirmé depuis un vrai export CSV).
        sex = (row.get("SEX") or "").strip()
        if sex and sex != "_T":
            continue

        age = (row.get("AGE") or "").strip()
        if age and age != "_T":
            continue

        # Idem pour les autres dimensions de désagrégation présentes dans
        # ce dataset (quintile de richesse, milieu urbain/rural, niveau
        # d'éducation de la mère) : sans ce filtre, les valeurs par
        # sous-groupe (Q1..Q5, urbain/rural, etc.) se mélangeraient avec
        # la vraie valeur nationale et corromperaient l'agrégat par année.
        wealth = (row.get("WEALTH_QUINTILE") or "").strip()
        if wealth and wealth != "_T":
            continue

        residence = (row.get("RESIDENCE") or "").strip()
        if residence and residence != "_T":
            continue

        edu = (row.get("MATERNAL_EDU_LVL") or "").strip()
        if edu and edu != "_T":
            continue

        hoh = (row.get("HEAD_OF_HOUSE") or "").strip()
        if hoh and hoh != "_T":
            continue

        indicator = (row.get("INDICATOR") or "").strip()
        time_period = (row.get("TIME_PERIOD") or "").strip()
        obs_value = (row.get("OBS_VALUE") or "").strip()

        if not time_period or not obs_value:
            continue

        # TIME_PERIOD peut être une année pleine ("2011") ou une période
        # mensuelle ("2004-02") quand plusieurs enquêtes existent la même
        # année — on récupère l'année dans les deux cas, mais on garde la
        # période brute comme clé pour ne pas fusionner deux enquêtes
        # distinctes de la même année.
        try:
            annee = int(time_period.split("-")[0])
        except ValueError:
            continue
        try:
            value = float(obs_value)
        except ValueError:
            continue

        if len(time_period) >= 7:  # "YYYY-MM"
            try:
                mois = int(time_period.split("-")[1])
            except (ValueError, IndexError):
                mois = 7
            date_obs = date(annee, mois, 1)
        else:
            date_obs = date(annee, 7, 1)

        par_periode.setdefault(time_period, {"annee": annee, "date_obs": date_obs})
        if indicator == INDICATOR_GAM:
            par_periode[time_period]["gam_pct"] = value
        elif indicator == INDICATOR_SAM:
            par_periode[time_period]["sam_pct"] = value
        elif indicator == INDICATOR_STUNTING:
            par_periode[time_period]["stunting_pct"] = value

    return par_periode


def classification_oms(gam_pct: float) -> str:
    if gam_pct < 5.0:
        return "acceptable"
    elif gam_pct < 10.0:
        return "alerte"
    elif gam_pct < 15.0:
        return "urgence"
    return "crise"


def build_records(
    par_periode: Dict[str, Dict[str, Any]],
    start_year: int,
    end_year: int,
) -> List[Dict[str, Any]]:
    """
    Construit un enregistrement nutrition_observations par (région, période),
    en dupliquant la valeur nationale sur les 22 régions.
    """
    records: List[Dict[str, Any]] = []

    for periode, valeurs in sorted(par_periode.items()):
        annee = valeurs["annee"]
        if not (start_year <= annee <= end_year):
            continue

        gam = valeurs.get("gam_pct")
        if gam is None:
            log.warning("{} : pas de valeur GAM (wasting), ligne ignorée", periode)
            continue

        sam = valeurs.get("sam_pct")
        mam = round(gam - sam, 2) if (sam is not None and gam >= sam) else None
        stunting = valeurs.get("stunting_pct")
        date_obs: date = valeurs["date_obs"]

        for region in sorted(REGIONS_MADAGASCAR):
            records.append({
                "region_code":        region,
                "date_enquete":       date_obs.isoformat(),
                "gam_pct":            round(gam, 2),
                "mam_pct":            mam,
                "sam_pct":            round(sam, 2) if sam is not None else None,
                "classification_oms": classification_oms(gam),
                "groupe_cible":       GROUPE_NATIONAL,
                "n_enfants_enquetes": None,
                "score_sca":          None,
                "source":             SOURCE_LABEL,
                "metadata": (
                    '{"import": "fetch_unicef_nutrition", '
                    f'"periode_source": "{periode}", '
                    f'"stunting_pct_national": {stunting if stunting is not None else "null"}, '
                    '"avertissement": "valeur nationale dupliquee sur toutes les regions, pas une mesure regionale reelle"}'
                ),
            })

    return records


def inserer_nutrition(engine: Engine, records: List[Dict[str, Any]], batch_size: int = 500):
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
    inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        with engine.begin() as conn:
            result = conn.execute(sql, batch)
            inserted += result.rowcount
    return inserted, len(records) - inserted


def verifier_insertion(engine: Engine) -> None:
    sql = text("""
        SELECT count(*) AS n, min(date_enquete) AS premiere, max(date_enquete) AS derniere
        FROM public.nutrition_observations
        WHERE source = :source;
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"source": SOURCE_LABEL}).fetchone()
    if not row or row.n == 0:
        log.warning("Vérification : aucune ligne source='{}' en base.", SOURCE_LABEL)
        return
    log.info(
        "Vérification — source='{}' : {} lignes, {} → {}",
        SOURCE_LABEL, row.n, row.premiere, row.derniere
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Récupère les indicateurs nutrition UNICEF (national) pour Madagascar"
    )
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    stats = FetchStats()

    try:
        raw_csv = fetch_unicef_csv()
    except Exception as exc:
        log.error("Échec de la requête UNICEF SDMX : {}", exc)
        sys.exit(1)

    par_annee = parse_unicef_csv(raw_csv)
    stats.n_annees_recuperees = len(par_annee)

    if not par_annee:
        log.error(
            "Aucune donnée exploitable dans la réponse UNICEF (CSV vide ou "
            "structure inattendue). Colle l'URL loguée ci-dessus dans un "
            "navigateur pour inspecter la réponse brute."
        )
        sys.exit(1)

    log.info(
        "{} années récupérées depuis UNICEF : {}",
        stats.n_annees_recuperees, sorted(par_annee.keys())
    )

    records = build_records(par_annee, args.start_year, args.end_year)
    stats.n_records_generes = len(records)

    log.warning(
        "ATTENTION : {} lignes générées en dupliquant la valeur NATIONALE sur "
        "les {} régions — ce ne sont PAS des mesures régionales réelles.",
        len(records), len(REGIONS_MADAGASCAR)
    )

    if args.dry_run:
        log.info("DRY-RUN : {} lignes seraient insérées, rien écrit en base.", len(records))
        for r in records[: len(REGIONS_MADAGASCAR)]:  # aperçu d'une année
            log.info(
                "  {} | {} | gam={} mam={} sam={} | {}",
                r["region_code"], r["date_enquete"], r["gam_pct"],
                r["mam_pct"], r["sam_pct"], r["classification_oms"]
            )
        return

    engine = sa.create_engine(settings.database.sync_url, pool_pre_ping=True)
    try:
        inserted, skipped = inserer_nutrition(engine, records)
        stats.n_inserees = inserted
        stats.n_doublons = skipped
        log.info(
            "Import terminé : {} insérées, {} doublons ignorés (déjà en base)",
            inserted, skipped
        )
        if args.verify:
            verifier_insertion(engine)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()