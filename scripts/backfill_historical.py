"""
Import des données historiques (3 ans).

Ce script orchestre le chargement massif des données historiques depuis
toutes les sources disponibles :

  Sources supportées :
    --source weather       → NASA POWER API (gratuit, 30+ ans d'historique)
    --source malaria       → WHO GHO API + fichiers CSV DHIS2 locaux
    --source nutrition     → Fichiers CSV UNICEF / enquêtes SMART
    --source all           → Toutes les sources (défaut)

  Modes d'exécution :
    --years 3              → Nombre d'années à importer (défaut: 3)
    --region MDG-ANA       → Une région spécifique (défaut: toutes)
    --resume               → Reprend depuis le dernier checkpoint
    --dry-run              → Simule sans écrire en base
    --workers 4            → Parallelisme (régions en parallèle)
    --batch-size 500       → Taille des batchs d'insertion

Usage :
    python scripts/backfill_historical.py --source all --years 3
    python scripts/backfill_historical.py --source weather --region MDG-ANA --years 5
    python scripts/backfill_historical.py --resume --workers 8
    python scripts/backfill_historical.py --source malaria --dry-run

Architecture :
    ┌─────────────────────────────────────────────────────┐
    │  Orchestrateur  (BackfillOrchestrator)              │
    │    ├── WeatherBackfiller  → NASA POWER API          │
    │    ├── MalariaBackfiller  → WHO GHO + CSV DHIS2     │
    │    └── NutritionBackfiller → CSV UNICEF / SMART     │
    │                                                     │
    │  Checkpoint system  → Redis ou fichier JSON local   │
    │  Progress tracking  → tqdm + log structuré          │
    │  Error recovery     → retry exponentiel + skip      │
    └─────────────────────────────────────────────────────┘

Prérequis :
    pip install httpx tqdm tenacity pandas

Auteur : Équipe Data
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from src.utils.constants import REGIONS_MADAGASCAR, Saison, get_saison_courante
from src.utils.geo_helpers import _REGIONS_DB, get_region_metadata
from src.utils.logger import get_logger, log_collecte, setup_logging
from src.utils.validators import validate_weather_payload

setup_logging()
log = get_logger("backfill")


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

NASA_POWER_BASE_URL  = "https://power.larc.nasa.gov/api/temporal/daily/point"
WHO_GHO_BASE_URL     = "https://ghoapi.azureedge.net/api"
CHECKPOINT_FILE      = ROOT / "data" / "processed" / ".backfill_checkpoint.json"
BATCH_SIZE_DEFAULT   = 500
MAX_RETRIES          = 4
HTTP_TIMEOUT_SEC     = 60
MAX_CONCURRENT_REQS  = 3   # NASA POWER rate limit

# Variables NASA POWER requises
NASA_POWER_PARAMS = [
    "T2M",        # Température 2m (°C)
    "T2M_MIN",    # Température min (°C)
    "T2M_MAX",    # Température max (°C)
    "RH2M",       # Humidité relative (%)
    "PRECTOTCORR",# Précipitations corrigées (mm/j)
    "WS2M",       # Vitesse vent 2m (m/s)
    "ALLSKY_SFC_SW_DWN",  # Rayonnement solaire (proxy NDVI)
]


# ─────────────────────────────────────────────────────────────────
# Dataclasses de progression
# ─────────────────────────────────────────────────────────────────

@dataclass
class BackfillStats:
    """Statistiques d'un job de backfill."""
    source:          str
    region_id:     str
    date_debut:      date
    date_fin:        date
    n_records_total: int   = 0
    n_inserted:      int   = 0
    n_skipped:       int   = 0
    n_errors:        int   = 0
    duree_sec:       float = 0.0
    statut:          str   = "en_cours"   # en_cours | termine | erreur | partiel

    @property
    def taux_succes(self) -> float:
        if self.n_records_total == 0:
            return 0.0
        return round((self.n_inserted + self.n_skipped) / self.n_records_total, 4)


@dataclass
class BackfillCheckpoint:
    """
    Point de reprise pour les backfills interrompus.
    Sérialisé en JSON dans data/processed/.backfill_checkpoint.json
    """
    regions_terminees:  List[str]            = field(default_factory=list)
    regions_en_erreur:  List[str]            = field(default_factory=list)
    derniere_region:    Optional[str]        = None
    derniere_date:      Optional[str]        = None   # ISO format
    stats_globales:     Dict[str, Any]       = field(default_factory=dict)
    timestamp_debut:    str                  = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ─────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────

def load_checkpoint() -> Optional[BackfillCheckpoint]:
    """Charge le checkpoint depuis le fichier JSON."""
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cp = BackfillCheckpoint(**data)
        log.info(
            "Checkpoint chargé : {} régions terminées, {} en erreur",
            len(cp.regions_terminees), len(cp.regions_en_erreur)
        )
        return cp
    except Exception as exc:
        log.warning("Impossible de charger le checkpoint : {} — démarrage à zéro", exc)
        return None


def save_checkpoint(cp: BackfillCheckpoint) -> None:
    """Sauvegarde le checkpoint."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(asdict(cp), f, ensure_ascii=False, indent=2)


def clear_checkpoint() -> None:
    """Supprime le checkpoint (fin de backfill réussi)."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        log.info("Checkpoint supprimé")


# ─────────────────────────────────────────────────────────────────
# 1. WEATHER BACKFILLER — NASA POWER API
# ─────────────────────────────────────────────────────────────────

class WeatherBackfiller:
    """
    Import historique météo via NASA POWER API (gratuit, pas de clé API).

    NASA POWER fournit des données journalières depuis 1981 avec une
    résolution spatiale de 0.5° × 0.5° (≈ 55 km) — suffisant pour
    des analyses régionales à Madagascar.

    Doc API : https://power.larc.nasa.gov/docs/services/api/temporal/daily/
    """

    def __init__(self, engine: Engine, batch_size: int = BATCH_SIZE_DEFAULT):
        self.engine     = engine
        self.batch_size = batch_size
        self.client     = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC)
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQS)

    async def close(self) -> None:
        await self.client.aclose()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _fetch_nasa_power(
        self,
        lat: float,
        lon: float,
        date_start: date,
        date_end: date,
    ) -> Dict[str, Any]:
        """Appel HTTP vers NASA POWER avec retry exponentiel."""
        params = {
            "parameters": ",".join(NASA_POWER_PARAMS),
            "community":  "AG",
            "longitude":  lon,
            "latitude":   lat,
            "start":      date_start.strftime("%Y%m%d"),
            "end":        date_end.strftime("%Y%m%d"),
            "format":     "JSON",
        }
        async with self._semaphore:
            response = await self.client.get(NASA_POWER_BASE_URL, params=params)
            response.raise_for_status()
            return response.json()

    def _parse_nasa_response(
        self,
        data: Dict[str, Any],
        region_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse la réponse NASA POWER en liste de dicts pour l'insertion.
        Retourne les records valides uniquement.
        """
        records = []
        try:
            params_data = data["properties"]["parameter"]
            dates = list(params_data["T2M"].keys())  # Format YYYYMMDD

            for date_str in dates:
                try:
                    ts = datetime.strptime(date_str, "%Y%m%d")

                    # NASA POWER utilise -999 pour les valeurs manquantes
                    def safe_val(param: str, default=None):
                        v = params_data.get(param, {}).get(date_str, -999)
                        return None if v == -999 else v

                    temp     = safe_val("T2M")
                    temp_min = safe_val("T2M_MIN")
                    temp_max = safe_val("T2M_MAX")
                    humidity = safe_val("RH2M")
                    precip   = safe_val("PRECTOTCORR", 0.0)
                    wind_ms  = safe_val("WS2M")

                    # Skip si données critiques manquantes
                    if temp is None or humidity is None:
                        continue

                    # Conversion vent m/s → km/h
                    wind_kmh = round(wind_ms * 3.6, 2) if wind_ms is not None else None

                    # Proxy NDVI depuis rayonnement solaire (approximation)
                    radiation = safe_val("ALLSKY_SFC_SW_DWN")
                    ndvi_proxy = round(min(0.8, radiation / 25.0), 3) if radiation else None

                    record = {
                        "region_id":      region_id,
                        "horodatage":    ts.isoformat(),
                        "temperature_c":    round(temp, 2),
                        "temp_min_c":       round(temp_min, 2) if temp_min else None,
                        "temp_max_c":       round(temp_max, 2) if temp_max else None,
                        "humidite_pct":     round(min(100.0, max(0.0, humidity)), 1),
                        "precipitations_mm": round(max(0.0, precip or 0.0), 2),
                        "vent_kmh": wind_kmh,
                        "ndvi":             ndvi_proxy,
                        "source_api":       "nasa_power",
                        "qualite_flag":     0,
                    }
                    records.append(record)

                except (KeyError, ValueError) as exc:
                    log.debug("Parse NASA POWER date={} : {}", date_str, exc)
                    continue

        except KeyError as exc:
            log.error("Structure NASA POWER inattendue : {}", exc)

        return records

    async def backfill_region(
        self,
        region_id: str,
        date_start: date,
        date_end: date,
        dry_run: bool = False,
    ) -> BackfillStats:
        """Backfill météo complet pour une région et une période."""
        stats = BackfillStats(
            source="nasa_power",
            region_id=region_id,
            date_debut=date_start,
            date_fin=date_end,
        )
        t0 = time.perf_counter()

        meta = get_region_metadata(region_id)
        if not meta:
            log.error("Région inconnue : {}", region_id)
            stats.statut = "erreur"
            return stats

        lat, lon = meta.centroid

        # Découpage en fenêtres de 1 an (limite API NASA POWER)
        windows: List[Tuple[date, date]] = []
        current = date_start
        while current < date_end:
            window_end = min(current + timedelta(days=364), date_end)
            windows.append((current, window_end))
            current = window_end + timedelta(days=1)

        all_records: List[Dict] = []

        for win_start, win_end in windows:
            log.debug(
                "NASA POWER {} : {} → {}",
                region_id, win_start, win_end
            )
            try:
                raw = await self._fetch_nasa_power(lat, lon, win_start, win_end)
                records = self._parse_nasa_response(raw, region_id)
                all_records.extend(records)
                stats.n_records_total += len(records)
                log.debug(
                    "  {} records parsés pour {} [{} → {}]",
                    len(records), region_id, win_start, win_end
                )
            except Exception as exc:
                log.error(
                    "Erreur NASA POWER {} [{} → {}] : {}",
                    region_id, win_start, win_end, exc
                )
                stats.n_errors += 1

        if not dry_run and all_records:
            inserted, skipped = self._bulk_insert_weather(all_records)
            stats.n_inserted = inserted
            stats.n_skipped  = skipped
        elif dry_run:
            log.info(
                "DRY-RUN {} : {} records à insérer",
                region_id, len(all_records)
            )
            stats.n_inserted = 0

        stats.duree_sec = round(time.perf_counter() - t0, 2)
        stats.statut    = "partiel" if stats.n_errors > 0 else "termine"

        log.info(
            "  {} : {} récupérés depuis NASA POWER | {} insérés | {} déjà présents (skip)",
            region_id, stats.n_records_total, stats.n_inserted, stats.n_skipped
        )
        log_collecte(
            source="nasa_power",
            region_id=region_id,
            n_records=stats.n_inserted,
            duree_sec=stats.duree_sec,
            statut=stats.statut,
        )
        return stats

    def _bulk_insert_weather(
        self, records: List[Dict]
    ) -> Tuple[int, int]:
        """
        Insertion en bulk avec ON CONFLICT DO NOTHING.
        Retourne (n_inserted, n_skipped).
        """
        # NB : weather_observations est partitionnée par horodatage (PK
        # composite (id, horodatage)). Aujourd'hui aucune contrainte UNIQUE
        # ne porte sur (region_id, horodatage) — seuls des index non-uniques
        # existent (idx_weather_region_time, idx_weather_region_ts). Sans la
        # migration `uq_weather_region_horodatage` fournie séparément, ce
        # ON CONFLICT échoue avec :
        #   psycopg2.errors.InvalidColumnReference: there is no unique or
        #   exclusion constraint matching the ON CONFLICT specification
        sql = text("""
            INSERT INTO public.weather_observations (
                region_id, horodatage,
                temperature_c, temp_min_c, temp_max_c,
                humidite_pct, precipitations_mm,
                vent_kmh, ndvi, source_api, qualite_flag
            ) VALUES (
                :region_id, :horodatage,
                :temperature_c, :temp_min_c, :temp_max_c,
                :humidite_pct, :precipitations_mm,
                :vent_kmh, :ndvi, :source_api, :qualite_flag
            )
            ON CONFLICT (region_id, horodatage) DO NOTHING;
        """)

        inserted = 0
        # Insertion par batches
        for i in range(0, len(records), self.batch_size):
            batch = records[i : i + self.batch_size]
            with self.engine.begin() as conn:
                result = conn.execute(sql, batch)
                inserted += result.rowcount

        skipped = len(records) - inserted
        return inserted, skipped


# ─────────────────────────────────────────────────────────────────
# 2. MALARIA BACKFILLER — WHO GHO API + CSV DHIS2
# ─────────────────────────────────────────────────────────────────

class MalariaBackfiller:
    """
    Import historique paludisme depuis :
      1. WHO Global Health Observatory (API REST) — données agrégées nationales/régionales
      2. Fichiers CSV exportés depuis DHIS2 (données district par district)

    Stratégie :
        - WHO GHO pour les données 2010–2020 (historique solide)
        - CSV DHIS2 pour 2020–présent (données opérationnelles récentes)
        - Fusion avec priorisation DHIS2 en cas de conflit
    """

    WHO_MALARIA_INDICATOR = "MALARIA_EST_INCIDENCE"  # Incidence pour 1000 pop

    def __init__(self, engine: Engine, batch_size: int = BATCH_SIZE_DEFAULT):
        self.engine     = engine
        self.batch_size = batch_size
        self.client     = httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC)

    async def close(self) -> None:
        await self.client.aclose()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    async def _fetch_who_gho(
        self, indicator: str, country: str = "MDG"
    ) -> List[Dict]:
        """Récupère les données WHO GHO pour Madagascar."""
        url = f"{WHO_GHO_BASE_URL}/{indicator}"
        params = {
            "$filter": f"SpatialDim eq '{country}'",
            "$select": "TimeDim,NumericValue,Low,High,SpatialDim",
            "$top": 1000,
        }
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json().get("value", [])

    async def backfill_from_who(
        self,
        region_id: str,
        date_start: date,
        date_end: date,
        dry_run: bool = False,
    ) -> BackfillStats:
        """
        Importe les données WHO GHO.
        Note : WHO fournit des données annuelles — on les distribue
        uniformément sur les 52 semaines de l'année avec saisonnalité.
        """
        stats = BackfillStats(
            source="who_gho",
            region_id=region_id,
            date_debut=date_start,
            date_fin=date_end,
        )
        t0 = time.perf_counter()

        try:
            raw_data = await self._fetch_who_gho(self.WHO_MALARIA_INDICATOR)
        except Exception as exc:
            log.error("Erreur WHO GHO : {}", exc)
            stats.statut = "erreur"
            stats.n_errors = 1
            return stats

        # Filtrage par années demandées
        years_range = range(date_start.year, date_end.year + 1)
        annual_data = {
            int(d["TimeDim"]): d.get("NumericValue", 0)
            for d in raw_data
            if d.get("TimeDim") and int(d["TimeDim"]) in years_range
        }

        records = []
        meta = get_region_metadata(region_id)
        if not meta:
            stats.statut = "erreur"
            return stats

        # Distribution des cas annuels sur les semaines avec saisonnalité
        for year, annual_incidence in annual_data.items():
            if annual_incidence is None:
                continue
            # Estimation population régionale
            pop = meta.population_estimee
            # Incidence pour 1000 → nombre de cas annuels
            cas_annuels = int((annual_incidence / 1000) * pop)

            for semaine in range(1, 53):
                # Pondération saisonnière : saison des pluies = 3× plus de cas
                mois_approx = ((semaine - 1) // 4) + 1
                saison = get_saison_courante(mois_approx)
                if saison == Saison.SAISON_PLUIES:
                    poids = 3.0
                elif saison == Saison.TRANSITION:
                    poids = 1.5
                else:
                    poids = 0.5

                # Normalisation pour sommer à 100%
                # Simplification : 22 semaines pluies (3x), 8 transition (1.5x), 22 sèches (0.5x)
                total_poids = 22 * 3.0 + 8 * 1.5 + 22 * 0.5
                cas_semaine = int(cas_annuels * poids / total_poids)

                debut_semaine = date(year, 1, 1) + timedelta(weeks=semaine - 1)
                if not (date_start <= debut_semaine <= date_end):
                    continue

                records.append({
                    "region_code":       region_id,
                    "semaine_iso":       semaine,
                    "annee":             year,
                    "date_debut_semaine": debut_semaine.isoformat(),
                    "cas_confirmes":     max(0, cas_semaine),
                    "cas_presumes":      None,
                    "deces":             None,
                    "tests_realises":    None,
                    "district":          None,
                    "dhis2_org_unit_id": None,
                    "source":            "who_gho_distribue",
                    "valide":            True,
                })

        stats.n_records_total = len(records)

        if not dry_run and records:
            inserted, skipped = self._bulk_insert_malaria(records)
            stats.n_inserted = inserted
            stats.n_skipped  = skipped
        elif dry_run:
            log.info("DRY-RUN malaria WHO {} : {} records", region_id, len(records))

        stats.duree_sec = round(time.perf_counter() - t0, 2)
        stats.statut    = "termine"
        return stats

    def backfill_from_csv(
        self,
        csv_path: Path,
        dry_run: bool = False,
    ) -> BackfillStats:
        """
        Importe des données DHIS2 depuis un CSV exporté.

        Format CSV attendu (colonnes) :
            region_id, semaine_iso, annee, date_debut_semaine,
            cas_confirmes, cas_presumes, deces, tests_realises,
            district, dhis2_org_unit_id

        Tous les autres formats sont ignorés avec warning.
        """
        REQUIRED_COLS = {
            "region_id", "semaine_iso", "annee",
            "cas_confirmes",
        }

        stats = BackfillStats(
            source="dhis2_csv",
            region_id="ALL",
            date_debut=date(2020, 1, 1),
            date_fin=date.today(),
        )
        t0 = time.perf_counter()

        if not csv_path.exists():
            log.error("Fichier CSV introuvable : {}", csv_path)
            stats.statut = "erreur"
            return stats

        records = []
        errors  = 0

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            if not REQUIRED_COLS.issubset(set(reader.fieldnames or [])):
                missing = REQUIRED_COLS - set(reader.fieldnames or [])
                log.error(
                    "CSV DHIS2 : colonnes manquantes : {}. "
                    "Colonnes présentes : {}",
                    missing, reader.fieldnames
                )
                stats.statut = "erreur"
                return stats

            for i, row in enumerate(reader, start=2):  # Line 2 = première donnée
                try:
                    region = row["region_id"].strip().upper()
                    if region not in REGIONS_MADAGASCAR:
                        log.debug("Ligne {} : région inconnue '{}' — ignorée", i, region)
                        errors += 1
                        continue

                    semaine = int(row["semaine_iso"])
                    annee   = int(row["annee"])
                    cas     = int(row.get("cas_confirmes", 0) or 0)

                    if not (1 <= semaine <= 53):
                        log.warning("Ligne {} : semaine ISO invalide {} — ignorée", i, semaine)
                        errors += 1
                        continue

                    debut = row.get("date_debut_semaine", "")
                    if not debut:
                        debut = (
                            date(annee, 1, 1) + timedelta(weeks=semaine - 1)
                        ).isoformat()

                    records.append({
                        "region_code":       region,
                        "semaine_iso":       semaine,
                        "annee":             annee,
                        "date_debut_semaine": debut,
                        "cas_confirmes":     max(0, cas),
                        "cas_presumes":      int(row["cas_presumes"]) if row.get("cas_presumes") else None,
                        "deces":             int(row["deces"])        if row.get("deces")        else None,
                        "tests_realises":    int(row["tests_realises"]) if row.get("tests_realises") else None,
                        "district":          row.get("district", "").strip() or None,
                        "dhis2_org_unit_id": row.get("dhis2_org_unit_id", "").strip() or None,
                        "source":            "dhis2_csv",
                        "valide":            True,
                    })

                except (ValueError, KeyError) as exc:
                    log.warning("Ligne {} CSV malaria invalide : {}", i, exc)
                    errors += 1

        stats.n_records_total = len(records)
        stats.n_errors        = errors

        if not dry_run and records:
            inserted, skipped = self._bulk_insert_malaria(records)
            stats.n_inserted = inserted
            stats.n_skipped  = skipped
        elif dry_run:
            log.info("DRY-RUN malaria CSV : {} records valides, {} erreurs", len(records), errors)

        stats.duree_sec = round(time.perf_counter() - t0, 2)
        stats.statut    = "termine" if errors == 0 else "partiel"

        log_collecte(
            source="dhis2_csv",
            region_id=None,
            n_records=stats.n_inserted,
            duree_sec=stats.duree_sec,
            statut=stats.statut,
        )
        return stats

    def _bulk_insert_malaria(self, records: List[Dict]) -> Tuple[int, int]:
        """Insertion bulk avec ON CONFLICT DO NOTHING (unicité region+semaine+annee+source)."""
        # NB : malaria_observations est partitionnée par date_debut_semaine
        # (PK composite (id, date_debut_semaine)). Toute contrainte UNIQUE
        # doit donc inclure la colonne de partitionnement — voir la migration
        # `uq_malaria_region_periode` (region_code, semaine_iso, annee, source,
        # date_debut_semaine) fournie séparément.
        sql = text("""
            INSERT INTO public.malaria_observations (
                region_code, semaine_iso, annee, date_debut_semaine,
                cas_confirmes, cas_presumes, deces, tests_realises,
                district, dhis2_org_unit_id, source, valide
            ) VALUES (
                :region_code, :semaine_iso, :annee, :date_debut_semaine,
                :cas_confirmes, :cas_presumes, :deces, :tests_realises,
                :district, :dhis2_org_unit_id, :source, :valide
            )
            ON CONFLICT (region_code, semaine_iso, annee, source, date_debut_semaine)
            DO NOTHING;
        """)
        inserted = 0
        for i in range(0, len(records), self.batch_size):
            batch = records[i : i + self.batch_size]
            with self.engine.begin() as conn:
                result = conn.execute(sql, batch)
                inserted += result.rowcount
        return inserted, len(records) - inserted


# ─────────────────────────────────────────────────────────────────
# 3. NUTRITION BACKFILLER — CSV UNICEF / SMART
# ─────────────────────────────────────────────────────────────────

class NutritionBackfiller:
    """
    Import historique nutrition depuis fichiers CSV d'enquêtes SMART / UNICEF.

    Format CSV attendu :
        region_id, date_enquete, gam_pct, mam_pct, sam_pct,
        groupe_cible, n_enfants_enquetes, score_sca, source

    Les données nutrition sont rares (1–4 enquêtes/an par région) —
    pas de génération synthétique.
    """

    REQUIRED_COLS = {
        "region_id", "date_enquete", "gam_pct", "groupe_cible",
    }

    def __init__(self, engine: Engine, batch_size: int = BATCH_SIZE_DEFAULT):
        self.engine     = engine
        self.batch_size = batch_size

    def backfill_from_csv(
        self,
        csv_path: Path,
        dry_run: bool = False,
    ) -> BackfillStats:
        """Import depuis CSV d'enquêtes SMART."""
        stats = BackfillStats(
            source="nutrition_csv",
            region_id="ALL",
            date_debut=date(2010, 1, 1),
            date_fin=date.today(),
        )
        t0 = time.perf_counter()

        if not csv_path.exists():
            log.error("Fichier CSV nutrition introuvable : {}", csv_path)
            stats.statut = "erreur"
            return stats

        records = []
        errors  = 0

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            cols = set(reader.fieldnames or [])
            if not self.REQUIRED_COLS.issubset(cols):
                missing = self.REQUIRED_COLS - cols
                log.error("CSV nutrition : colonnes manquantes : {}", missing)
                stats.statut = "erreur"
                return stats

            for i, row in enumerate(reader, start=2):
                try:
                    region   = row["region_id"].strip().upper()
                    if region not in REGIONS_MADAGASCAR:
                        errors += 1
                        continue

                    gam = float(row["gam_pct"])
                    if not (0.0 <= gam <= 60.0):
                        log.warning("Ligne {} : GAM={} hors plage — ignorée", i, gam)
                        errors += 1
                        continue

                    # Classification OMS automatique
                    if gam < 5.0:
                        classif = "acceptable"
                    elif gam < 10.0:
                        classif = "alerte"
                    elif gam < 15.0:
                        classif = "urgence"
                    else:
                        classif = "crise"

                    records.append({
                        "region_code":        region,
                        "date_enquete":       row["date_enquete"].strip(),
                        "gam_pct":            round(gam, 2),
                        "mam_pct":            float(row["mam_pct"]) if row.get("mam_pct") else None,
                        "sam_pct":            float(row["sam_pct"]) if row.get("sam_pct") else None,
                        "classification_oms": classif,
                        "groupe_cible":       row["groupe_cible"].strip(),
                        "n_enfants_enquetes": int(row["n_enfants_enquetes"]) if row.get("n_enfants_enquetes") else None,
                        "score_sca":          float(row["score_sca"]) if row.get("score_sca") else None,
                        "source":             row.get("source", "smart_enquete").strip(),
                        "metadata":           json.dumps({"import": "backfill_historique"}),
                    })

                except (ValueError, KeyError) as exc:
                    log.warning("Ligne {} CSV nutrition invalide : {}", i, exc)
                    errors += 1

        stats.n_records_total = len(records)
        stats.n_errors        = errors

        if not dry_run and records:
            inserted, skipped = self._bulk_insert_nutrition(records)
            stats.n_inserted = inserted
            stats.n_skipped  = skipped
        elif dry_run:
            log.info(
                "DRY-RUN nutrition CSV : {} records valides, {} erreurs",
                len(records), errors
            )

        stats.duree_sec = round(time.perf_counter() - t0, 2)
        stats.statut    = "termine" if errors == 0 else "partiel"

        log_collecte(
            source="nutrition_csv",
            region_id=None,
            n_records=stats.n_inserted,
            duree_sec=stats.duree_sec,
            statut=stats.statut,
        )
        return stats

    def _bulk_insert_nutrition(self, records: List[Dict]) -> Tuple[int, int]:
        # NB : nutrition_observations n'a aujourd'hui aucune contrainte UNIQUE
        # (seule la PK sur `id`). Un "ON CONFLICT DO NOTHING" sans cible est
        # un no-op silencieux : il n'empêche AUCUN doublon. Voir la migration
        # `uq_nutrition_region_enquete` fournie séparément (region_code,
        # date_enquete, groupe_cible, source).
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
        for i in range(0, len(records), self.batch_size):
            batch = records[i : i + self.batch_size]
            with self.engine.begin() as conn:
                result = conn.execute(sql, batch)
                inserted += result.rowcount
        return inserted, len(records) - inserted


# ─────────────────────────────────────────────────────────────────
# 4. ORCHESTRATEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────────

class BackfillOrchestrator:
    """
    Orchestre le backfill complet de toutes les sources.

    Gestion :
        - Sélection des régions et de la période
        - Checkpoint / reprise
        - Parallélisme (asyncio + semaphore)
        - Rapport final consolidé
    """

    def __init__(
        self,
        engine:     Engine,
        source:     str   = "all",
        years:      int   = 3,
        regions:    Optional[List[str]] = None,
        batch_size: int   = BATCH_SIZE_DEFAULT,
        workers:    int   = 3,
        dry_run:    bool  = False,
        resume:     bool  = False,
        malaria_csv: Optional[Path] = None,
        nutrition_csv: Optional[Path] = None,
    ):
        self.engine         = engine
        self.source         = source
        self.years          = years
        self.batch_size     = batch_size
        self.workers        = min(workers, MAX_CONCURRENT_REQS)
        self.dry_run        = dry_run
        self.resume         = resume
        self.malaria_csv    = malaria_csv
        self.nutrition_csv  = nutrition_csv

        self.date_end   = date.today()
        self.date_start = date(self.date_end.year - years, self.date_end.month, self.date_end.day)

        # Sélection des régions
        self.regions = regions or list(_REGIONS_DB.keys())

        # Chargement checkpoint
        self.checkpoint = load_checkpoint() if resume else BackfillCheckpoint()

        # Filtre si reprise
        if resume and self.checkpoint:
            deja_faites = set(self.checkpoint.regions_terminees)
            self.regions = [r for r in self.regions if r not in deja_faites]
            log.info(
                "Reprise : {} régions restantes ({} déjà traitées)",
                len(self.regions), len(deja_faites)
            )

        # Stats globales
        self.all_stats: List[BackfillStats] = []

    async def run(self) -> Dict[str, Any]:
        """Lance le backfill complet et retourne le rapport final."""
        t_start = time.perf_counter()

        log.info(
            "Démarrage backfill — source={} années={} régions={} workers={} dry_run={}",
            self.source, self.years, len(self.regions), self.workers, self.dry_run
        )
        log.info("Période : {} → {}", self.date_start, self.date_end)

        # ── Météo (async, régions en parallèle) ─────────────────
        if self.source in ("weather", "all"):
            await self._run_weather_backfill()

        # ── Paludisme ────────────────────────────────────────────
        if self.source in ("malaria", "all"):
            await self._run_malaria_backfill()

        # ── Nutrition ────────────────────────────────────────────
        if self.source in ("nutrition", "all"):
            self._run_nutrition_backfill()

        # ── Rafraîchissement vues matérialisées ──────────────────
        if not self.dry_run:
            self._refresh_views()

        # ── Rapport final ────────────────────────────────────────
        elapsed = round(time.perf_counter() - t_start, 1)
        rapport = self._build_report(elapsed)

        # Suppression checkpoint si succès complet
        if rapport["statut_global"] == "succes":
            clear_checkpoint()
        else:
            save_checkpoint(self.checkpoint)

        self._print_report(rapport)
        return rapport

    async def _run_weather_backfill(self) -> None:
        """Backfill météo avec pool de workers async."""
        log.info("=== BACKFILL MÉTÉO ({} régions) ===", len(self.regions))
        backfiller = WeatherBackfiller(self.engine, self.batch_size)

        semaphore = asyncio.Semaphore(self.workers)

        async def process_region(code: str) -> BackfillStats:
            async with semaphore:
                log.info("  Météo → {}", code)
                try:
                    stats = await backfiller.backfill_region(
                        code, self.date_start, self.date_end, self.dry_run
                    )
                    self.checkpoint.regions_terminees.append(code)
                    save_checkpoint(self.checkpoint)
                    return stats
                except Exception as exc:
                    log.error("  Échec météo {} : {}", code, exc)
                    self.checkpoint.regions_en_erreur.append(code)
                    return BackfillStats(
                        source="nasa_power", region_id=code,
                        date_debut=self.date_start, date_fin=self.date_end,
                        statut="erreur", n_errors=1,
                    )

        tasks  = [process_region(code) for code in self.regions]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        self.all_stats.extend(results)

        await backfiller.close()

        n_ok  = sum(1 for s in results if s.statut == "termine")
        n_err = sum(1 for s in results if s.statut == "erreur")
        total = sum(s.n_inserted for s in results)
        log.info(
            "  Météo terminé : {}/{} régions OK, {} records insérés",
            n_ok, len(self.regions), total
        )

    async def _run_malaria_backfill(self) -> None:
        """Backfill paludisme : WHO GHO async + CSV sync si fourni."""
        log.info("=== BACKFILL PALUDISME ===")
        backfiller = MalariaBackfiller(self.engine, self.batch_size)

        # CSV DHIS2 en priorité si fourni
        if self.malaria_csv:
            log.info("  Source CSV DHIS2 : {}", self.malaria_csv)
            stats = backfiller.backfill_from_csv(self.malaria_csv, self.dry_run)
            self.all_stats.append(stats)
            log.info(
                "  CSV paludisme : {} insérés, {} erreurs",
                stats.n_inserted, stats.n_errors
            )

        # WHO GHO pour compléter (données manquantes)
        log.info("  Source WHO GHO pour {} régions...", len(self.regions))
        semaphore = asyncio.Semaphore(self.workers)

        async def process_region(code: str) -> BackfillStats:
            async with semaphore:
                try:
                    return await backfiller.backfill_from_who(
                        code, self.date_start, self.date_end, self.dry_run
                    )
                except Exception as exc:
                    log.error("  Échec WHO GHO {} : {}", code, exc)
                    return BackfillStats(
                        source="who_gho", region_id=code,
                        date_debut=self.date_start, date_fin=self.date_end,
                        statut="erreur", n_errors=1,
                    )

        tasks   = [process_region(code) for code in self.regions]
        results = await asyncio.gather(*tasks)
        self.all_stats.extend(results)

        await backfiller.close()
        total = sum(s.n_inserted for s in results)
        log.info("  WHO GHO terminé : {} records insérés", total)

    def _run_nutrition_backfill(self) -> None:
        """Backfill nutrition depuis CSV."""
        log.info("=== BACKFILL NUTRITION ===")

        if not self.nutrition_csv:
            log.warning(
                "  Aucun fichier CSV nutrition fourni (--nutrition-csv). "
                "Les données nutrition doivent être importées manuellement "
                "depuis les enquêtes SMART UNICEF."
            )
            log.info(
                "  Format attendu : data/raw/nutrition_smart_surveys.csv\n"
                "  Colonnes : region_id, date_enquete, gam_pct, mam_pct, "
                "sam_pct, groupe_cible, n_enfants_enquetes, score_sca, source"
            )
            return

        backfiller = NutritionBackfiller(self.engine, self.batch_size)
        stats = backfiller.backfill_from_csv(self.nutrition_csv, self.dry_run)
        self.all_stats.append(stats)
        log.info(
            "  Nutrition : {} insérés, {} erreurs ({})",
            stats.n_inserted, stats.n_errors, stats.statut
        )

    def _refresh_views(self) -> None:
        """Rafraîchit les vues matérialisées après le backfill."""
        log.info("Rafraîchissement des vues matérialisées...")
        # NB : le schéma "ml" n'existe pas dans malaria_db (\d ne liste que
        # public et topology) et "ml_predictions" est une table normale, pas
        # une vue matérialisée. La ligne "ml.mv_latest_predictions" a été
        # retirée : elle échouait systématiquement (silencieusement avalée
        # par le except Exception ci-dessous).
        views = [
            "public.mv_malaria_weekly_summary",
            "public.mv_nutrition_status",
        ]
        with self.engine.begin() as conn:
            for view in views:
                try:
                    conn.execute(text(f"REFRESH MATERIALIZED VIEW {view};"))
                    log.info("  ✓ {}", view)
                except Exception as exc:
                    log.warning("  ⚠ {} : {}", view, exc)

    def _build_report(self, elapsed_sec: float) -> Dict[str, Any]:
        """Construit le rapport de synthèse du backfill."""
        by_source: Dict[str, Dict] = defaultdict(lambda: {
            "n_regions": 0, "n_inserted": 0, "n_errors": 0, "n_skipped": 0
        })

        for s in self.all_stats:
            src = by_source[s.source]
            src["n_regions"]  += 1
            src["n_inserted"] += s.n_inserted
            src["n_errors"]   += s.n_errors
            src["n_skipped"]  += s.n_skipped

        total_inserted = sum(s.n_inserted for s in self.all_stats)
        total_errors   = sum(s.n_errors   for s in self.all_stats)
        regions_en_err = set(self.checkpoint.regions_en_erreur)

        statut_global = (
            "succes"  if total_errors == 0 and not regions_en_err else
            "partiel" if total_inserted > 0 else
            "erreur"
        )

        return {
            "statut_global":      statut_global,
            "duree_sec":          elapsed_sec,
            "periode":            f"{self.date_start} → {self.date_end}",
            "n_regions_traitees": len(self.regions),
            "n_regions_erreur":   len(regions_en_err),
            "total_records_inserits": total_inserted,
            "total_erreurs":      total_errors,
            "dry_run":            self.dry_run,
            "detail_par_source":  dict(by_source),
            "regions_en_erreur":  list(regions_en_err),
            "timestamp":          datetime.utcnow().isoformat(),
        }

    def _print_report(self, rapport: Dict) -> None:
        """Affiche un résumé formaté dans les logs."""
        sep = "─" * 60
        log.info(sep)
        log.info("RAPPORT BACKFILL HISTORIQUE")
        log.info(sep)
        log.info("Statut global    : {}", rapport["statut_global"].upper())
        log.info("Durée totale     : {:.0f}s ({:.1f} min)", rapport["duree_sec"], rapport["duree_sec"] / 60)
        log.info("Période          : {}", rapport["periode"])
        log.info("Régions traitées : {}", rapport["n_regions_traitees"])
        log.info("Records insérés  : {:,}", rapport["total_records_inserits"])
        log.info("Erreurs          : {}", rapport["total_erreurs"])
        log.info("Dry-run          : {}", rapport["dry_run"])
        log.info("")
        log.info("Détail par source :")
        for src, d in rapport["detail_par_source"].items():
            log.info(
                "  {:<25} : {:>8,} insérés | {:>8,} déjà présents | {:>5,} erreurs",
                src, d["n_inserted"], d["n_skipped"], d["n_errors"]
            )
        if rapport["regions_en_erreur"]:
            log.warning("Régions en erreur : {}", rapport["regions_en_erreur"])
        log.info(sep)

        # Sauvegarde rapport JSON
        rapport_path = ROOT / "data" / "processed" / f"backfill_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        rapport_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rapport_path, "w", encoding="utf-8") as f:
            json.dump(rapport, f, ensure_ascii=False, indent=2)
        log.info("Rapport sauvegardé : {}", rapport_path)


# ─────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import des données historiques (météo, paludisme, nutrition)"
    )
    parser.add_argument(
        "--source",
        choices=["weather", "malaria", "nutrition", "all"],
        default="all",
        help="Source(s) de données à importer (défaut: all)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Nombre d'années à importer (défaut: 3)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Code région spécifique (ex: MDG-ANA). Défaut: toutes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=f"Taille des batchs d'insertion (défaut: {BATCH_SIZE_DEFAULT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Nombre de régions traitées en parallèle (défaut: 3, max: 3 NASA limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simule le backfill sans écrire en base",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprend depuis le dernier checkpoint",
    )
    parser.add_argument(
        "--malaria-csv",
        type=Path,
        default=None,
        help="Chemin vers le CSV DHIS2 paludisme (optionnel)",
    )
    parser.add_argument(
        "--nutrition-csv",
        type=Path,
        default=None,
        help="Chemin vers le CSV enquêtes nutrition SMART (optionnel)",
    )
    args = parser.parse_args()

    # Validation région
    regions = None
    if args.region:
        if args.region not in REGIONS_MADAGASCAR:
            log.error(
                "Région inconnue : '{}'. Codes valides : {}",
                args.region, REGIONS_MADAGASCAR
            )
            sys.exit(1)
        regions = [args.region]

    # Connexion DB
    engine = sa.create_engine(
        settings.database.sync_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=5,
    )

    orchestrator = BackfillOrchestrator(
        engine=engine,
        source=args.source,
        years=args.years,
        regions=regions,
        batch_size=args.batch_size,
        workers=args.workers,
        dry_run=args.dry_run,
        resume=args.resume,
        malaria_csv=args.malaria_csv,
        nutrition_csv=args.nutrition_csv,
    )

    try:
        rapport = asyncio.run(orchestrator.run())
        exit_code = 0 if rapport["statut_global"] in ("succes", "partiel") else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log.warning("Backfill interrompu par l'utilisateur — checkpoint sauvegardé")
        save_checkpoint(orchestrator.checkpoint)
        sys.exit(130)
    except Exception as exc:
        log.exception("Erreur fatale : {}", exc)
        sys.exit(1)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()