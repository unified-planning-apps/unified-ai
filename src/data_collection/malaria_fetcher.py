"""
Collecte de données épidémiologiques paludisme :
  - DHIS2 (District Health Information System) — Ministère Santé Madagascar
  - WHO Global Health Observatory (GHO) API
  - Traitement et normalisation des données brutes
  - Calcul des indicateurs épidémiologiques (taux incidence, positivité TDR)
  - Détection automatique des seuils d'alerte

Sources :
  DHIS2 : cas confirmés par district, hebdomadaire
  WHO GHO : données consolidées nationales, historique long
"""

from __future__ import annotations

import asyncio
import base64
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings


# ─────────────────────────────────────────────────────────────────
# Mapping DHIS2 : ID OrgUnit → region_id interne
# ─────────────────────────────────────────────────────────────────
DHIS2_ORG_UNIT_MAP: Dict[str, str] = {
    "OU_MDG_ANA":  "MDG-ANA",
    "OU_MDG_VAK":  "MDG-VAK",
    "OU_MDG_ITM":  "MDG-ITM",
    "OU_MDG_BMT":  "MDG-BMT",
    "OU_MDG_MAT":  "MDG-MAT",
    "OU_MDG_ATI":  "MDG-ATI",
    "OU_MDG_VAT":  "MDG-VAT",
    "OU_MDG_FIT":  "MDG-FIT",
    "OU_MDG_ANO":  "MDG-ANO",
    "OU_MDG_ATS":  "MDG-ATS",
    "OU_MDG_ANA2": "MDG-ANA2",
    "OU_MDG_ALA":  "MDG-ALA",
    "OU_MDG_BOE":  "MDG-BOE",
    "OU_MDG_SOF":  "MDG-SOF",
    "OU_MDG_MEN":  "MDG-MEN",
    "OU_MDG_MEN2": "MDG-MEN2",
    "OU_MDG_DIA":  "MDG-DIA",
    "OU_MDG_SAV":  "MDG-SAV",
    "OU_MDG_IHO":  "MDG-IHO",
    "OU_MDG_ASO":  "MDG-ASO",
    "OU_MDG_AND":  "MDG_AND",
    "OU_MDG_AAN":  "MDG-AAN",
}

# Population par région (pour calcul taux incidence)
POPULATION_REGIONS: Dict[str, int] = {
    "MDG-ANA": 3800000, "MDG-VAK": 1850000, "MDG-ITM": 750000,
    "MDG-BMT": 500000,  "MDG-MAT": 1600000, "MDG-ATI": 850000,
    "MDG-VAT": 1100000, "MDG-FIT": 900000,  "MDG-ANO": 1050000,
    "MDG-ATS": 1450000, "MDG-ANA2":1050000, "MDG-ALA": 1050000,
    "MDG-BOE": 850000,  "MDG-SOF": 1100000, "MDG-MEN": 300000,
    "MDG-MEN2": 550000, "MDG-DIA": 650000,  "MDG-SAV": 1000000,
    "MDG-IHO": 350000,  "MDG-ASO": 1250000, "MDG_AND": 750000,
    "MDG-AAN": 650000,
}

# IDs DataElements DHIS2 (à adapter selon instance Madagascar)
DHIS2_DATA_ELEMENTS = {
    "cas_confirmes_pf": "jt8mzqlDEjd",
    "cas_confirmes_pv": "ImgnHPhcNYE",
    "cas_confirmes_mixte": "HUPFagklWaN",
    "tests_malaria": "qdjVZojEK8S",
    "tdr_positifs": "wZwzzRnr9N4",
    "tdr_negatifs": "Qk9nnX0i7lZ",
    "hospitalisations": "p4K11MFEWtw",
    "deces": "r6nrJANOqMw"
}
# IDs indicateurs WHO GHO pour Madagascar
WHO_GHO_INDICATORS = {
    "incidence_pour_1000":    "MALARIA_ESTIMATED_INCIDENCE",
    "mortalite_pour_100000":  "MALARIA_ESTIMATED_MORTALITY",
    "prevalence_pct":         "MALARIA_PF_PR_2_10",
    "couverture_milda_pct":   "MALARIA_ITNUSE_CHILDREN",
}


class DHIS2AuthError(Exception):
    pass

class DHIS2DataError(Exception):
    pass


class MalariaFetcher:
    """
    Collecteur de données épidémiologiques paludisme.

    Hiérarchie des sources :
      1. DHIS2 Madagascar (données fraiches, district-level)
      2. WHO GHO (données consolidées nationales)
      3. Données synthétiques estimées (dernier recours)
    """

    def __init__(self):
        self._dhis2_url  = settings.health_api.dhis2_base_url
        self._dhis2_user = settings.health_api.dhis2_username
        self._dhis2_pass = settings.health_api.dhis2_password
        self._who_url    = settings.health_api.who_gho_base_url
        self._session: Optional[aiohttp.ClientSession] = None

    # ─────────────────────────────────────────────
    # Session HTTP
    # ─────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60, connect=15)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Accept": "application/json"},
            )
        return self._session

    def _dhis2_auth_header(self) -> str:
        credentials = f"{self._dhis2_user}:{self._dhis2_pass}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ─────────────────────────────────────────────
    # DHIS2 — Cas hebdomadaires par région
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        retry=retry_if_exception_type(aiohttp.ClientError),
        reraise=True,
    )
    async def get_cas_dhis2(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
    ) -> List[Dict[str, Any]]:
        """
        Récupère les cas de paludisme depuis DHIS2.
        Retourne une liste de points hebdomadaires.
        """
        # Trouve le code OrgUnit DHIS2 correspondant à la région
        org_unit = self._region_to_orgunit(region_id)
        if not org_unit:
            logger.warning("Région {} non mappée vers DHIS2 OrgUnit", region_id)
            return []

        # Période DHIS2 en format ISO weeks : 2024W01, 2024W02…
        periodes = self._generer_periodes_semaines(date_debut, date_fin)

        try:
            raw = await self._dhis2_analytics_query(
                org_units=[org_unit],
                data_elements=list(DHIS2_DATA_ELEMENTS.values()),
                periodes=periodes,
            )
            records = self._parse_dhis2_response(raw, region_id)
            logger.info(
                "DHIS2 OK — {} : {} semaines récupérées",
                region_id, len(records)
            )
            return records

        except DHIS2AuthError:
            logger.error("Authentification DHIS2 échouée — vérifier credentials")
            return []
        except Exception as exc:
            logger.warning("DHIS2 échoué pour {} : {} — fallback WHO GHO", region_id, exc)
            return await self.get_cas_who_gho(region_id, date_debut, date_fin)

    async def _dhis2_analytics_query(
        self,
        org_units: List[str],
        data_elements: List[str],
        periodes: List[str],
    ) -> Dict:
        """Requête DHIS2 Analytics API."""
        session = await self._get_session()

        # Encodage dimension DHIS2
        dx = ";".join(data_elements)
        pe = ";".join(periodes)
        ou = ";".join(org_units)

        params = {
            "dimension": [f"dx:{dx}", f"pe:{pe}", f"ou:{ou}"],
            "displayProperty": "NAME",
            "includeNumDen": "false",
            "skipMeta": "false",
            "paging": "false",
        }

        url = f"{self._dhis2_url}/analytics"
        headers = {"Authorization": self._dhis2_auth_header()}

        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status == 401:
                raise DHIS2AuthError("Credentials DHIS2 invalides")
            if resp.status == 404:
                raise DHIS2DataError(f"Endpoint DHIS2 introuvable : {url}")
            resp.raise_for_status()
            return await resp.json()

    def _parse_dhis2_response(
        self, raw: Dict, region_id: str
    ) -> List[Dict[str, Any]]:
        """
        Transforme la réponse DHIS2 Analytics (format pivot) en liste de records.
        Format DHIS2 : rows = [dx, pe, ou, value]
        """
        headers = [h["name"] for h in raw.get("headers", [])]
        rows    = raw.get("rows", [])
        meta    = raw.get("metaData", {})

        # Index des colonnes
        try:
            idx_dx  = headers.index("dx")
            idx_pe  = headers.index("pe")
            idx_val = headers.index("value")
        except ValueError:
            logger.error("Format DHIS2 inattendu : colonnes manquantes")
            return []

        # Reverse mapping DataElement ID → nom lisible
        de_reverse = {v: k for k, v in DHIS2_DATA_ELEMENTS.items()}

        # Agrégation par période
        from collections import defaultdict
        agg: Dict[str, Dict[str, float]] = defaultdict(dict)

        for row in rows:
            de_id  = row[idx_dx]
            period = row[idx_pe]  # format : 2024W01
            try:
                val = float(row[idx_val])
            except (ValueError, TypeError):
                val = 0.0
            de_name = de_reverse.get(de_id, de_id)
            agg[period][de_name] = val

        population = POPULATION_REGIONS.get(region_id, 1_000_000)
        records = []

        for period_str, vals in sorted(agg.items()):
            # Parse période DHIS2 (ex: "2024W05")
            try:
                annee = int(period_str[:4])
                semaine = int(period_str[5:])
                date_rapport = date.fromisocalendar(annee, semaine, 1)
            except (ValueError, IndexError):
                continue

            cas_conf_tdr   = vals.get("cas_confirmes_pf", 0)
            cas_conf_micro = vals.get("cas_confirmes_pv", 0)
            cas_confirmes  = cas_conf_tdr + cas_conf_micro
            cas_confirmes_mixte   = vals.get("cas_confirmes_mixte", 0)
            deces          = vals.get("deces", 0)
            hospitalisations = vals.get("hospitalisations", 0)
            tests_malaria  = vals.get("tests_malaria", 0)
            tdr_positifs   = vals.get("tdr_positifs", 0)

            taux_positivite = (
                round(tdr_positifs / tests_malaria * 100, 2)
                if tests_malaria > 0 else 0.0
            )
            taux_incidence = round(
                cas_confirmes / population * 1000, 4
            )

            records.append({
                "region_id": region_id,
                "region_name": self._region_id_to_name(region_id),
                "district": None,
                "semaine_epidemio": semaine,
                "annee": annee,
                "date_rapport": str(date_rapport),
                "cas_confirmes": int(cas_confirmes),
                "cas_confirmes_mixte": int(cas_confirmes_mixte),
                "deces": int(deces),
                "hospitalisations": int(hospitalisations),
                "taux_incidence_pour_mille": taux_incidence,
                "taux_positivite_tdr_pct": taux_positivite,
                "tests_malaria": int(tests_malaria),
                "population_a_risque": population,
                "source": "DHIS2",
                "fiabilite_donnees": "confirmée",
                "period_dhis2": period_str,
            })

        return records

    # ─────────────────────────────────────────────
    # DHIS2 — Collecte toutes les régions (batch)
    # ─────────────────────────────────────────────

    async def get_cas_toutes_regions(
        self,
        date_debut: date,
        date_fin: date,
        concurrency: int = 4,
    ) -> Dict[str, List[Dict]]:
        """Collecte les cas de paludisme pour toutes les 22 régions."""
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_one(region_id: str):
            async with semaphore:
                try:
                    data = await self.get_cas_dhis2(region_id, date_debut, date_fin)
                    return region_id, data
                except Exception as exc:
                    logger.error("Erreur collecte malaria {} : {}", region_id, exc)
                    return region_id, []

        from src.utils.constants import REGIONS_MADAGASCAR
        tasks = [fetch_one(rid) for rid in REGIONS_MADAGASCAR]
        pairs = await asyncio.gather(*tasks)
        result = dict(pairs)

        total_records = sum(len(v) for v in result.values())
        logger.info(
            "Collecte malaria batch — {} régions, {} records total",
            len(result), total_records
        )
        return result

    # ─────────────────────────────────────────────
    # DHIS2 — Données districts (granularité fine)
    # ─────────────────────────────────────────────

    async def get_cas_districts(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
    ) -> List[Dict[str, Any]]:
        """
        Collecte les cas par district (niveau sous-régional).
        Permet d'identifier les foyers épidémiques locaux.
        """
        org_unit = self._region_to_orgunit(region_id)
        if not org_unit:
            return []

        periodes = self._generer_periodes_semaines(date_debut, date_fin)

        try:
            # Niveau CHILDREN dans DHIS2 = districts de la région
            raw = await self._dhis2_analytics_query(
                org_units=[f"{org_unit};LEVEL-3"],  # Niveau district
                data_elements=[
                    DHIS2_DATA_ELEMENTS["cas_confirmes_pf"],
                    DHIS2_DATA_ELEMENTS["deces"],
                ],
                periodes=periodes,
            )
            return self._parse_dhis2_response(raw, region_id)
        except Exception as exc:
            logger.warning("DHIS2 districts {} échoué : {}", region_id, exc)
            return []

    # ─────────────────────────────────────────────
    # WHO GHO — Données consolidées nationales
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=20),
        reraise=True,
    )
    async def get_cas_who_gho(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
    ) -> List[Dict[str, Any]]:
        """
        Récupère les données consolidées depuis WHO GHO.
        Données nationales agrégées — moins précises que DHIS2 mais plus robustes.
        """
        annee_debut = date_debut.year
        annee_fin   = date_fin.year

        try:
            session = await self._get_session()
            records_by_year = {}

            for indicator_name, indicator_code in WHO_GHO_INDICATORS.items():
                url = (
                    f"{self._who_url}/{indicator_code}"
                    f"?$filter=SpatialDimType eq 'COUNTRY' and SpatialDim eq 'MDG'"
                    f"&$select=TimeDim,NumericValue,Low,High"
                )
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data.get("value", []):
                            year = int(item.get("TimeDim", 0))
                            if annee_debut <= year <= annee_fin:
                                if year not in records_by_year:
                                    records_by_year[year] = {}
                                records_by_year[year][indicator_name] = {
                                    "valeur": item.get("NumericValue"),
                                    "borne_basse": item.get("Low"),
                                    "borne_haute": item.get("High"),
                                }

            # Convertit en format standardisé (annuel → synthétique hebdomadaire)
            records = []
            population = POPULATION_REGIONS.get(region_id, 1_000_000)

            for annee, indicateurs in sorted(records_by_year.items()):
                incidence = indicateurs.get("incidence_pour_1000", {}).get("valeur")
                if incidence is not None:
                    # Estimation hebdomadaire à partir de l'annuel
                    cas_annuels_estimes = int(incidence * population / 1000)
                    cas_par_semaine = cas_annuels_estimes // 52
                    for semaine in range(1, 53):
                        records.append({
                            "region_id": region_id,
                            "region_name": self._region_id_to_name(region_id),
                            "semaine_epidemio": semaine,
                            "annee": annee,
                            "date_rapport": str(date.fromisocalendar(annee, semaine, 1)),
                            "cas_confirmes": cas_par_semaine,
                            "cas_confirmes_mixte": int(cas_par_semaine * 1.3),
                            "deces": 0,
                            "hospitalisations": int(cas_par_semaine * 0.05),
                            "taux_incidence_pour_mille": round(incidence / 52, 4),
                            "taux_positivite_tdr_pct": 0.0,
                            "population_a_risque": population,
                            "source": "WHO GHO",
                            "fiabilite_donnees": "estimée",
                            "indicateurs_who": indicateurs,
                        })

            logger.info("WHO GHO {} : {} records pour années {}-{}",
                        region_id, len(records), annee_debut, annee_fin)
            return records

        except Exception as exc:
            logger.error("WHO GHO échoué pour {} : {}", region_id, exc)
            return self._donnees_synthetiques(region_id, date_debut, date_fin)

    # ─────────────────────────────────────────────
    # Indicateurs épidémiologiques dérivés
    # ─────────────────────────────────────────────

    def calculer_alertes(
        self,
        records: List[Dict],
        region_id: str,
    ) -> List[Dict]:
        """
        Détecte les alertes épidémiologiques sur une série temporelle :
          1. Dépassement seuil absolu (taux positivité TDR > 40%)
          2. Tendance hausse rapide (doublement en 2 semaines)
          3. Dépassement seuil incidence (> 5/1000/semaine)
          4. Détection cluster (variation > 3σ)
        """
        if len(records) < 4:
            return []

        alertes = []
        import statistics

        cas_list = [r.get("cas_confirmes", 0) for r in records]
        tdr_list  = [r.get("taux_positivite_tdr_pct", 0) for r in records]

        # Statistiques de base
        mean_cas = statistics.mean(cas_list) if cas_list else 0
        std_cas  = statistics.stdev(cas_list) if len(cas_list) > 1 else 0

        for i, record in enumerate(records):
            cas_this_week = record.get("cas_confirmes", 0)
            tdr_this_week = record.get("taux_positivite_tdr_pct", 0)
            date_rec      = record.get("date_rapport", "")

            # 1. Seuil TDR
            if tdr_this_week > 40:
                alertes.append(self._creer_alerte(
                    region_id=region_id,
                    type_alerte="seuil_tdr_depasse",
                    severite="urgence" if tdr_this_week > 60 else "alerte",
                    valeur=tdr_this_week,
                    seuil=40.0,
                    date=date_rec,
                    description=f"Taux positivité TDR : {tdr_this_week:.1f}% (seuil 40%)",
                ))

            # 2. Doublement en 2 semaines
            if i >= 2:
                cas_2sem_avant = records[i - 2].get("cas_confirmes", 0)
                if cas_2sem_avant > 0 and cas_this_week >= cas_2sem_avant * 2:
                    alertes.append(self._creer_alerte(
                        region_id=region_id,
                        type_alerte="tendance_hausse_rapide",
                        severite="urgence",
                        valeur=cas_this_week,
                        seuil=cas_2sem_avant,
                        date=date_rec,
                        description=(
                            f"Doublement des cas en 2 semaines : "
                            f"{cas_2sem_avant} → {cas_this_week}"
                        ),
                    ))

            # 3. Seuil incidence
            taux_inc = record.get("taux_incidence_pour_mille", 0)
            if taux_inc > 5.0:
                alertes.append(self._creer_alerte(
                    region_id=region_id,
                    type_alerte="seuil_depasse",
                    severite="alerte",
                    valeur=taux_inc,
                    seuil=5.0,
                    date=date_rec,
                    description=f"Taux d'incidence : {taux_inc:.2f}/1000 (seuil 5/1000)",
                ))

            # 4. Détection anomalie statistique (3σ)
            if std_cas > 0 and (cas_this_week - mean_cas) / std_cas > 3.0:
                alertes.append(self._creer_alerte(
                    region_id=region_id,
                    type_alerte="anomalie_cluster",
                    severite="urgence",
                    valeur=cas_this_week,
                    seuil=mean_cas + 3 * std_cas,
                    date=date_rec,
                    description=(
                        f"Anomalie statistique (> 3σ) : {cas_this_week} cas "
                        f"(moyenne {mean_cas:.0f} ± {std_cas:.0f})"
                    ),
                ))

        # Déduplique les alertes (garde la plus sévère par semaine)
        alertes = self._deduplicer_alertes(alertes)
        logger.info("{} alertes détectées pour {}", len(alertes), region_id)
        return alertes

    def _creer_alerte(
        self,
        region_id: str,
        type_alerte: str,
        severite: str,
        valeur: float,
        seuil: float,
        date: str,
        description: str,
    ) -> Dict:
        import uuid
        from src.utils.constants import REGION_NAMES  # noqa

        severite_actions = {
            "surveillance": [
                "Surveillance de routine renforcée",
                "Rapport hebdomadaire au niveau régional",
            ],
            "alerte": [
                "Notification au médecin inspecteur régional",
                "Vérification et renforcement stocks TDR/ACT",
                "Mobilisation agents de santé communautaires",
            ],
            "urgence": [
                "Notification IMMÉDIATE au niveau national",
                "Activation plan riposte épidémique",
                "Demande renforts matériels et humains",
                "Information des partenaires (UNICEF, OMS)",
            ],
            "crise": [
                "Déclaration état d'urgence sanitaire",
                "Activation cellule de crise nationale",
                "Appel à l'aide internationale",
            ],
        }

        return {
            "alerte_id": str(uuid.uuid4()),
            "region_id": region_id,
            "region_name": self._region_id_to_name(region_id),
            "type_alerte": type_alerte,
            "severite": severite,
            "seuil_depasse": seuil,
            "valeur_actuelle": valeur,
            "date_detection": datetime.utcnow().isoformat(),
            "statut": "active",
            "description": description,
            "actions_requises": severite_actions.get(severite, []),
            "responsable_notification": self._get_responsable(region_id),
        }

    def _deduplicer_alertes(self, alertes: List[Dict]) -> List[Dict]:
        """Garde les alertes uniques par (region, date, type) — severite max."""
        from collections import defaultdict
        severity_rank = {"surveillance": 0, "alerte": 1, "urgence": 2, "crise": 3}

        grouped: Dict[str, Dict] = {}
        for a in alertes:
            key = f"{a['region_id']}:{a.get('date_detection','')[:10]}:{a['type_alerte']}"
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = a
            else:
                if (severity_rank.get(a["severite"], 0)
                        > severity_rank.get(existing["severite"], 0)):
                    grouped[key] = a

        return list(grouped.values())

    # ─────────────────────────────────────────────
    # Agrégation et indicateurs dérivés
    # ─────────────────────────────────────────────

    def calculer_tendances(
        self, records: List[Dict], fenetre: int = 4
    ) -> Dict[str, Any]:
        """
        Calcule les tendances sur les dernières `fenetre` semaines :
          - Variation relative vs fenêtre précédente
          - Tendance (hausse/stable/baisse)
          - Taux de croissance hebdomadaire
        """
        if len(records) < fenetre * 2:
            return {"tendance": "données insuffisantes"}

        recent   = records[-fenetre:]
        previous = records[-fenetre * 2:-fenetre]

        cas_recent   = sum(r.get("cas_confirmes", 0) for r in recent)
        cas_previous = sum(r.get("cas_confirmes", 0) for r in previous)

        if cas_previous == 0:
            variation_pct = 0.0
        else:
            variation_pct = (cas_recent - cas_previous) / cas_previous * 100

        if variation_pct > 15:
            tendance = "hausse"
        elif variation_pct < -15:
            tendance = "baisse"
        else:
            tendance = "stable"

        # Taux de croissance hebdomadaire moyen (CAGR-like)
        cas_semaine_1 = recent[0].get("cas_confirmes", 0)
        cas_semaine_n = recent[-1].get("cas_confirmes", 0)
        if cas_semaine_1 > 0 and fenetre > 1:
            taux_croissance = (
                (cas_semaine_n / cas_semaine_1) ** (1 / (fenetre - 1)) - 1
            ) * 100
        else:
            taux_croissance = 0.0

        return {
            "tendance": tendance,
            "variation_pct_vs_periode_prec": round(variation_pct, 1),
            "taux_croissance_hebdo_pct": round(taux_croissance, 2),
            "cas_total_recent": cas_recent,
            "cas_total_precedent": cas_previous,
            "fenetre_semaines": fenetre,
        }

    def calculer_saisonnalite(
        self,
        records: List[Dict],
        region_id: str,
    ) -> Dict[str, Any]:
        """
        Analyse la position dans le cycle saisonnier du paludisme.
        """
        from collections import defaultdict
        from src.utils.constants import get_saison_courante

        semaine_actuelle = date.today().isocalendar()[1]
        mois_actuel      = date.today().month
        saison_actuelle  = get_saison_courante(mois_actuel)

        # Pic historique par semaine épidémio
        cas_par_semaine: Dict[int, List[int]] = defaultdict(list)
        for r in records:
            sem = r.get("semaine_epidemio", 0)
            cas = r.get("cas_confirmes", 0)
            if 1 <= sem <= 52:
                cas_par_semaine[sem].append(cas)

        # Semaine du pic historique moyen
        moyenne_par_semaine = {
            sem: sum(vals) / len(vals)
            for sem, vals in cas_par_semaine.items()
        }
        semaine_pic = max(
            moyenne_par_semaine, key=moyenne_par_semaine.get, default=10
        )

        semaines_avant_pic = (semaine_pic - semaine_actuelle) % 52

        return {
            "region_id": region_id,
            "saison_courante": saison_actuelle.value,
            "semaine_dans_saison": semaine_actuelle,
            "pic_historique_semaine": semaine_pic,
            "semaines_avant_pic_estime": semaines_avant_pic,
            "cas_cumules_saison": sum(
                r.get("cas_confirmes", 0) for r in records[-13:]
            ),
            "cas_cumules_saison_precedente": sum(
                r.get("cas_confirmes", 0) for r in records[-65:-52]
            ) if len(records) >= 65 else 0,
            "variation_pct": 0.0,
            "tendance": "données insuffisantes",
        }

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _region_to_orgunit(region_id: str) -> Optional[str]:
        reverse = {v: k for k, v in DHIS2_ORG_UNIT_MAP.items()}
        return reverse.get(region_id)

    @staticmethod
    def _region_id_to_name(region_id: str) -> str:
        NAMES = {
            "MDG-ANA": "Analamanga", "MDG-ATS": "Atsinanana",
            "MDG-BOE": "Boeny",      "MDG_AND": "Androy",
            "MDG-ASO": "Atsimo-Andrefana",
        }
        return NAMES.get(region_id, region_id)

    @staticmethod
    def _get_responsable(region_id: str) -> str:
        """Retourne le responsable de notification selon la région."""
        return f"Médecin Inspecteur Régional — {region_id}"

    @staticmethod
    def _generer_periodes_semaines(
        date_debut: date, date_fin: date
    ) -> List[str]:
        """Génère la liste des semaines ISO en format DHIS2 (ex: '2024W01')."""
        periodes = []
        current = date_debut
        while current <= date_fin:
            iso = current.isocalendar()
            periodes.append(f"{iso[0]}W{iso[1]:02d}")
            current += timedelta(weeks=1)
        return list(dict.fromkeys(periodes))  # déduplique

    def _donnees_synthetiques(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
    ) -> List[Dict[str, Any]]:
        """
        Génère des données synthétiques basées sur l'endémicité connue.
        Utilisé en dernier recours quand toutes les sources échouent.
        """
        from src.utils.constants import REGIONS_MADAGASCAR
        import json
        from pathlib import Path

        logger.warning("Utilisation données synthétiques pour {}", region_id)

        # Endémicité par défaut selon metadata
        endemicite_cas_base = {
            "low":       2,
            "medium":    8,
            "high":      25,
            "very_high": 60,
        }

        try:
            with Path("config/regions_metadata.json").open() as f:
                meta = json.load(f)
            region_meta = next(
                (r for r in meta["regions"] if r["id"] == region_id), {}
            )
            endemicite = region_meta.get("malaria_endemicity", "medium")
        except Exception:
            endemicite = "medium"

        cas_base = endemicite_cas_base.get(endemicite, 8)
        population = POPULATION_REGIONS.get(region_id, 500_000)
        records = []

        periodes = self._generer_periodes_semaines(date_debut, date_fin)
        for period_str in periodes:
            try:
                annee = int(period_str[:4])
                semaine = int(period_str[5:])
                dt = date.fromisocalendar(annee, semaine, 1)
            except ValueError:
                continue

            # Saisonnalité synthétique : pic en semaines 1-15 et 48-52
            import math
            facteur_saisonnier = 1.0 + 0.8 * abs(
                math.sin(math.pi * semaine / 52)
            )
            cas = int(cas_base * facteur_saisonnier * (population / 100_000))

            records.append({
                "region_id": region_id,
                "region_name": self._region_id_to_name(region_id),
                "semaine_epidemio": semaine,
                "annee": annee,
                "date_rapport": str(dt),
                "cas_confirmes": cas,
                "cas_confirmes_mixte": int(cas * 1.5),
                "deces": max(0, int(cas * 0.005)),
                "hospitalisations": int(cas * 0.05),
                "taux_incidence_pour_mille": round(cas / population * 1000, 4),
                "taux_positivite_tdr_pct": min(80, cas_base * 2.5),
                "population_a_risque": population,
                "source": "Synthétique (tous services indisponibles)",
                "fiabilite_donnees": "estimée",
            })

        return records