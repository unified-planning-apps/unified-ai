"""
Collecte de données nutrition et sécurité alimentaire :
  - FAO FAOSTAT API : production agricole, disponibilité alimentaire
  - WFP VAM API : Food Consumption Score, prix denrées, marchés
  - UNICEF MICS / SMART surveys (import fichiers)
  - Calcul automatique des indicateurs dérivés (FCS, HDDS, rCSI)
  - Détection périodes de soudure
  - Gestion robuste des données manquantes (Madagascar = connectivité limitée)
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta
from pathlib import Path
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
# Configuration par région : cultures principales et saisonnalité
# ─────────────────────────────────────────────────────────────────
REGION_FOOD_PROFILE: Dict[str, Dict] = {
    "MDG-ANA":  {
        "cultures_principales": ["riz", "manioc", "pomme_de_terre"],
        "mois_recolte_principale": [3, 4, 5],
        "mois_soudure": [10, 11, 12],
        "indice_vulnerabilite": 0.3,
    },
    "MDG_AND":  {
        "cultures_principales": ["manioc", "mais", "haricots"],
        "mois_recolte_principale": [4, 5],
        "mois_soudure": [11, 12, 1, 2, 3],
        "indice_vulnerabilite": 0.9,  # Grand Sud — très vulnérable
    },
    "MDG-ASO":  {
        "cultures_principales": ["mais", "manioc", "patate_douce"],
        "mois_recolte_principale": [4, 5],
        "mois_soudure": [11, 12, 1],
        "indice_vulnerabilite": 0.8,
    },
    "MDG-IHO":  {
        "cultures_principales": ["zebu", "mais", "riz"],
        "mois_recolte_principale": [3, 4],
        "mois_soudure": [10, 11],
        "indice_vulnerabilite": 0.6,
    },
    "MDG-ATS":  {
        "cultures_principales": ["riz", "girofle", "poivre", "vanille"],
        "mois_recolte_principale": [4, 5, 6],
        "mois_soudure": [11, 12],
        "indice_vulnerabilite": 0.4,
    },
}

# Défaut pour les régions non profilees
DEFAULT_FOOD_PROFILE = {
    "cultures_principales": ["riz", "manioc"],
    "mois_recolte_principale": [4, 5],
    "mois_soudure": [11, 12],
    "indice_vulnerabilite": 0.5,
}

# Groupes alimentaires HDDS (Household Dietary Diversity Score)
GROUPES_ALIMENTAIRES_HDDS = [
    "cereales",         # Riz, maïs, blé
    "racines_tubercules", # Manioc, patate douce
    "legumineuses",     # Haricots, lentilles
    "legumes",          # Légumes verts
    "fruits",           # Fruits frais
    "viande_volaille",  # Viande, abats, volaille
    "oeufs",            # Œufs
    "poisson",          # Poisson, fruits de mer
    "produits_laitiers",# Lait, fromage, yaourt
    "huiles_graisses",  # Huile de palme, graisses
    "sucres",           # Sucre, miel
    "divers",           # Épices, condiments, café
]

# Pondération FCS (Food Consumption Score) WFP
FCS_WEIGHTS = {
    "cereales": 2, "racines_tubercules": 2, "legumineuses": 3,
    "legumes": 1, "fruits": 1, "viande_volaille": 4,
    "oeufs": 4, "poisson": 4, "produits_laitiers": 4,
    "huiles_graisses": 0.5, "sucres": 0.5, "divers": 0,
}

# IDs WFP VAM pour Madagascar
WFP_COUNTRY_CODE = "MG"
WFP_MARKET_IDS: Dict[str, List[str]] = {
    "MDG-ANA":  ["MDG001", "MDG002"],  # Antananarivo
    "MDG_AND":  ["MDG025"],            # Ambovombe
    "MDG-ASO":  ["MDG030"],            # Toliara
    "MDG-ATS":  ["MDG010"],            # Toamasina
}

# Codes FAO pour Madagascar et indicateurs
FAO_COUNTRY_CODE = "701"  # Madagascar
FAO_FOOD_ITEMS = {
    "riz_paddy":       "27",
    "manioc":          "125",
    "mais":            "56",
    "patate_douce":    "122",
    "haricots_secs":   "176",
    "arachides":       "234",
}


class NutritionFetcher:
    """
    Collecteur de données nutrition et sécurité alimentaire.

    Ordre de priorité :
      1. WFP VAM (marchés, FCS, prix) — mise à jour bimensuelle
      2. FAO FAOSTAT (production agricole) — annuel
      3. Profils régionaux internes (fallback)
      4. Valeurs synthétiques estimées
    """

    def __init__(self):
        self._fao_url  = settings.nutrition_api.fao_base_url
        self._wfp_url  = settings.nutrition_api.wfp_base_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=45, connect=15)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "UNICEF-Madagascar-Nutrition/1.0",
                },
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ─────────────────────────────────────────────
    # WFP VAM — Food Consumption Score & prix
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(aiohttp.ClientError),
        reraise=True,
    )
    async def get_food_consumption(
        self, region_id: str
    ) -> Dict[str, Any]:
        """
        Récupère le Food Consumption Score (FCS) et la diversité alimentaire
        depuis WFP VAM Data Bridges API.

        FCS > 42 : consommation acceptable
        FCS 21-42 : consommation limite
        FCS < 21 : consommation pauvre
        """
        try:
            session = await self._get_session()

            # Endpoint WFP food security indicators
            url = f"{self._wfp_url}/FoodSecurity/FoodConsumption"
            params = {
                "CountryCode": WFP_COUNTRY_CODE,
                "PageSize": 50,
            }
            async with session.get(url, params=params) as resp:
                if resp.status in (401, 403):
                    logger.warning("WFP API auth échouée — clé API requise")
                    return await self._fcs_from_profile(region_id)
                resp.raise_for_status()
                data = await resp.json()

            return self._parse_wfp_fcs(data, region_id)

        except Exception as exc:
            logger.warning("WFP FCS échoué pour {} : {}", region_id, exc)
            return await self._fcs_from_profile(region_id)

    def _parse_wfp_fcs(self, raw: Dict, region_id: str) -> Dict:
        """Parse la réponse WFP et calcule le FCS/HDDS."""
        items = raw.get("items", raw.get("data", []))

        # Filtre les données pour Madagascar
        mdg_data = [
            item for item in items
            if item.get("countryCode", "") == WFP_COUNTRY_CODE
        ]

        if not mdg_data:
            return self._fcs_synthetique(region_id)

        # Dernière observation disponible
        latest = sorted(
            mdg_data,
            key=lambda x: x.get("surveyDate", ""),
            reverse=True,
        )[0]

        fcs = latest.get("fcsMean", 0) or latest.get("fcs", 0)
        return {
            "region_id": region_id,
            "date_observation": datetime.utcnow().date().isoformat(),
            "score_fcs": round(float(fcs), 1),
            "classification_fcs": self._classifier_fcs(fcs),
            "hdds": latest.get("hdds", 5.0),
            "rcsi": latest.get("rcsi", 0),
            "source": "WFP VAM",
        }

    async def _fcs_from_profile(self, region_id: str) -> Dict:
        """Estime le FCS à partir du profil régional et de la saison."""
        profile = REGION_FOOD_PROFILE.get(region_id, DEFAULT_FOOD_PROFILE)
        mois_actuel = date.today().month

        # FCS plus bas en période de soudure
        en_soudure = mois_actuel in profile.get("mois_soudure", [])
        vulnerabilite = profile.get("indice_vulnerabilite", 0.5)

        fcs_base = 50.0 * (1 - vulnerabilite)
        if en_soudure:
            fcs_base *= 0.65  # -35% pendant la soudure

        return {
            "region_id": region_id,
            "date_observation": date.today().isoformat(),
            "score_fcs": round(max(5, fcs_base), 1),
            "classification_fcs": self._classifier_fcs(fcs_base),
            "hdds": round(6 * (1 - vulnerabilite * 0.5), 1),
            "rcsi": int(vulnerabilite * 20),
            "en_periode_soudure": en_soudure,
            "source": "Profil régional (fallback)",
        }

    # ─────────────────────────────────────────────
    # WFP — Prix des denrées alimentaires
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        reraise=True,
    )
    async def get_prix_denrees(
        self,
        region_id: str,
        mois: int = 3,
    ) -> Dict[str, Any]:
        """
        Récupère les prix des denrées de base sur les marchés locaux.
        Source : WFP VAM Data Bridges — Commodity Prices.
        """
        try:
            session = await self._get_session()
            date_fin = date.today()
            date_debut = date_fin - timedelta(days=mois * 30)

            url = f"{self._wfp_url}/MarketPrices/PriceMonthly"
            params = {
                "CountryCode": WFP_COUNTRY_CODE,
                "startDate": date_debut.strftime("%Y-%m-%d"),
                "endDate":   date_fin.strftime("%Y-%m-%d"),
                "PageSize":  200,
            }

            async with session.get(url, params=params) as resp:
                if resp.status in (401, 403):
                    return self._prix_synthetiques(region_id)
                resp.raise_for_status()
                raw = await resp.json()

            return self._parse_prix_wfp(raw, region_id)

        except Exception as exc:
            logger.warning("WFP prix échoué pour {} : {}", region_id, exc)
            return self._prix_synthetiques(region_id)

    def _parse_prix_wfp(self, raw: Dict, region_id: str) -> Dict:
        """Parse les prix WFP et calcule les variations."""
        items = raw.get("items", raw.get("data", []))

        prix: Dict[str, List[float]] = {}
        for item in items:
            commodity = item.get("commodityName", "").lower()
            price     = item.get("price")
            if price and commodity:
                key = self._normaliser_commodity(commodity)
                if key:
                    prix.setdefault(key, []).append(float(price))

        # Calcul prix moyens et variation
        prix_moyens = {k: sum(v) / len(v) for k, v in prix.items() if v}

        # Variation par rapport au mois précédent (si 2 mois de données)
        prix_1m = {k: v[-1] for k, v in prix.items() if v}
        prix_2m = {k: v[0] for k, v in prix.items() if len(v) > 1}

        variation_pct = None
        if prix_1m and prix_2m:
            prix_panier_actuel = sum(prix_1m.values())
            prix_panier_prec   = sum(prix_2m.values())
            if prix_panier_prec > 0:
                variation_pct = round(
                    (prix_panier_actuel - prix_panier_prec) / prix_panier_prec * 100, 1
                )

        return {
            "region_id": region_id,
            "date_observation": date.today().isoformat(),
            "prix_riz_kg":       prix_moyens.get("riz"),
            "prix_manioc_kg":    prix_moyens.get("manioc"),
            "prix_mais_kg":      prix_moyens.get("mais"),
            "prix_haricots_kg":  prix_moyens.get("haricots"),
            "prix_huile_litre":  prix_moyens.get("huile"),
            "variation_prix_pct_1m": variation_pct,
            "source": "WFP VAM",
        }

    @staticmethod
    def _normaliser_commodity(nom: str) -> Optional[str]:
        """Normalise les noms de denrées WFP vers le format interne."""
        mappings = {
            "rice": "riz", "paddy": "riz", "riz": "riz",
            "cassava": "manioc", "manioc": "manioc",
            "maize": "mais", "corn": "mais", "mais": "mais",
            "beans": "haricots", "haricots": "haricots",
            "oil": "huile", "cooking oil": "huile", "huile": "huile",
            "sugar": "sucre", "sucre": "sucre",
        }
        for pattern, normalized in mappings.items():
            if pattern in nom:
                return normalized
        return None

    # ─────────────────────────────────────────────
    # FAO FAOSTAT — Production agricole
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        reraise=True,
    )
    async def get_production_agricole(
        self,
        region_id: str,
        annee_debut: int,
        annee_fin: int,
    ) -> List[Dict[str, Any]]:
        """
        Récupère les données de production agricole depuis FAO FAOSTAT.
        Données nationales (Madagascar) — pas de granularité régionale dans FAOSTAT.
        """
        try:
            session = await self._get_session()

            items_codes = ",".join(FAO_FOOD_ITEMS.values())
            url = (
                f"{self._fao_url}/data/QCL"
                f"?area={FAO_COUNTRY_CODE}"
                f"&item={items_codes}"
                f"&element=5510"  # 5510 = Area harvested, 5312 = Production
                f"&year={annee_debut}:{annee_fin}"
                f"&outputType=objects"
            )

            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("FAO FAOSTAT indisponible ({})", resp.status)
                    return []
                data = await resp.json()

            return self._parse_fao_production(data, region_id)

        except Exception as exc:
            logger.warning("FAO FAOSTAT échoué pour {} : {}", region_id, exc)
            return []

    def _parse_fao_production(
        self, raw: Dict, region_id: str
    ) -> List[Dict]:
        """Parse la réponse FAOSTAT et structure les données de production."""
        records = []
        reverse_items = {v: k for k, v in FAO_FOOD_ITEMS.items()}

        for item in raw.get("data", []):
            item_code = str(item.get("Item Code", ""))
            item_name = reverse_items.get(item_code, item.get("Item", "inconnu"))
            annee = item.get("Year")
            valeur = item.get("Value")
            unite = item.get("Unit", "ha")

            if valeur is None:
                continue

            records.append({
                "region_id": region_id,
                "annee": annee,
                "culture": item_name,
                "valeur": float(valeur),
                "unite": unite,
                "element": item.get("Element", "Production"),
                "source": "FAO FAOSTAT",
            })

        logger.debug(
            "FAO FAOSTAT {} : {} records de production", region_id, len(records)
        )
        return records

    # ─────────────────────────────────────────────
    # Statut nutritionnel (GAM / SAM / MAM)
    # ─────────────────────────────────────────────

    async def get_statut_nutritionnel(
        self, region_id: str
    ) -> Dict[str, Any]:
        """
        Retourne le statut nutritionnel actuel.
        Sources prioritaires : base UNICEF → WFP → profil régional.
        """
        # 1. Tentative lecture base UNICEF locale
        try:
            unicef_data = await self._get_unicef_survey_data(region_id)
            if unicef_data:
                logger.debug("Statut nutritionnel UNICEF local OK — {}", region_id)
                return unicef_data
        except Exception as exc:
            logger.debug("Données UNICEF locales non disponibles : {}", exc)

        # 2. Estimation à partir du profil régional + saison
        return self._estimer_statut_nutritionnel(region_id)

    async def _get_unicef_survey_data(self, region_id: str) -> Optional[Dict]:
        """
        Charge les données d'enquêtes SMART/MICS depuis la base locale.
        Fichiers JSON dans data/external/nutrition/
        """
        survey_path = Path("data/external/nutrition") / f"{region_id}_latest.json"
        if not survey_path.exists():
            return None

        with survey_path.open() as f:
            data = json.load(f)

        gam = data.get("gam_pct", 0)
        return {
            "region_id": region_id,
            "region_name": data.get("region_name", region_id),
            "date_enquete": data.get("date_enquete", str(date.today())),
            "source": data.get("source", "Enquête SMART"),
            "gam_pct": gam,
            "sam_pct": data.get("sam_pct", gam * 0.3),
            "mam_pct": data.get("mam_pct", gam * 0.7),
            "stunting_pct": data.get("stunting_pct", 0),
            "underweight_pct": data.get("underweight_pct", 0),
            "enfants_5ans_affectes": data.get("enfants_5ans_affectes", 0),
            "femmes_enceintes_malnutries": data.get("femmes_enceintes_malnutries", 0),
            "classification_who": self._classifier_gam(gam),
            "tendance_vs_periode_prec": data.get("tendance", "stable"),
            "fiabilite_donnees": "confirmée",
        }

    def _estimer_statut_nutritionnel(self, region_id: str) -> Dict:
        """
        Estimation du statut nutritionnel basée sur :
        - Profil de vulnérabilité régionale
        - Saison actuelle (soudure vs récolte)
        - Endémicité du paludisme (co-facteur de malnutrition)
        """
        profile = REGION_FOOD_PROFILE.get(region_id, DEFAULT_FOOD_PROFILE)
        vulnerabilite = profile.get("indice_vulnerabilite", 0.5)
        mois = date.today().month
        en_soudure = mois in profile.get("mois_soudure", [])

        # GAM de base selon vulnérabilité
        gam_base = 5.0 + vulnerabilite * 15.0
        if en_soudure:
            gam_base *= 1.4  # +40% en période de soudure

        gam = round(min(40, max(1, gam_base)), 1)
        sam = round(gam * 0.28, 1)
        mam = round(gam * 0.72, 1)

        # Population enfants < 5 ans (17% de la pop)
        from src.utils.constants import PCT_ENFANTS_MOINS_5ANS
        from src.data_collection.malaria_fetcher import POPULATION_REGIONS
        population = POPULATION_REGIONS.get(region_id, 500_000)
        enfants_5ans = int(population * PCT_ENFANTS_MOINS_5ANS)

        return {
            "region_id": region_id,
            "region_name": region_id,
            "date_enquete": str(date.today()),
            "source": "Estimation (profil régional)",
            "gam_pct": gam,
            "sam_pct": sam,
            "mam_pct": mam,
            "stunting_pct": round(gam * 2.5, 1),
            "underweight_pct": round(gam * 1.8, 1),
            "enfants_5ans_affectes": int(enfants_5ans * gam / 100),
            "femmes_enceintes_malnutries": int(population * 0.04 * gam / 100),
            "classification_who": self._classifier_gam(gam),
            "tendance_vs_periode_prec": "stable",
            "fiabilite_donnees": "estimée",
            "en_periode_soudure": en_soudure,
        }

    # ─────────────────────────────────────────────
    # Disponibilité alimentaire complète
    # ─────────────────────────────────────────────

    async def get_disponibilite_complete(
        self, region_id: str
    ) -> Dict[str, Any]:
        """
        Agrège FCS, prix, stocks et profil alimentaire en un seul objet.
        """
        fcs_data, prix_data = await asyncio.gather(
            self.get_food_consumption(region_id),
            self.get_prix_denrees(region_id),
            return_exceptions=True,
        )

        fcs  = fcs_data  if not isinstance(fcs_data, Exception)  else {}
        prix = prix_data if not isinstance(prix_data, Exception) else {}

        profile = REGION_FOOD_PROFILE.get(region_id, DEFAULT_FOOD_PROFILE)
        mois = date.today().month

        # Disponibilité par groupe (0=absent, 1=rare, 2=limité, 3=disponible)
        dispo = self._calculer_disponibilite_groupes(region_id, mois, profile)

        return {
            "region_id": region_id,
            "date_observation": str(date.today()),
            "score_fcs": fcs.get("score_fcs", 35.0),
            "classification_fcs": fcs.get("classification_fcs", "limite"),
            "hdds": fcs.get("hdds", 5.0),
            "rcsi": fcs.get("rcsi", 0),
            "prix_riz_kg":       prix.get("prix_riz_kg"),
            "prix_manioc_kg":    prix.get("prix_manioc_kg"),
            "prix_mais_kg":      prix.get("prix_mais_kg"),
            "prix_haricots_kg":  prix.get("prix_haricots_kg"),
            "prix_huile_litre":  prix.get("prix_huile_litre"),
            "variation_prix_pct_1m": prix.get("variation_prix_pct_1m"),
            **dispo,
            "source": "WFP VAM + Profil régional",
        }

    def _calculer_disponibilite_groupes(
        self, region_id: str, mois: int, profile: Dict
    ) -> Dict[str, int]:
        """
        Calcule la disponibilité de chaque groupe alimentaire (0-3)
        selon la saison et les cultures principales.
        """
        cultures = profile.get("cultures_principales", [])
        mois_recolte = profile.get("mois_recolte_principale", [4, 5])
        en_recolte = mois in mois_recolte
        en_soudure = mois in profile.get("mois_soudure", [11, 12])

        # Niveau de base selon saison
        niveau_base = 3 if en_recolte else (1 if en_soudure else 2)

        return {
            "disponibilite_cereales": (
                3 if "riz" in cultures or "mais" in cultures else niveau_base
            ),
            "disponibilite_legumineuses": (
                2 if "haricots" in cultures else max(1, niveau_base - 1)
            ),
            "disponibilite_proteines_animales": (
                2 if "zebu" in cultures else max(1, niveau_base - 1)
            ),
            "disponibilite_legumes": niveau_base,
            "disponibilite_fruits": max(1, niveau_base - (1 if en_soudure else 0)),
        }

    # ─────────────────────────────────────────────
    # Détection période de soudure
    # ─────────────────────────────────────────────

    async def get_statut_soudure(
        self, region_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Identifie les régions en période de soudure alimentaire.
        Période de soudure = intervalle entre épuisement des stocks
        de la récolte précédente et début de la nouvelle récolte.
        """
        from src.utils.constants import REGIONS_MADAGASCAR

        regions_a_traiter = (
            [region_id] if region_id else REGIONS_MADAGASCAR
        )
        mois_actuel = date.today().month
        resultats = []

        for rid in regions_a_traiter:
            profile = REGION_FOOD_PROFILE.get(rid, DEFAULT_FOOD_PROFILE)
            mois_soudure  = profile.get("mois_soudure", [])
            mois_recolte  = profile.get("mois_recolte_principale", [4, 5])
            vulnerabilite = profile.get("indice_vulnerabilite", 0.5)

            en_soudure = mois_actuel in mois_soudure

            # Calcul semaines avant soudure
            if not en_soudure:
                prochains_mois_soudure = [
                    m for m in mois_soudure if m > mois_actuel
                ]
                if not prochains_mois_soudure:
                    prochains_mois_soudure = [
                        m + 12 for m in mois_soudure
                    ]
                mois_debut_soudure = min(prochains_mois_soudure)
                semaines_avant = max(0, (mois_debut_soudure - mois_actuel) * 4)
            else:
                semaines_avant = None

            # Niveau de risque pendant la soudure
            if vulnerabilite >= 0.8:
                niveau_risque = "critique"
            elif vulnerabilite >= 0.6:
                niveau_risque = "élevé"
            elif vulnerabilite >= 0.4:
                niveau_risque = "modéré"
            else:
                niveau_risque = "faible"

            resultats.append({
                "region_id": rid,
                "region_name": rid,
                "en_periode_soudure": en_soudure,
                "semaines_avant_soudure": semaines_avant,
                "duree_soudure_historique_semaines": len(mois_soudure) * 4,
                "niveau_risque_soudure": niveau_risque,
                "denrees_principales_affectees": profile.get("cultures_principales", []),
                "strategies_coping_observees": self._get_strategies_coping(vulnerabilite),
            })

        return resultats

    # ─────────────────────────────────────────────
    # Collecte batch — toutes régions
    # ─────────────────────────────────────────────

    async def get_nutrition_toutes_regions(
        self, concurrency: int = 4
    ) -> Dict[str, Dict]:
        """Collecte le statut nutritionnel pour toutes les 22 régions."""
        from src.utils.constants import REGIONS_MADAGASCAR
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_one(rid: str):
            async with semaphore:
                try:
                    statut = await self.get_statut_nutritionnel(rid)
                    dispo  = await self.get_disponibilite_complete(rid)
                    return rid, {
                        "statut": statut,
                        "disponibilite": dispo,
                    }
                except Exception as exc:
                    logger.error("Erreur nutrition {} : {}", rid, exc)
                    return rid, {"erreur": str(exc)}

        tasks = [fetch_one(rid) for rid in REGIONS_MADAGASCAR]
        pairs = await asyncio.gather(*tasks)
        result = dict(pairs)

        ok = sum(1 for v in result.values() if "erreur" not in v)
        logger.info("Collecte nutrition batch — {}/{} régions OK", ok, len(result))
        return result

    # ─────────────────────────────────────────────
    # Calcul indicateurs dérivés
    # ─────────────────────────────────────────────

    @staticmethod
    def calculer_fcs(
        frequences: Dict[str, int],
    ) -> float:
        """
        Calcule le Food Consumption Score à partir des fréquences
        de consommation de chaque groupe alimentaire (jours/semaine).

        frequences : {groupe_alimentaire: jours_par_semaine_0_a_7}
        """
        score = 0.0
        for groupe, poids in FCS_WEIGHTS.items():
            freq = min(7, max(0, frequences.get(groupe, 0)))
            score += freq * poids
        return round(score, 1)

    @staticmethod
    def calculer_hdds(groupes_consommes: List[str]) -> int:
        """
        Household Dietary Diversity Score : nombre de groupes alimentaires
        consommés au cours des dernières 24h (0-12).
        """
        groupes_valides = set(GROUPES_ALIMENTAIRES_HDDS)
        return len(set(groupes_consommes) & groupes_valides)

    @staticmethod
    def calculer_rcsi(strategies: Dict[str, int]) -> int:
        """
        Reduced Coping Strategies Index.
        strategies : {stratégie: fréquence (0-7 jours)}

        Stratégies pondérées selon sévérité :
          manger_moins_preferes     : 1
          reduire_quantite          : 1
          reduire_repas_adultes     : 3
          emprunter_nourriture      : 2
          reduire_repas_journee     : 1
        """
        poids_rcsi = {
            "manger_moins_preferes": 1,
            "reduire_quantite":      1,
            "reduire_repas_adultes": 3,
            "emprunter_nourriture":  2,
            "reduire_repas_journee": 1,
        }
        score = 0
        for strategie, poids in poids_rcsi.items():
            freq = min(7, max(0, strategies.get(strategie, 0)))
            score += freq * poids
        return score

    def calculer_score_insecurite_alimentaire(
        self, fcs: float, hdds: int, rcsi: int
    ) -> Dict[str, Any]:
        """
        Score composite d'insécurité alimentaire (0-100).
        Combine FCS, HDDS et rCSI en un indicateur unique.
        """
        # Normalisation
        fcs_norm  = max(0, min(100, (fcs / 112) * 100))
        hdds_norm = max(0, min(100, (hdds / 12) * 100))
        rcsi_norm = max(0, min(100, 100 - (rcsi / 56) * 100))

        # Pondération : FCS 50%, HDDS 30%, rCSI 20%
        score = 0.5 * fcs_norm + 0.3 * hdds_norm + 0.2 * rcsi_norm

        if score >= 75:
            niveau = "sécurité alimentaire"
        elif score >= 50:
            niveau = "insécurité légère"
        elif score >= 25:
            niveau = "insécurité modérée"
        else:
            niveau = "insécurité sévère"

        return {
            "score_composite": round(score, 1),
            "niveau": niveau,
            "fcs_contribution": round(fcs_norm, 1),
            "hdds_contribution": round(hdds_norm, 1),
            "rcsi_contribution": round(rcsi_norm, 1),
        }

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _classifier_fcs(fcs: float) -> str:
        if fcs < 21:
            return "pauvre"
        elif fcs < 35:
            return "limite"
        elif fcs <= 42:
            return "acceptable_bas"
        else:
            return "acceptable"

    @staticmethod
    def _classifier_gam(gam: float) -> str:
        if gam < 5:
            return "acceptable"
        elif gam < 10:
            return "alerte"
        elif gam < 15:
            return "urgence"
        else:
            return "crise"

    @staticmethod
    def _get_strategies_coping(vulnerabilite: float) -> List[str]:
        strategies_base = ["réduction portions", "consommation aliments moins préférés"]
        if vulnerabilite >= 0.6:
            strategies_base += ["emprunt nourriture", "réduction repas adultes"]
        if vulnerabilite >= 0.8:
            strategies_base += [
                "vente actifs productifs",
                "migration saisonnière pour nourriture",
                "mendicité",
            ]
        return strategies_base

    def _fcs_synthetique(self, region_id: str) -> Dict:
        """FCS synthétique basé uniquement sur la vulnérabilité régionale."""
        profile = REGION_FOOD_PROFILE.get(region_id, DEFAULT_FOOD_PROFILE)
        vuln = profile.get("indice_vulnerabilite", 0.5)
        fcs = round(60 * (1 - vuln), 1)
        return {
            "region_id": region_id,
            "date_observation": date.today().isoformat(),
            "score_fcs": max(10, fcs),
            "classification_fcs": self._classifier_fcs(fcs),
            "hdds": round(8 * (1 - vuln * 0.6), 1),
            "rcsi": int(vuln * 25),
            "source": "Estimation synthétique",
        }

    def _prix_synthetiques(self, region_id: str) -> Dict:
        """
        Prix synthétiques basés sur les prix moyens nationaux Madagascar.
        Référence : WFP Madagascar Country Brief (estimations 2024).
        """
        # Prix en Ariary (MGA) / kg — valeurs approx.
        profile = REGION_FOOD_PROFILE.get(region_id, DEFAULT_FOOD_PROFILE)
        vuln = profile.get("indice_vulnerabilite", 0.5)

        # Prix plus élevés dans les zones enclavées (transport)
        coeff_enclavement = 1 + vuln * 0.5

        return {
            "region_id": region_id,
            "date_observation": date.today().isoformat(),
            "prix_riz_kg":       round(1800 * coeff_enclavement),
            "prix_manioc_kg":    round(400 * coeff_enclavement),
            "prix_mais_kg":      round(800 * coeff_enclavement),
            "prix_haricots_kg":  round(3000 * coeff_enclavement),
            "prix_huile_litre":  round(6000 * coeff_enclavement),
            "variation_prix_pct_1m": None,
            "source": "Prix synthétiques (référence nationale)",
            "unite": "MGA",
        }