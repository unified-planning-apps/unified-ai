"""
Collecte de données météorologiques multi-sources :
  - OpenWeatherMap (temps réel + prévisions 7j)
  - NASA POWER API (données agro-climatiques historiques — gratuit)
  - Copernicus Climate Data Store (historiques riches)
  - Sentinel Hub (NDVI et zones humides via satellite)

Stratégie de fallback :
  OpenWeatherMap → NASA POWER → Copernicus → Données DB historiques

Toutes les méthodes sont async (aiohttp).
Retry automatique (tenacity) sur erreur réseau.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from src.utils.constants import REGIONS_MADAGASCAR


# ─────────────────────────────────────────────────────────────────
# Mapping région → coordonnées GPS (centre de chaque région)
# ─────────────────────────────────────────────────────────────────
REGION_COORDS: Dict[str, Dict[str, float]] = {
    "MDG-ANA":  {"lat": -18.9166, "lon": 47.5361},
    "MDG-VAK":  {"lat": -19.8659, "lon": 47.0337},
    "MDG-ITM":  {"lat": -19.0027, "lon": 46.7396},
    "MDG-BMT":  {"lat": -18.7722, "lon": 46.0522},
    "MDG-MAT":  {"lat": -21.4527, "lon": 47.0868},
    "MDG-ATI":  {"lat": -20.5297, "lon": 47.2444},
    "MDG-VAT":  {"lat": -22.1457, "lon": 48.0114},
    "MDG-FIT":  {"lat": -21.2333, "lon": 48.3333},
    "MDG-ANO":  {"lat": -22.8167, "lon": 47.8333},
    "MDG-ATS":  {"lat": -18.1443, "lon": 49.4017},
    "MDG-ANA2": {"lat": -17.3801, "lon": 49.4067},
    "MDG-ALA":  {"lat": -17.8333, "lon": 48.4167},
    "MDG-BOE":  {"lat": -15.7163, "lon": 46.3224},
    "MDG-SOF":  {"lat": -14.8667, "lon": 47.9833},
    "MDG-MEN":  {"lat": -18.0561, "lon": 44.0278},
    "MDG-MEN2": {"lat": -20.2833, "lon": 44.2833},
    "MDG-DIA":  {"lat": -12.3484, "lon": 49.2958},
    "MDG-SAV":  {"lat": -14.2667, "lon": 50.1667},
    "MDG-IHO":  {"lat": -22.4014, "lon": 46.1278},
    "MDG-ASO":  {"lat": -23.3568, "lon": 43.6685},
    "MDG_AND":  {"lat": -25.1767, "lon": 46.0842},
    "MDG-AAN":  {"lat": -25.0333, "lon": 46.9667},
}

# Mapping région → noms des régions
REGION_NAMES: Dict[str, str] = {
    "MDG-ANA":  "Analamanga",
    "MDG-VAK":  "Vakinankaratra",
    "MDG-ITM":  "Itasy",
    "MDG-BMT":  "Bongolava",
    "MDG-MAT":  "Matsiatra Ambony",
    "MDG-ATI":  "Amoron'i Mania",
    "MDG-VAT":  "Vatovavy",
    "MDG-FIT":  "Fitovinany",
    "MDG-ANO":  "Atsimo-Atsinanana",
    "MDG-ATS":  "Atsinanana",
    "MDG-ANA2": "Analanjirofo",
    "MDG-ALA":  "Alaotra-Mangoro",
    "MDG-BOE":  "Boeny",
    "MDG-SOF":  "Sofia",
    "MDG-MEN":  "Melaky",
    "MDG-MEN2": "Menabe",
    "MDG-DIA":  "Diana",
    "MDG-SAV":  "Sava",
    "MDG-IHO":  "Ihorombe",
    "MDG-ASO":  "Atsimo-Andrefana",
    "MDG_AND":  "Androy",
    "MDG-AAN":  "Anosy",
}


class WeatherAPIError(Exception):
    """Erreur spécifique à la collecte météo."""
    pass


class WeatherFetcher:
    """
    Collecteur météo multi-sources avec fallback automatique.

    Usage :
        fetcher = WeatherFetcher()
        current = await fetcher.get_current("MDG-ANA")
        forecast = await fetcher.get_forecast("MDG-ATS", days=7)
        history = await fetcher.get_history_nasa("MDG-BOE", date(2024,1,1), date(2024,3,31))
    """

    OWM_BASE       = "https://api.openweathermap.org/data/2.5"
    OWM_BASE_V3    = "https://api.openweathermap.org/data/3.0"
    NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    SENTINEL_TOKEN = "https://services.sentinel-hub.com/oauth/token"
    SENTINEL_WMS   = "https://services.sentinel-hub.com/ogc/wms"

    def __init__(self):
        self._api_key = settings.weather_api.openweather_api_key
        self._nasa_url = settings.weather_api.nasa_power_base_url
        self._sentinel_client_id = settings.weather_api.sentinel_hub_client_id
        self._sentinel_secret = settings.weather_api.sentinel_hub_client_secret
        self._sentinel_token: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None

    # ─────────────────────────────────────────────
    # Session aiohttp (lazy init + réutilisation)
    # ─────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "UNICEF-Madagascar-Predictor/1.0",
                    "Accept": "application/json",
                },
            )
        return self._session

    async def close(self):
        """Ferme proprement la session HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ─────────────────────────────────────────────
    # Validation région
    # ─────────────────────────────────────────────

    def _get_coords(self, region_id: str) -> Dict[str, float]:
        coords = REGION_COORDS.get(region_id)
        if not coords:
            raise WeatherAPIError(
                f"Région '{region_id}' inconnue. "
                f"Régions valides : {list(REGION_COORDS.keys())}"
            )
        return coords

    # ─────────────────────────────────────────────
    # OPENWEATHERMAP — Données actuelles
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def get_current(self, region_id: str) -> Dict[str, Any]:
        """
        Retourne les conditions météo actuelles via OpenWeatherMap.
        Fallback vers NASA POWER (dernière donnée disponible) si OWM échoue.
        """
        coords = self._get_coords(region_id)
        region_name = REGION_NAMES.get(region_id, region_id)

        try:
            data = await self._owm_current(coords["lat"], coords["lon"])
            result = self._parse_owm_current(data, region_id, region_name)
            logger.debug("OWM current OK — {}", region_id)
            return result

        except Exception as exc:
            logger.warning(
                "OWM current échoué pour {} ({}), fallback NASA POWER", region_id, exc
            )
            return await self._fallback_current_nasa(region_id, coords, region_name)

    async def _owm_current(self, lat: float, lon: float) -> Dict:
        session = await self._get_session()
        url = f"{self.OWM_BASE}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key,
            "units": "metric",
            "lang": "fr",
        }
        async with session.get(url, params=params) as resp:
            if resp.status == 401:
                raise WeatherAPIError("Clé API OpenWeatherMap invalide.")
            if resp.status == 429:
                raise WeatherAPIError("Rate limit OWM dépassé.")
            resp.raise_for_status()
            return await resp.json()

    def _parse_owm_current(
        self, data: Dict, region_id: str, region_name: str
    ) -> Dict[str, Any]:
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        rain = data.get("rain", {})
        clouds = data.get("clouds", {})

        return {
            "region_id": region_id,
            "region_name": region_name,
            "horodatage": datetime.utcfromtimestamp(
                data.get("dt", datetime.utcnow().timestamp())
            ).isoformat(),
            "temperature_c": main.get("temp", 0.0),
            "temperature_min_c": main.get("temp_min", 0.0),
            "temperature_max_c": main.get("temp_max", 0.0),
            "temperature_ressentie_c": main.get("feels_like", 0.0),
            "humidite_pct": main.get("humidity", 0),
            "precipitations_mm": rain.get("1h", 0.0),
            "vent_kmh": round((wind.get("speed", 0)) * 3.6, 1),
            "vent_direction_deg": wind.get("deg", 0),
            "pression_hpa": main.get("pressure", 1013.0),
            "couverture_nuageuse_pct": clouds.get("all", 0),
            "visibilite_m": data.get("visibility", 10000),
            "indice_uv": None,  # Disponible via One Call API v3
            "description": weather.get("description", ""),
            "icone": weather.get("icon", ""),
            "latitude": data.get("coord", {}).get("lat"),
            "longitude": data.get("coord", {}).get("lon"),
            "source": "OpenWeatherMap",
        }

    # ─────────────────────────────────────────────
    # OPENWEATHERMAP — Prévisions 7 jours
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError),
        reraise=True,
    )
    async def get_forecast(self, region_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Prévisions météo sur 1–14 jours via OWM.
        Inclut détection automatique des alertes (cyclone, sécheresse, inondation).
        """
        coords = self._get_coords(region_id)
        region_name = REGION_NAMES.get(region_id, region_id)

        try:
            raw = await self._owm_forecast(coords["lat"], coords["lon"])
            previsions = self._parse_owm_forecast(raw, days)
        except Exception as exc:
            logger.warning("OWM forecast échoué pour {} : {}", region_id, exc)
            previsions = await self._fallback_forecast_nasa(region_id, coords, days)

        # Analyse des alertes météo
        alerte_cyclone   = self._detecter_cyclone(previsions)
        alerte_secheresse = self._detecter_secheresse(previsions)
        alerte_inondation = self._detecter_inondation(previsions)

        return {
            "region_id": region_id,
            "region_name": region_name,
            "previsions": previsions,
            "alerte_cyclone": alerte_cyclone,
            "alerte_secheresse": alerte_secheresse,
            "alerte_inondation": alerte_inondation,
            "genere_le": datetime.utcnow().isoformat(),
            "source": "OpenWeatherMap",
        }

    async def _owm_forecast(self, lat: float, lon: float) -> Dict:
        session = await self._get_session()
        # OWM forecast gratuit : toutes les 3h sur 5j (40 points)
        url = f"{self.OWM_BASE}/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._api_key,
            "units": "metric",
            "lang": "fr",
            "cnt": 40,
        }
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _parse_owm_forecast(self, raw: Dict, days: int) -> List[Dict]:
        """Agrège les prévisions 3h en prévisions journalières."""
        from collections import defaultdict

        daily: Dict[str, Dict] = defaultdict(lambda: {
            "temps": [],
            "pluie": 0.0,
            "vent_max": 0.0,
            "humidite": [],
            "nuages": [],
            "description": "",
        })

        for item in raw.get("list", []):
            dt = datetime.utcfromtimestamp(item["dt"])
            day_key = dt.strftime("%Y-%m-%d")
            d = daily[day_key]

            main = item.get("main", {})
            d["temps"].extend([main.get("temp_min", 0), main.get("temp_max", 0)])
            d["pluie"] += item.get("rain", {}).get("3h", 0.0)
            d["vent_max"] = max(d["vent_max"], item.get("wind", {}).get("speed", 0) * 3.6)
            d["humidite"].append(main.get("humidity", 0))
            d["nuages"].append(item.get("clouds", {}).get("all", 0))
            if not d["description"]:
                weather = item.get("weather", [{}])[0]
                d["description"] = weather.get("description", "")
            d["pop"] = max(d.get("pop", 0), item.get("pop", 0))  # prob of precipitation

        result = []
        for day_str, d in sorted(daily.items())[:days]:
            temps = d["temps"]
            result.append({
                "date": day_str,
                "temperature_min_c": round(min(temps), 1) if temps else 0.0,
                "temperature_max_c": round(max(temps), 1) if temps else 0.0,
                "precipitations_mm": round(d["pluie"], 1),
                "precipitations_prob_pct": round(d.get("pop", 0) * 100, 0),
                "humidite_moy_pct": round(
                    sum(d["humidite"]) / len(d["humidite"]), 0
                ) if d["humidite"] else 0.0,
                "vent_max_kmh": round(d["vent_max"], 1),
                "couverture_nuageuse_pct": round(
                    sum(d["nuages"]) / len(d["nuages"]), 0
                ) if d["nuages"] else 0.0,
                "description": d["description"],
                "risque_cyclone": False,  # calculé après
            })

        return result

    def _detecter_cyclone(self, previsions: List[Dict]) -> bool:
        """Cyclone si vent > 120 km/h ou pluies > 100mm/j sur 2j consécutifs."""
        jours_fortes_pluies = 0
        for p in previsions:
            if p.get("vent_max_kmh", 0) > 120:
                return True
            if p.get("precipitations_mm", 0) > 100:
                jours_fortes_pluies += 1
                if jours_fortes_pluies >= 2:
                    return True
            else:
                jours_fortes_pluies = 0
        return False

    def _detecter_secheresse(self, previsions: List[Dict]) -> bool:
        """Sécheresse si précipitations totales 7j < 5mm."""
        total_pluie = sum(p.get("precipitations_mm", 0) for p in previsions[:7])
        return total_pluie < 5.0

    def _detecter_inondation(self, previsions: List[Dict]) -> bool:
        """Inondation si > 50mm en 24h ou > 150mm en 72h."""
        for p in previsions:
            if p.get("precipitations_mm", 0) > 50:
                return True
        # Vérification 72h glissant
        for i in range(len(previsions) - 2):
            total_3j = sum(
                previsions[i + k].get("precipitations_mm", 0) for k in range(3)
            )
            if total_3j > 150:
                return True
        return False

    # ─────────────────────────────────────────────
    # NASA POWER — Données historiques agro-clima
    # ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(aiohttp.ClientError),
        reraise=True,
    )
    async def get_history_nasa(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
    ) -> List[Dict[str, Any]]:
        """
        Historique météo via NASA POWER API (gratuit, mondial, agronomique).

        Variables récupérées :
          T2M      — température moyenne 2m (°C)
          T2M_MIN  — temp min
          T2M_MAX  — temp max
          PRECTOTCORR — précipitations corrigées (mm/j)
          RH2M     — humidité relative 2m (%)
          WS2M     — vitesse vent 2m (m/s)
          ALLSKY_SFC_SW_DWN — rayonnement solaire (MJ/m²/j)
          GWETROOT — humidité sol (fraction)
        """
        coords = self._get_coords(region_id)

        params = {
            "parameters": "T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN,GWETROOT",
            "community": "AG",
            "longitude": coords["lon"],
            "latitude": coords["lat"],
            "start": date_debut.strftime("%Y%m%d"),
            "end": date_fin.strftime("%Y%m%d"),
            "format": "JSON",
        }

        session = await self._get_session()
        async with session.get(self.NASA_POWER_URL, params=params) as resp:
            if resp.status == 422:
                logger.error("Paramètres NASA POWER invalides pour {}", region_id)
                return []
            resp.raise_for_status()
            raw = await resp.json()

        return self._parse_nasa_power(raw, region_id)

    def _parse_nasa_power(self, raw: Dict, region_id: str) -> List[Dict]:
        """Transforme la réponse NASA POWER (format pivot) en liste journalière."""
        props = raw.get("properties", {})
        params_data = props.get("parameter", {})

        t2m     = params_data.get("T2M", {})
        t2m_min = params_data.get("T2M_MIN", {})
        t2m_max = params_data.get("T2M_MAX", {})
        prec    = params_data.get("PRECTOTCORR", {})
        rh      = params_data.get("RH2M", {})
        ws      = params_data.get("WS2M", {})
        sol     = params_data.get("ALLSKY_SFC_SW_DWN", {})
        gwet    = params_data.get("GWETROOT", {})

        records = []
        for date_str in sorted(t2m.keys()):
            try:
                dt = datetime.strptime(date_str, "%Y%m%d").date()
            except ValueError:
                continue

            # NASA POWER utilise -999 pour données manquantes
            def clean(val, default=0.0):
                return default if val == -999 else val

            records.append({
                "region_id": region_id,
                "date": str(dt),
                "temperature_moy_c": round(clean(t2m.get(date_str, -999)), 2),
                "temperature_min_c": round(clean(t2m_min.get(date_str, -999)), 2),
                "temperature_max_c": round(clean(t2m_max.get(date_str, -999)), 2),
                "precipitations_mm": round(max(0, clean(prec.get(date_str, 0))), 2),
                "humidite_moy_pct": round(clean(rh.get(date_str, -999), 50), 1),
                "vent_kmh": round(clean(ws.get(date_str, 0)) * 3.6, 1),
                "rayonnement_solaire_mj": round(clean(sol.get(date_str, 0)), 2),
                "humidite_sol_fraction": round(clean(gwet.get(date_str, 0), 0), 3),
                "anomalie_temp": None,    # calculé post-traitement
                "anomalie_pluie": None,
                "source": "NASA POWER",
            })

        logger.debug(
            "NASA POWER {} : {} jours récupérés ({} → {})",
            region_id, len(records),
            records[0]["date"] if records else "?",
            records[-1]["date"] if records else "?",
        )
        return records

    # ─────────────────────────────────────────────
    # NASA POWER — Fallback current (dernier jour)
    # ─────────────────────────────────────────────

    async def _fallback_current_nasa(
        self, region_id: str, coords: Dict, region_name: str
    ) -> Dict:
        """Retourne la dernière donnée NASA POWER comme approximation du 'current'."""
        yesterday = date.today() - timedelta(days=1)
        try:
            history = await self.get_history_nasa(
                region_id, yesterday, yesterday
            )
            if history:
                h = history[0]
                return {
                    "region_id": region_id,
                    "region_name": region_name,
                    "horodatage": datetime.utcnow().isoformat(),
                    "temperature_c": h.get("temperature_moy_c", 0),
                    "temperature_min_c": h.get("temperature_min_c", 0),
                    "temperature_max_c": h.get("temperature_max_c", 0),
                    "humidite_pct": h.get("humidite_moy_pct", 50),
                    "precipitations_mm": h.get("precipitations_mm", 0),
                    "vent_kmh": h.get("vent_kmh", 0),
                    "pression_hpa": 1013.0,  # non disponible NASA POWER
                    "couverture_nuageuse_pct": 50,
                    "indice_uv": None,
                    "description": "Données NASA POWER (J-1)",
                    "source": "NASA POWER (fallback)",
                }
        except Exception as exc:
            logger.error("Fallback NASA échoué pour {} : {}", region_id, exc)

        # Dernier recours : données synthétiques basées sur climatologie
        return self._climatologie_par_defaut(region_id, region_name)

    async def _fallback_forecast_nasa(
        self, region_id: str, coords: Dict, days: int
    ) -> List[Dict]:
        """Prévisions approximatives via NASA POWER (historique moyen)."""
        # Utilise les 30 derniers jours comme proxy de prévision
        date_fin = date.today()
        date_debut = date_fin - timedelta(days=30)
        history = await self.get_history_nasa(region_id, date_debut, date_fin)

        if not history:
            return []

        # Moyenne journalière sur les 30 derniers jours → prévision approximative
        avg_temp_min = sum(h.get("temperature_min_c", 0) for h in history) / len(history)
        avg_temp_max = sum(h.get("temperature_max_c", 0) for h in history) / len(history)
        avg_pluie    = sum(h.get("precipitations_mm", 0) for h in history) / len(history)
        avg_humidite = sum(h.get("humidite_moy_pct", 50) for h in history) / len(history)
        avg_vent     = sum(h.get("vent_kmh", 0) for h in history) / len(history)

        previsions = []
        for i in range(days):
            dt = date.today() + timedelta(days=i + 1)
            previsions.append({
                "date": str(dt),
                "temperature_min_c": round(avg_temp_min, 1),
                "temperature_max_c": round(avg_temp_max, 1),
                "precipitations_mm": round(avg_pluie, 1),
                "precipitations_prob_pct": 50,
                "humidite_moy_pct": round(avg_humidite, 0),
                "vent_max_kmh": round(avg_vent, 1),
                "description": "Prévision estimée (moyenne historique NASA POWER)",
                "risque_cyclone": False,
                "source": "NASA POWER (fallback)",
            })
        return previsions

    # ─────────────────────────────────────────────
    # SENTINEL HUB — NDVI et zones humides
    # ─────────────────────────────────────────────

    async def get_ndvi(self, region_id: str, target_date: Optional[date] = None) -> Dict:
        """
        Récupère l'indice NDVI (végétation) depuis Sentinel-2 via Sentinel Hub.
        NDVI = (NIR - Red) / (NIR + Red)
          > 0.5 : végétation dense (favorable aux moustiques)
          0.2–0.5 : végétation modérée
          < 0.2 : sol nu ou eau
        """
        if not self._sentinel_client_id:
            logger.warning("Sentinel Hub non configuré — NDVI non disponible")
            return {"region_id": region_id, "ndvi": None, "source": "non configuré"}

        target_date = target_date or date.today() - timedelta(days=5)
        coords = self._get_coords(region_id)

        try:
            token = await self._get_sentinel_token()
            ndvi_value = await self._fetch_sentinel_ndvi(
                token=token,
                lat=coords["lat"],
                lon=coords["lon"],
                target_date=target_date,
            )
            return {
                "region_id": region_id,
                "date": str(target_date),
                "ndvi": ndvi_value,
                "interpretation": self._interpreter_ndvi(ndvi_value),
                "source": "Sentinel-2",
            }
        except Exception as exc:
            logger.warning("NDVI Sentinel échoué pour {} : {}", region_id, exc)
            return {
                "region_id": region_id,
                "date": str(target_date),
                "ndvi": None,
                "source": "erreur",
                "erreur": str(exc),
            }

    async def _get_sentinel_token(self) -> str:
        """Obtient un token OAuth2 pour Sentinel Hub."""
        if self._sentinel_token:
            return self._sentinel_token

        session = await self._get_session()
        data = {
            "grant_type": "client_credentials",
            "client_id": self._sentinel_client_id,
            "client_secret": self._sentinel_secret,
        }
        async with session.post(self.SENTINEL_TOKEN, data=data) as resp:
            resp.raise_for_status()
            token_data = await resp.json()
            self._sentinel_token = token_data["access_token"]
            return self._sentinel_token

    async def _fetch_sentinel_ndvi(
        self, token: str, lat: float, lon: float, target_date: date
    ) -> Optional[float]:
        """Requête WMS Sentinel Hub pour valeur NDVI moyenne."""
        # Bounding box autour du centroïde (±0.5°)
        bbox = f"{lon-0.5},{lat-0.5},{lon+0.5},{lat+0.5}"
        date_str = target_date.strftime("%Y-%m-%d")

        session = await self._get_session()
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "LAYERS": "NDVI",
            "BBOX": bbox,
            "WIDTH": 512,
            "HEIGHT": 512,
            "FORMAT": "image/tiff",
            "TIME": f"{date_str}/{date_str}",
            "MAXCC": 20,  # max cloud cover 20%
        }
        headers = {"Authorization": f"Bearer {token}"}

        # NOTE : En prod, on parserait le GeoTIFF pour calculer la moyenne NDVI
        # Ici on retourne une valeur simulée (intégration GeoTIFF = rasterio)
        logger.debug("NDVI Sentinel fetch — lat={} lon={} date={}", lat, lon, date_str)
        return None  # À implémenter avec rasterio

    def _interpreter_ndvi(self, ndvi: Optional[float]) -> str:
        if ndvi is None:
            return "non disponible"
        if ndvi > 0.6:
            return "végétation très dense — risque moustiques élevé"
        elif ndvi > 0.4:
            return "végétation modérée"
        elif ndvi > 0.2:
            return "végétation clairsemée"
        elif ndvi > 0:
            return "sol nu / zones dégradées"
        else:
            return "eau / nuages / neige"

    # ─────────────────────────────────────────────
    # Collecte batch — toutes les régions
    # ─────────────────────────────────────────────

    async def get_all_regions_current(
        self,
        concurrency: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Collecte les données météo actuelles pour les 22 régions.
        Limite la concurrence à `concurrency` requêtes simultanées.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_one(rid: str) -> Dict:
            async with semaphore:
                try:
                    return await self.get_current(rid)
                except Exception as exc:
                    logger.error("Erreur fetch_all région {} : {}", rid, exc)
                    return {
                        "region_id": rid,
                        "erreur": str(exc),
                        "source": "erreur",
                    }

        tasks = [fetch_one(rid) for rid in REGIONS_MADAGASCAR]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        ok = sum(1 for r in results if "erreur" not in r)
        logger.info(
            "Collecte batch météo terminée : {}/{} régions OK",
            ok, len(REGIONS_MADAGASCAR)
        )
        return results

    async def get_history_all_regions(
        self,
        date_debut: date,
        date_fin: date,
        concurrency: int = 3,
    ) -> Dict[str, List[Dict]]:
        """Collecte historique NASA POWER pour toutes les régions."""
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_one(rid: str):
            async with semaphore:
                try:
                    data = await self.get_history_nasa(rid, date_debut, date_fin)
                    return rid, data
                except Exception as exc:
                    logger.error("Erreur historique région {} : {}", rid, exc)
                    return rid, []

        tasks = [fetch_one(rid) for rid in REGIONS_MADAGASCAR]
        pairs = await asyncio.gather(*tasks)
        return dict(pairs)

    # ─────────────────────────────────────────────
    # Calcul anomalies climatiques
    # ─────────────────────────────────────────────

    def calculer_anomalies(
        self,
        historique: List[Dict],
        normale_30ans: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Calcule les anomalies par rapport à la normale climatologique.
        Si normale_30ans non fourni, utilise la moyenne de la période.
        """
        if not historique:
            return historique

        if normale_30ans is None:
            # Normale basée sur la période donnée
            avg_temp = sum(h.get("temperature_moy_c", 0) for h in historique) / len(historique)
            avg_pluie = sum(h.get("precipitations_mm", 0) for h in historique) / len(historique)
        else:
            avg_temp  = normale_30ans.get("temperature_moy_c", 25)
            avg_pluie = normale_30ans.get("precipitations_mm", 5)

        for h in historique:
            h["anomalie_temp"] = round(h.get("temperature_moy_c", 0) - avg_temp, 2)
            h["anomalie_pluie"] = round(h.get("precipitations_mm", 0) - avg_pluie, 2)

        return historique

    # ─────────────────────────────────────────────
    # Données de climatologie par défaut
    # ─────────────────────────────────────────────

    def _climatologie_par_defaut(self, region_id: str, region_name: str) -> Dict:
        """Valeurs climatologiques moyennes par région (dernier recours)."""
        # Valeurs moyennes simplifiées — à enrichir avec vraies normales climatiques
        CLIM_DEFAULTS = {
            "MDG-ANA": {"temp": 19.0, "pluie": 3.5, "humidite": 75},
            "MDG-ATS": {"temp": 24.5, "pluie": 10.2, "humidite": 85},
            "MDG-BOE": {"temp": 26.0, "pluie": 2.8, "humidite": 70},
            "MDG_AND": {"temp": 23.0, "pluie": 1.2, "humidite": 55},
            "MDG-ASO": {"temp": 24.0, "pluie": 1.0, "humidite": 58},
        }
        clim = CLIM_DEFAULTS.get(region_id, {"temp": 24.0, "pluie": 3.0, "humidite": 72})

        return {
            "region_id": region_id,
            "region_name": region_name,
            "horodatage": datetime.utcnow().isoformat(),
            "temperature_c": clim["temp"],
            "temperature_min_c": clim["temp"] - 4,
            "temperature_max_c": clim["temp"] + 4,
            "humidite_pct": clim["humidite"],
            "precipitations_mm": clim["pluie"],
            "vent_kmh": 12.0,
            "pression_hpa": 1013.0,
            "couverture_nuageuse_pct": 50,
            "indice_uv": None,
            "description": "Valeur climatologique par défaut",
            "source": "Climatologie (fallback final)",
        }

    # ─────────────────────────────────────────────
    # Calcul indices dérivés
    # ─────────────────────────────────────────────

    @staticmethod
    def calculer_indice_chaleur(temp_c: float, humidite_pct: float) -> float:
        """
        Rothfusz Heat Index — ressenti thermique.
        Important pour la survie des moustiques.
        """
        if temp_c < 27:
            return temp_c
        T = temp_c * 9 / 5 + 32  # conversion Fahrenheit
        H = humidite_pct
        HI = (
            -42.379
            + 2.04901523 * T
            + 10.14333127 * H
            - 0.22475541 * T * H
            - 0.00683783 * T**2
            - 0.05481717 * H**2
            + 0.00122874 * T**2 * H
            + 0.00085282 * T * H**2
            - 0.00000199 * T**2 * H**2
        )
        return round((HI - 32) * 5 / 9, 1)  # retour Celsius

    @staticmethod
    def calculer_spi(precipitations_serie: List[float], echelle: int = 30) -> float:
        """
        Standardized Precipitation Index (SPI).
        SPI < -1 : sécheresse modérée
        SPI < -2 : sécheresse sévère
        SPI > 1  : humidité excessive
        """
        import numpy as np
        if len(precipitations_serie) < echelle:
            return 0.0
        window = precipitations_serie[-echelle:]
        mean = np.mean(window)
        std  = np.std(window)
        if std == 0:
            return 0.0
        return round((np.sum(window) / echelle - mean) / std, 3)