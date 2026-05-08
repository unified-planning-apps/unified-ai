"""
src/api/routers/weather.py
===========================
Endpoints météorologiques.
- Données actuelles par région
- Prévisions 7 jours
- Historique (séries temporelles)
- Indices climatiques (NDVI, humidité sol)
- Détection d'anomalies météo
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Path, Query, status
from loguru import logger
from pydantic import BaseModel, Field

from src.api.dependencies import AuthUser, Cache, DbSession, Pagination

router = APIRouter()


# ─────────────────────────────────────────────────────────────────
# Schémas Pydantic (Request / Response)
# ─────────────────────────────────────────────────────────────────

class WeatherCurrent(BaseModel):
    region_id: str
    region_name: str
    horodatage: datetime
    temperature_c: float = Field(..., description="Température en °C")
    temperature_min_c: float
    temperature_max_c: float
    humidite_pct: float = Field(..., ge=0, le=100)
    precipitations_mm: float = Field(..., ge=0)
    vent_kmh: float = Field(..., ge=0)
    pression_hpa: float
    couverture_nuageuse_pct: float = Field(..., ge=0, le=100)
    indice_uv: Optional[float] = None
    description: str
    source: str = Field(default="OpenWeatherMap")


class WeatherForecastDay(BaseModel):
    date: date
    temperature_min_c: float
    temperature_max_c: float
    precipitations_mm: float
    precipitations_prob_pct: float = Field(..., ge=0, le=100)
    humidite_moy_pct: float
    vent_max_kmh: float
    description: str
    risque_cyclone: bool = False


class WeatherForecast(BaseModel):
    region_id: str
    region_name: str
    previsions: List[WeatherForecastDay]
    alerte_cyclone: bool = False
    alerte_secheresse: bool = False
    alerte_inondation: bool = False
    genere_le: datetime


class WeatherHistoryPoint(BaseModel):
    date: date
    temperature_moy_c: float
    precipitations_mm: float
    humidite_moy_pct: float
    anomalie_temp: Optional[float] = Field(
        None, description="Écart à la normale climatologique (°C)"
    )
    anomalie_pluie: Optional[float] = Field(
        None, description="Écart à la normale pluie (mm)"
    )


class ClimateIndex(BaseModel):
    region_id: str
    date: date
    ndvi: Optional[float] = Field(None, ge=-1, le=1, description="Indice végétation")
    spi: Optional[float] = Field(
        None, description="Standardized Precipitation Index"
    )
    humidite_sol_pct: Optional[float] = Field(None, ge=0, le=100)
    zones_humides_pct: Optional[float] = Field(
        None, ge=0, le=100, description="% zone humide favorable aux moustiques"
    )


class WeatherAnomaly(BaseModel):
    region_id: str
    type_anomalie: str  # "chaleur_extreme" | "secheresse" | "inondation" | "cyclone"
    severite: str       # "faible" | "moderee" | "severe" | "extreme"
    debut: date
    fin_estimee: Optional[date]
    description: str
    impact_paludisme: str
    impact_nutrition: str


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@router.get(
    "/actuel/{region_id}",
    response_model=WeatherCurrent,
    summary="Météo actuelle d'une région",
    description=(
        "Retourne les conditions météorologiques actuelles pour une région de Madagascar. "
        "Données mises en cache 1h (OpenWeatherMap)."
    ),
)
async def get_current_weather(
    region_id: str = Path(..., description="ID région (ex: MDG-ANA)", example="MDG-ANA"),
    user: AuthUser = None,
    cache: Cache = None,
    db: DbSession = None,
):
    cache_key = f"weather:current:{region_id}"

    # 1. Tentative lecture cache
    cached = await cache.get(cache_key)
    if cached:
        logger.debug("Cache HIT météo actuelle — région {}", region_id)
        return json.loads(cached)

    # 2. Fetch depuis API météo
    logger.info("Cache MISS — fetch météo actuelle pour {}", region_id)
    try:
        from src.data_collection.weather_fetcher import WeatherFetcher
        fetcher = WeatherFetcher()
        data = await fetcher.get_current(region_id)
    except Exception as exc:
        logger.error("Erreur fetch météo {} : {}", region_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "METEO_INDISPONIBLE",
                "message": f"Impossible de récupérer la météo pour {region_id}.",
            },
        ) from exc

    # 3. Mise en cache 1h
    await cache.set(cache_key, json.dumps(data, default=str), ttl=3600)
    return data


@router.get(
    "/previsions/{region_id}",
    response_model=WeatherForecast,
    summary="Prévisions météo 7 jours",
    description="Prévisions météo sur 7 jours avec alertes cyclone/sécheresse/inondation.",
)
async def get_weather_forecast(
    region_id: str = Path(..., description="ID région", example="MDG-ATS"),
    jours: int = Query(default=7, ge=1, le=14, description="Nombre de jours (1-14)"),
    user: AuthUser = None,
    cache: Cache = None,
):
    cache_key = f"weather:forecast:{region_id}:{jours}"

    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    try:
        from src.data_collection.weather_fetcher import WeatherFetcher
        fetcher = WeatherFetcher()
        data = await fetcher.get_forecast(region_id, days=jours)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "PREVISIONS_INDISPONIBLES", "message": str(exc)},
        ) from exc

    await cache.set(cache_key, json.dumps(data, default=str), ttl=3600)
    return data


@router.get(
    "/historique/{region_id}",
    response_model=List[WeatherHistoryPoint],
    summary="Historique météo d'une région",
    description="Série temporelle météo avec anomalies climatiques. Max 365 jours.",
)
async def get_weather_history(
    region_id: str = Path(..., description="ID région"),
    date_debut: date = Query(..., description="Date début (YYYY-MM-DD)"),
    date_fin: date = Query(..., description="Date fin (YYYY-MM-DD)"),
    user: AuthUser = None,
    db: DbSession = None,
    pagination: Pagination = None,
):
    # Validation plage
    delta = (date_fin - date_debut).days
    if delta < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "PERIODE_INVALIDE",
                "message": "date_debut doit être antérieure à date_fin.",
            },
        )
    if delta > 365:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "PERIODE_TROP_LONGUE",
                "message": "La période ne peut pas dépasser 365 jours.",
            },
        )

    from src.database.repositories.weather_repo import WeatherRepository
    repo = WeatherRepository(db)
    records = await repo.get_history(
        region_id=region_id,
        date_debut=date_debut,
        date_fin=date_fin,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return records


@router.get(
    "/indices/{region_id}",
    response_model=List[ClimateIndex],
    summary="Indices climatiques (NDVI, SPI, humidité sol)",
    description=(
        "Indices agrégés : NDVI (végétation satellite), SPI (sécheresse), "
        "humidité du sol, zones humides (facteur risque paludisme)."
    ),
)
async def get_climate_indices(
    region_id: str = Path(..., description="ID région"),
    date_debut: date = Query(
        default_factory=lambda: date.today() - timedelta(days=30),
        description="Début période",
    ),
    date_fin: date = Query(
        default_factory=lambda: date.today(),
        description="Fin période",
    ),
    user: AuthUser = None,
    db: DbSession = None,
    cache: Cache = None,
):
    cache_key = f"climate:indices:{region_id}:{date_debut}:{date_fin}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.weather_repo import WeatherRepository
    repo = WeatherRepository(db)
    indices = await repo.get_climate_indices(region_id, date_debut, date_fin)

    await cache.set(cache_key, json.dumps(indices, default=str), ttl=43200)  # 12h
    return indices


@router.get(
    "/anomalies",
    response_model=List[WeatherAnomaly],
    summary="Anomalies météo actives sur Madagascar",
    description=(
        "Liste les anomalies climatiques en cours (sécheresses, inondations, cyclones) "
        "avec leur impact estimé sur le paludisme et la nutrition."
    ),
)
async def get_active_anomalies(
    region_id: Optional[str] = Query(None, description="Filtrer par région"),
    type_anomalie: Optional[str] = Query(
        None,
        description="Type : chaleur_extreme | secheresse | inondation | cyclone",
    ),
    user: AuthUser = None,
    cache: Cache = None,
    db: DbSession = None,
):
    cache_key = f"weather:anomalies:{region_id or 'all'}:{type_anomalie or 'all'}"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.database.repositories.weather_repo import WeatherRepository
    repo = WeatherRepository(db)
    anomalies = await repo.get_active_anomalies(
        region_id=region_id,
        type_anomalie=type_anomalie,
    )

    await cache.set(cache_key, json.dumps(anomalies, default=str), ttl=1800)  # 30min
    return anomalies


@router.get(
    "/resume/national",
    summary="Résumé météo national — toutes les 22 régions",
    description="Snapshot météo actuelle pour les 22 régions de Madagascar. Cache 1h.",
)
async def get_national_weather_summary(
    user: AuthUser = None,
    cache: Cache = None,
    db: DbSession = None,
):
    cache_key = "weather:national:summary"
    cached = await cache.get(cache_key)
    if cached:
        return json.loads(cached)

    from src.data_collection.weather_fetcher import WeatherFetcher
    import json as _json
    from pathlib import Path

    # Charge les régions
    with Path("config/regions_metadata.json").open() as f:
        regions_data = _json.load(f)

    fetcher = WeatherFetcher()
    summary = []
    for region in regions_data["regions"]:
        try:
            current = await fetcher.get_current(region["id"])
            summary.append(
                {
                    "region_id": region["id"],
                    "region_name": region["name"],
                    "temperature_c": current.get("temperature_c"),
                    "precipitations_mm": current.get("precipitations_mm"),
                    "humidite_pct": current.get("humidite_pct"),
                    "description": current.get("description"),
                }
            )
        except Exception as exc:
            logger.warning("Échec météo région {} : {}", region["id"], exc)
            summary.append(
                {
                    "region_id": region["id"],
                    "region_name": region["name"],
                    "erreur": "données non disponibles",
                }
            )

    await cache.set(cache_key, json.dumps(summary, default=str), ttl=3600)
    return {"regions": summary, "total": len(summary), "source": "OpenWeatherMap"}