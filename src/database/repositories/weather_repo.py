"""
Repository pour toutes les opérations DB liées à la météo.

Méthodes publiques (contrat avec routers et feature_engineering) :
  get_history(region_id, date_debut, date_fin, limit, offset)
  get_climate_indices(region_id, date_debut, date_fin)
  get_active_anomalies(region_id, type_anomalie)
  save_observation(data)
  save_observations_batch(data_list)
  get_latest(region_id)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import Date, and_, desc, func, select, text, cast
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import WeatherObservation, GeoNDVI


class WeatherRepository:
    """Repository async pour les données météorologiques."""

    def __init__(self, db: AsyncSession):
        self._db = db

    # ─────────────────────────────────────────────
    # READ
    # ─────────────────────────────────────────────

    async def get_history(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
        limit: int = 365,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retourne l'historique météo journalier agrégé pour une région.
        Appelé par :
          - GET /meteo/historique/{region_id}
          - src/preprocessing/feature_engineering.py (_fetch_weather_from_db)
        """
        try:
            if isinstance(date_debut, str):
                date_debut = date.fromisoformat(date_debut)

            if isinstance(date_fin, str):
                date_fin = date.fromisoformat(date_fin)

            day = cast(
                    func.date_trunc("day", WeatherObservation.horodatage),
                    Date
                ).label("date")

            stmt = (
                select(
                    day,
                    func.avg(WeatherObservation.temperature_c).label("temperature_moy_c"),
                    func.min(WeatherObservation.temperature_c).label("temperature_min_c"),
                    func.max(WeatherObservation.temperature_c).label("temperature_max_c"),
                    func.sum(WeatherObservation.precipitations_mm).label("precipitations_mm"),
                    func.avg(WeatherObservation.humidite_pct).label("humidite_moy_pct"),
                    func.avg(WeatherObservation.vent_kmh).label("vent_kmh"),
                    func.avg(WeatherObservation.pression_hpa).label("pression_hpa"),
                    func.avg(WeatherObservation.couverture_nuageuse_pct).label("couverture_nuageuse_pct"),
                    func.avg(WeatherObservation.rayonnement_solaire_mj).label("rayonnement_solaire_mj"),
                    func.avg(WeatherObservation.humidite_sol_fraction).label("humidite_sol_fraction"),
                )
                .where(
                    WeatherObservation.region_id == region_id,
                    WeatherObservation.horodatage >= datetime.combine(date_debut, datetime.min.time()),
                    WeatherObservation.horodatage <= datetime.combine(date_fin, datetime.max.time()),
                )
                .group_by(day)
                .order_by(day)
                .limit(limit)
                .offset(offset)
            )

            result = await self._db.execute(stmt)
            rows   = result.fetchall()

            records = []
            for row in rows:
                r = dict(row._mapping)
                # Calcul anomalies (placeholder — enrichi par WeatherProcessor)
                r["anomalie_temp"]  = None
                r["anomalie_pluie"] = None
                r["region_id"]      = region_id
                records.append(r)

            logger.debug(
                "WeatherRepo.get_history {} : {} records [{} → {}]",
                region_id, len(records), date_debut, date_fin
            )
            return records

        except Exception as exc:
            logger.exception("WeatherRepo.get_history {} failed", region_id)
            raise

    async def get_latest(self, region_id: str) -> Optional[Dict[str, Any]]:
        """Retourne la dernière observation météo disponible pour une région."""
        try:
            stmt = (
                select(WeatherObservation)
                .where(WeatherObservation.region_id == region_id)
                .order_by(desc(WeatherObservation.horodatage))
                .limit(1)
            )
            result = await self._db.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return self._obs_to_dict(row)
        except Exception as exc:
            logger.error("WeatherRepo.get_latest {} : {}", region_id, exc)
            return None

    async def get_climate_indices(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les indices climatiques dérivés (NDVI, SPI, humidité sol).
        Appelé par GET /meteo/indices/{region_id}.
        """
        try:
            # Données météo agrégées journalières
            weather_records = await self.get_history(region_id, date_debut, date_fin)

            # NDVI depuis table geo_ndvi
            ndvi_stmt = (
                select(
                    GeoNDVI.observation_date.label("date"),
                    GeoNDVI.ndvi_mean.label("ndvi"),
                    GeoNDVI.cloud_cover_pct,
                )
                .where(
                    and_(
                        GeoNDVI.region_id == region_id,
                        GeoNDVI.observation_date >= date_debut,
                        GeoNDVI.observation_date <= date_fin,
                    )
                )
                .order_by(GeoNDVI.observation_date)
            )
            ndvi_result = await self._db.execute(ndvi_stmt)
            ndvi_by_date = {
                str(row.date): float(row.ndvi)
                for row in ndvi_result.fetchall()
            }

            # Calcul SPI sur la série
            from src.preprocessing.weather_processor import WeatherProcessor
            processor = WeatherProcessor()
            pluies    = [float(r.get("precipitations_mm", 0)) for r in weather_records]

            indices = []
            for i, record in enumerate(weather_records):
                d = str(record.get("date", ""))
                spi = processor.compute_spi(pluies[:i+1], scale=min(30, i+1))

                # Zones humides estimées
                from src.preprocessing.geo_processor import GeoProcessor
                geo_proc = GeoProcessor(self._db)
                zones_humides = await geo_proc.get_zones_humides_pct(region_id)

                indices.append({
                    "region_id":          region_id,
                    "date":               d,
                    "ndvi":               ndvi_by_date.get(d),
                    "spi":                spi,
                    "humidite_sol_pct":   round(
                        float(record.get("humidite_sol_fraction", 0) or 0) * 100, 1
                    ),
                    "zones_humides_pct":  zones_humides,
                })

            return indices

        except Exception as exc:
            logger.error("WeatherRepo.get_climate_indices {} : {}", region_id, exc)
            return []

    async def get_active_anomalies(
        self,
        region_id: Optional[str] = None,
        type_anomalie: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les anomalies météo actives.
        Appelé par GET /meteo/anomalies.
        """
        try:
            # Récupère données 30 derniers jours pour toutes les régions concernées
            date_fin   = date.today()
            date_debut = date_fin - timedelta(days=30)

            regions_to_check = [region_id] if region_id else await self._get_all_region_ids()
            all_anomalies    = []

            from src.models.weather_forecaster import WeatherForecaster
            forecaster = WeatherForecaster()

            for rid in regions_to_check:
                records = await self.get_history(rid, date_debut, date_fin)
                if not records:
                    continue

                anomalies = forecaster.detect_anomalies(records)
                for a in anomalies:
                    if type_anomalie and a.get("type_anomalie") != type_anomalie:
                        continue
                    a["region_id"] = rid
                    all_anomalies.append(a)

            return all_anomalies

        except Exception as exc:
            logger.error("WeatherRepo.get_active_anomalies : {}", exc)
            return []

    async def get_national_summary(self) -> List[Dict[str, Any]]:
        """Résumé météo pour toutes les régions (dernière observation disponible)."""
        try:
            stmt = text("""
                WITH ranked AS (
                    SELECT
                        region_id,
                        AVG(temperature_c) AS temperature_c,
                        SUM(precipitations_mm) AS precipitations_mm,
                        AVG(humidite_pct) AS humidite_pct,
                        MAX(horodatage) AS derniere_obs,
                        ROW_NUMBER() OVER (PARTITION BY region_id ORDER BY MAX(horodatage) DESC) AS rn
                    FROM weather_observations
                    WHERE horodatage >= NOW() - INTERVAL '24 hours'
                    GROUP BY region_id, DATE_TRUNC('day', horodatage)
                )
                SELECT region_id, temperature_c, precipitations_mm,
                       humidite_pct, derniere_obs
                FROM ranked
                WHERE rn = 1
                ORDER BY region_id
            """)
            result = await self._db.execute(stmt)
            return [dict(row._mapping) for row in result.fetchall()]
        except Exception as exc:
            logger.error("WeatherRepo.get_national_summary : {}", exc)
            return []

    # ─────────────────────────────────────────────
    # WRITE
    # ─────────────────────────────────────────────

    async def save_observation(self, data: Dict[str, Any]) -> WeatherObservation:
        """
        Insère ou met à jour une observation météo.
        Appelé par le scheduler Celery (_sauvegarder_meteo_db en async).
        """
        try:
            # Recherche enregistrement existant
            horodatage = data.get("horodatage")
            if isinstance(horodatage, str):
                horodatage = datetime.fromisoformat(horodatage.replace("Z", "+00:00"))

            stmt = select(WeatherObservation).where(
                and_(
                    WeatherObservation.region_id == data["region_id"],
                    WeatherObservation.horodatage == horodatage,
                )
            )
            result  = await self._db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                # Update
                for field in [
                    "temperature_c", "temperature_min_c", "temperature_max_c",
                    "humidite_pct", "precipitations_mm", "vent_kmh",
                    "pression_hpa", "couverture_nuageuse_pct",
                    "rayonnement_solaire_mj", "humidite_sol_fraction",
                ]:
                    val = data.get(field)
                    if val is not None:
                        setattr(existing, field, val)
                obs = existing
            else:
                obs = WeatherObservation(
                    region_id=data["region_id"],
                    horodatage=horodatage,
                    temperature_c=data.get("temperature_c"),
                    temperature_min_c=data.get("temperature_min_c"),
                    temperature_max_c=data.get("temperature_max_c"),
                    humidite_pct=data.get("humidite_pct"),
                    precipitations_mm=data.get("precipitations_mm", 0),
                    vent_kmh=data.get("vent_kmh"),
                    pression_hpa=data.get("pression_hpa"),
                    couverture_nuageuse_pct=data.get("couverture_nuageuse_pct"),
                    rayonnement_solaire_mj=data.get("rayonnement_solaire_mj"),
                    humidite_sol_fraction=data.get("humidite_sol_fraction"),
                    source=data.get("source", "API"),
                    raw_json=data,
                )
                self._db.add(obs)

            await self._db.flush()
            return obs

        except Exception as exc:
            logger.error("WeatherRepo.save_observation : {}", exc)
            raise

    async def save_observations_batch(
        self, data_list: List[Dict[str, Any]]
    ) -> int:
        """Insert batch d'observations météo — retourne le nombre d'insertions."""
        count = 0
        for data in data_list:
            try:
                await self.save_observation(data)
                count += 1
            except Exception as exc:
                logger.warning("Batch météo skip {} : {}", data.get("region_id"), exc)
        await self._db.flush()
        return count

    async def save_ndvi(self, region_id: str, observation_date: date, ndvi_mean: float) -> None:
        """Sauvegarde une valeur NDVI."""
        try:
            stmt = select(GeoNDVI).where(
                and_(
                    GeoNDVI.region_id == region_id,
                    GeoNDVI.observation_date == observation_date,
                )
            )
            result   = await self._db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.ndvi_mean = ndvi_mean
            else:
                ndvi = GeoNDVI(
                    region_id=region_id,
                    observation_date=observation_date,
                    ndvi_mean=ndvi_mean,
                )
                self._db.add(ndvi)

            await self._db.flush()
        except Exception as exc:
            logger.error("WeatherRepo.save_ndvi : {}", exc)

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _obs_to_dict(obs: WeatherObservation) -> Dict[str, Any]:
        return {
            "region_id":            obs.region_id,
            "horodatage":           obs.horodatage.isoformat() if obs.horodatage else None,
            "temperature_c":        float(obs.temperature_c) if obs.temperature_c else None,
            "temperature_min_c":    float(obs.temperature_min_c) if obs.temperature_min_c else None,
            "temperature_max_c":    float(obs.temperature_max_c) if obs.temperature_max_c else None,
            "humidite_pct":         float(obs.humidite_pct) if obs.humidite_pct else None,
            "precipitations_mm":    float(obs.precipitations_mm) if obs.precipitations_mm else 0.0,
            "vent_kmh":             float(obs.vent_kmh) if obs.vent_kmh else None,
            "pression_hpa":         float(obs.pression_hpa) if obs.pression_hpa else None,
            "source":               obs.source,
        }

    async def _get_all_region_ids(self) -> List[str]:
        """Retourne tous les region_ids connus dans la DB météo."""
        try:
            result = await self._db.execute(
                select(WeatherObservation.region_id).distinct()
            )
            return [row[0] for row in result.fetchall()]
        except Exception:
            from src.utils.constants import REGIONS_MADAGASCAR
            return list(REGIONS_MADAGASCAR)