import asyncio
from src.data_collection.weather_fetcher import WeatherFetcher
from src.database import AsyncSessionLocal
from src.database.models import WeatherObservation
from datetime import datetime

async def run():
    fetcher = WeatherFetcher()

    try:
        async with AsyncSessionLocal() as session:
            data = await fetcher.get_all_regions_current()

            for item in data:
                if "erreur" in item:
                    continue

                session.add(
                    WeatherObservation(
                        region_code=item["region_id"],
                        timestamp_utc=datetime.fromisoformat(item["horodatage"]),
                        temperature_c=item["temperature_c"],
                        temp_min_c=item.get("temperature_min_c"),
                        temp_max_c=item.get("temperature_max_c"),
                        humidite_pct=item["humidite_pct"],
                        precipitation_mm=item["precipitations_mm"],
                        vitesse_vent_kmh=item["vent_kmh"],
                        source_api=item["source"],
                        raw_payload=item,
                        qualite_flag=0
                    )
                )

            await session.commit()

    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(run())