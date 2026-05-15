import asyncio

from src.data_collection.nutrition_fetcher import NutritionFetcher
from src.database import AsyncSessionLocal
from src.database.models import NutritionStatus


async def run():
    fetcher = NutritionFetcher()

    try:
        data = await fetcher.fetch_food_prices()

        async with AsyncSessionLocal() as session:

            for item in data["results"]:

                session.add(
                    NutritionStatus(
                        region_code=item["region"],
                        prix_riz=item["rice_price"],
                        prix_mais=item["maize_price"],
                        malnutrition_rate=item.get("gam_rate"),
                        source="WFP"
                    )
                )

            await session.commit()

    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(run())