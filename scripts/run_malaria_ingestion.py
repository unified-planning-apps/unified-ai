import asyncio
from datetime import date, timedelta

from sqlalchemy import select

from src.data_collection.malaria_fetcher import MalariaFetcher
from src.database import AsyncSessionLocal
from src.database.models import MalariaCase


async def run():
    fetcher = MalariaFetcher()

    try:
        # 8 dernières semaines
        date_fin = date.today()
        date_debut = date_fin - timedelta(weeks=8)

        # Collecte toutes les régions
        data = await fetcher.get_cas_toutes_regions(
            date_debut=date_debut,
            date_fin=date_fin,
        )

        total_imported = 0

        async with AsyncSessionLocal() as session:

            for region_id, records in data.items():

                for record in records:

                    # Vérifie si déjà existant
                    stmt = select(MalariaCase).where(
                        MalariaCase.region_id == record["region_id"],
                        MalariaCase.annee == record["annee"],
                        MalariaCase.semaine_epidemio == record["semaine_epidemio"],
                    )

                    existing = await session.scalar(stmt)

                    if existing:
                        continue

                    malaria_case = MalariaCase(
                        region_id=record["region_id"],
                        district=record.get("district"),

                        annee=record["annee"],
                        semaine_epidemio=record["semaine_epidemio"],
                        date_rapport=record["date_rapport"],

                        cas_confirmes=record.get("cas_confirmes", 0),
                        cas_confirmes_mixte=record.get("cas_confirmes_mixte", 0),
                        deces=record.get("deces", 0),
                        hospitalisations=record.get("hospitalisations", 0),

                        tests_malaria=record.get("tests_malaria", 0),
                        tdr_positifs=record.get("tdr_positifs", 0),

                        tdr_negatifs=record.get(
                            "tdr_negatifs",
                            0
                        ),

                        taux_incidence_pour_mille=record.get(
                            "taux_incidence_pour_mille",
                            0
                        ),

                        taux_positivite_tdr_pct=record.get(
                            "taux_positivite_tdr_pct",
                            0
                        ),

                        taux_letalite_pct=record.get(
                            "taux_letalite_pct",
                            0
                        ),

                        population_a_risque=record.get(
                            "population_a_risque"
                        ),

                        source=record.get("source", "DHIS2"),

                        fiabilite_donnees=record.get(
                            "fiabilite_donnees",
                            "confirmée"
                        ),

                        period_dhis2=record.get("period_dhis2"),

                        raw_json=record,
                    )

                    session.add(malaria_case)
                    total_imported += 1

            await session.commit()

        print(f"{total_imported} lignes malaria importées")

    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(run())