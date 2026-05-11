"""
Repository pour toutes les opérations DB liées à la nutrition.

Méthodes publiques (contrat avec les routers nutrition.py, predictions.py, reports.py) :
  get_statut_actuel(region_id)
  get_disponibilite(region_id)
  get_recettes(region_id, saison, cible, score_min, limit)
  get_recette_by_id(recette_id)
  get_stocks(region_id)
  save_stocks(region_id, data)
  get_alertes(region_id, type_alerte, severite, statut)
  get_saison_soudure(region_id)
  get_gam_trend(region_id, date_debut, date_fin)
  get_national_stats()
  get_backtest_data(region_id, date_debut, date_fin)

Méthodes internes (scheduler, feature_engineering) :
  save_statut(region_id, data)
  save_food_security(region_id, data)
  save_prix(region_id, data)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import and_, cast, desc, func, or_, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import (
    EpidemioAlert,
    FoodPrice,
    HumanitarianStock,
    NutritionFoodSecurity,
    NutritionStatus,
    Recipe,
)


class NutritionRepository:
    """Repository async pour les données nutritionnelles et sécurité alimentaire."""

    def __init__(self, db: AsyncSession):
        self._db = db

    # ─────────────────────────────────────────────
    # READ — Statut nutritionnel
    # ─────────────────────────────────────────────

    async def get_statut_actuel(self, region_id: str) -> Optional[Dict[str, Any]]:
        """
        Retourne le statut nutritionnel le plus récent pour une région.
        Appelé par :
          - GET /nutrition/statut/{region_id}
          - GET /nutrition/carte-risque
          - src/preprocessing/feature_engineering.py (_fetch_nutrition_from_db)
        """
        try:
            stmt = (
                select(NutritionStatus)
                .where(NutritionStatus.region_id == region_id)
                .order_by(desc(NutritionStatus.date_observation))
                .limit(1)
            )
            result = await self._db.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                # Fallback sur estimation si aucune donnée DB
                return await self._estimer_statut(region_id)
            return self._status_to_dict(row)

        except Exception as exc:
            logger.error("NutritionRepo.get_statut_actuel {} : {}", region_id, exc)
            return await self._estimer_statut(region_id)

    async def get_gam_trend(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
    ) -> List[Dict[str, Any]]:
        """
        Retourne la courbe GAM sur une période.
        Appelé par :
          - GET /nutrition/tendance/{region_id}
          - GET /rapports/export/{region_id}
        """
        try:
            stmt = (
                select(
                    NutritionStatus.date_observation,
                    NutritionStatus.gam_pct,
                    NutritionStatus.sam_pct,
                    NutritionStatus.mam_pct,
                    NutritionStatus.stunting_pct,
                    NutritionStatus.classification_who,
                    NutritionStatus.source,
                )
                .where(
                    and_(
                        NutritionStatus.region_id == region_id,
                        NutritionStatus.date_observation >= date_debut,
                        NutritionStatus.date_observation <= date_fin,
                    )
                )
                .order_by(NutritionStatus.date_observation)
            )
            result = await self._db.execute(stmt)
            rows = result.fetchall()

            records = [
                {
                    "date_observation": str(row.date_observation),
                    "gam_pct":          float(row.gam_pct) if row.gam_pct else 0.0,
                    "sam_pct":          float(row.sam_pct) if row.sam_pct else 0.0,
                    "mam_pct":          float(row.mam_pct) if row.mam_pct else 0.0,
                    "stunting_pct":     float(row.stunting_pct) if row.stunting_pct else 0.0,
                    "classification_who": row.classification_who,
                    "source":           row.source,
                    "region_id":        region_id,
                }
                for row in rows
            ]

            # Si aucune données DB → génère une estimation synthétique
            if not records:
                records = await self._generer_trend_synthetique(
                    region_id, date_debut, date_fin
                )

            return records

        except Exception as exc:
            logger.error("NutritionRepo.get_gam_trend {} : {}", region_id, exc)
            return []

    async def get_national_stats(self) -> Dict[str, Any]:
        """
        KPIs nutritionnels nationaux.
        Appelé par GET /nutrition/statistiques/national.
        """
        try:
            today = date.today()

            # Dernières valeurs GAM par région
            subq = (
                select(
                    NutritionStatus.region_id,
                    func.max(NutritionStatus.date_observation).label("max_date"),
                )
                .group_by(NutritionStatus.region_id)
                .subquery()
            )

            stmt_latest = (
                select(
                    NutritionStatus.region_id,
                    NutritionStatus.gam_pct,
                    NutritionStatus.sam_pct,
                    NutritionStatus.classification_who,
                    NutritionStatus.enfants_5ans_affectes,
                )
                .join(
                    subq,
                    and_(
                        NutritionStatus.region_id == subq.c.region_id,
                        NutritionStatus.date_observation == subq.c.max_date,
                    ),
                )
            )
            result_latest = await self._db.execute(stmt_latest)
            rows = result_latest.fetchall()

            # Calculs agrégés
            regions_crise   = [r for r in rows if float(r.gam_pct or 0) >= 15]
            regions_urgence = [r for r in rows if 10 <= float(r.gam_pct or 0) < 15]
            regions_alerte  = [r for r in rows if 5 <= float(r.gam_pct or 0) < 10]

            gam_moyen = (
                sum(float(r.gam_pct or 0) for r in rows) / len(rows)
                if rows else 0.0
            )
            total_enfants_malnutris = sum(
                int(r.enfants_5ans_affectes or 0) for r in rows
            )

            # Alertes nutrition actives
            stmt_alertes = (
                select(func.count())
                .select_from(EpidemioAlert)
                .where(
                    and_(
                        EpidemioAlert.statut == "active",
                        EpidemioAlert.domaine == "nutrition",
                    )
                )
            )
            result_alertes = await self._db.execute(stmt_alertes)
            nb_alertes = int(result_alertes.scalar() or 0)

            # Stocks critiques (jours_couverture_sam < 30j)
            stmt_stocks = (
                select(func.count())
                .select_from(HumanitarianStock)
                .where(
                    and_(
                        HumanitarianStock.jours_couverture_sam < 30,
                        HumanitarianStock.date_inventaire >= today - timedelta(days=30),
                    )
                )
            )
            result_stocks = await self._db.execute(stmt_stocks)
            stocks_critiques = int(result_stocks.scalar() or 0)

            return {
                "date_calcul":               today.isoformat(),
                "regions_avec_donnees":      len(rows),
                "gam_moyen_national_pct":    round(gam_moyen, 2),
                "regions_en_crise":          len(regions_crise),      # GAM ≥ 15%
                "regions_en_urgence":        len(regions_urgence),    # GAM 10-15%
                "regions_en_alerte":         len(regions_alerte),     # GAM 5-10%
                "total_enfants_malnutris":   total_enfants_malnutris,
                "alertes_nutrition_actives": nb_alertes,
                "regions_stocks_critiques":  stocks_critiques,
                "seuils_oms": {
                    "crise":   "GAM ≥ 15%",
                    "urgence": "GAM 10-15%",
                    "alerte":  "GAM 5-10%",
                },
            }

        except Exception as exc:
            logger.error("NutritionRepo.get_national_stats : {}", exc)
            return {"erreur": str(exc)}

    # ─────────────────────────────────────────────
    # READ — Disponibilité alimentaire
    # ─────────────────────────────────────────────

    async def get_disponibilite(self, region_id: str) -> Optional[Dict[str, Any]]:
        """
        Retourne les données de disponibilité alimentaire (FCS, HDDS, prix).
        Appelé par GET /nutrition/disponibilite/{region_id}.
        """
        try:
            # FCS + HDDS
            stmt_fcs = (
                select(NutritionFoodSecurity)
                .where(NutritionFoodSecurity.region_id == region_id)
                .order_by(desc(NutritionFoodSecurity.date_observation))
                .limit(1)
            )
            result_fcs = await self._db.execute(stmt_fcs)
            fcs_row = result_fcs.scalar_one_or_none()

            # Prix alimentaires
            stmt_prix = (
                select(FoodPrice)
                .where(FoodPrice.region_id == region_id)
                .order_by(desc(FoodPrice.date_observation))
                .limit(1)
            )
            result_prix = await self._db.execute(stmt_prix)
            prix_row = result_prix.scalar_one_or_none()

            if fcs_row is None and prix_row is None:
                # Fallback NutritionFetcher
                return await self._estimer_disponibilite(region_id)

            result: Dict[str, Any] = {
                "region_id":        region_id,
                "date_observation": str(date.today()),
                "source":           "DB interne",
            }

            if fcs_row:
                result.update({
                    "score_fcs":        float(fcs_row.score_fcs or 35),
                    "classification_fcs": fcs_row.classification_fcs or "limite",
                    "hdds":             float(fcs_row.hdds or 5),
                    "rcsi":             float(fcs_row.rcsi or 0),
                    "disponibilite_cereales":           fcs_row.disponibilite_cereales or 2,
                    "disponibilite_legumineuses":       fcs_row.disponibilite_legumineuses or 2,
                    "disponibilite_proteines_animales": fcs_row.disponibilite_proteines_animales or 1,
                    "disponibilite_legumes":            fcs_row.disponibilite_legumes or 2,
                    "disponibilite_fruits":             fcs_row.disponibilite_fruits or 2,
                    "date_observation": str(fcs_row.date_observation),
                })

            if prix_row:
                result.update({
                    "prix_riz_kg":        float(prix_row.prix_riz_kg) if prix_row.prix_riz_kg else None,
                    "prix_manioc_kg":     float(prix_row.prix_manioc_kg) if prix_row.prix_manioc_kg else None,
                    "prix_mais_kg":       float(prix_row.prix_mais_kg) if prix_row.prix_mais_kg else None,
                    "prix_haricots_kg":   float(prix_row.prix_haricots_kg) if prix_row.prix_haricots_kg else None,
                    "prix_huile_litre":   float(prix_row.prix_huile_litre) if prix_row.prix_huile_litre else None,
                    "variation_prix_pct_1m": float(prix_row.variation_prix_pct_1m) if prix_row.variation_prix_pct_1m else None,
                })

            return result

        except Exception as exc:
            logger.error("NutritionRepo.get_disponibilite {} : {}", region_id, exc)
            return await self._estimer_disponibilite(region_id)

    # ─────────────────────────────────────────────
    # READ — Recettes
    # ─────────────────────────────────────────────

    async def get_recettes(
        self,
        region_id: Optional[str] = None,
        saison: Optional[str] = None,
        cible: Optional[str] = None,
        score_min: float = 60.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retourne des recettes nutritionnelles filtrées.
        Appelé par GET /nutrition/recettes.
        """
        try:
            stmt = (
                select(Recipe)
                .where(
                    and_(
                        Recipe.actif == True,
                        Recipe.score_nutritionnel >= score_min,
                    )
                )
                .order_by(desc(Recipe.score_nutritionnel))
                .limit(limit)
            )

            result = await self._db.execute(stmt)
            rows = result.scalars().all()

            # Filtrage post-query sur JSONB (région, saison, cible)
            filtered = []
            for r in rows:
                if region_id and r.regions_adaptees:
                    if region_id not in r.regions_adaptees and "toutes" not in r.regions_adaptees:
                        continue
                if saison and r.saison:
                    if saison not in r.saison and "toute_annee" not in r.saison:
                        continue
                if cible and r.cible:
                    if cible not in r.cible and "famille" not in r.cible:
                        continue
                filtered.append(self._recipe_to_dict(r))

            if not filtered:
                # Fallback : retourne des recettes synthétiques
                filtered = self._recettes_synthetiques(region_id, cible, limit)

            return filtered

        except Exception as exc:
            logger.error("NutritionRepo.get_recettes : {}", exc)
            return self._recettes_synthetiques(region_id, cible, limit)

    async def get_recette_by_id(self, recette_id: str) -> Optional[Dict[str, Any]]:
        """
        Retourne une recette par son ID.
        Appelé par GET /nutrition/recettes/{recette_id}.
        """
        try:
            stmt = select(Recipe).where(Recipe.recette_id == recette_id)
            result = await self._db.execute(stmt)
            row = result.scalar_one_or_none()
            return self._recipe_to_dict(row) if row else None
        except Exception as exc:
            logger.error("NutritionRepo.get_recette_by_id {} : {}", recette_id, exc)
            return None

    # ─────────────────────────────────────────────
    # READ — Stocks humanitaires
    # ─────────────────────────────────────────────

    async def get_stocks(self, region_id: str) -> Optional[Dict[str, Any]]:
        """
        Retourne les stocks humanitaires les plus récents pour une région.
        Appelé par GET /nutrition/stocks/{region_id}.
        """
        try:
            stmt = (
                select(HumanitarianStock)
                .where(HumanitarianStock.region_id == region_id)
                .order_by(desc(HumanitarianStock.date_inventaire))
                .limit(1)
            )
            result = await self._db.execute(stmt)
            row = result.scalar_one_or_none()

            if row is None:
                return self._stocks_synthetiques(region_id)

            return self._stocks_to_dict(row)

        except Exception as exc:
            logger.error("NutritionRepo.get_stocks {} : {}", region_id, exc)
            return self._stocks_synthetiques(region_id)

    # ─────────────────────────────────────────────
    # READ — Alertes nutrition
    # ─────────────────────────────────────────────

    async def get_alertes(
        self,
        region_id: Optional[str] = None,
        type_alerte: Optional[str] = None,
        severite: Optional[str] = None,
        statut: str = "active",
    ) -> List[Dict[str, Any]]:
        """
        Retourne les alertes nutrition actives.
        Appelé par GET /nutrition/alertes.
        """
        try:
            conditions = [EpidemioAlert.domaine == "nutrition"]
            if region_id:
                conditions.append(EpidemioAlert.region_id == region_id)
            if type_alerte:
                conditions.append(EpidemioAlert.type_alerte == type_alerte)
            if severite:
                conditions.append(EpidemioAlert.severite == severite)
            if statut and statut != "all":
                conditions.append(EpidemioAlert.statut == statut)

            stmt = (
                select(EpidemioAlert)
                .where(and_(*conditions))
                .order_by(
                    desc(
                        func.case(
                            (EpidemioAlert.severite == "crise", 4),
                            (EpidemioAlert.severite == "urgence", 3),
                            (EpidemioAlert.severite == "alerte", 2),
                            else_=1,
                        )
                    ),
                    desc(EpidemioAlert.date_detection),
                )
                .limit(200)
            )
            result = await self._db.execute(stmt)
            rows = result.scalars().all()

            return [
                {
                    "alerte_id":          r.alerte_id,
                    "region_id":          r.region_id,
                    "region_name":        r.region_name or r.region_id,
                    "type_alerte":        r.type_alerte,
                    "severite":           r.severite,
                    "indicateur_declencheur": r.indicateur_declencheur,
                    "valeur_actuelle":    float(r.valeur_actuelle) if r.valeur_actuelle else None,
                    "seuil_alerte":       float(r.seuil_depasse) if r.seuil_depasse else None,
                    "population_affectee": r.population_affectee,
                    "enfants_a_risque":   r.enfants_a_risque,
                    "date_detection":     r.date_detection.isoformat() if r.date_detection else None,
                    "statut":             r.statut,
                    "actions_requises":   r.actions_requises or [],
                }
                for r in rows
            ]

        except Exception as exc:
            logger.error("NutritionRepo.get_alertes : {}", exc)
            return []

    # ─────────────────────────────────────────────
    # READ — Soudure
    # ─────────────────────────────────────────────

    async def get_saison_soudure(
        self, region_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retourne le statut des périodes de soudure par région.
        Appelé par GET /nutrition/soudure.
        """
        try:
            from src.data_collection.nutrition_fetcher import NutritionFetcher
            fetcher = NutritionFetcher()
            result  = await fetcher.get_statut_soudure(region_id)
            return result
        except Exception as exc:
            logger.error("NutritionRepo.get_saison_soudure : {}", exc)
            return []

    # ─────────────────────────────────────────────
    # READ — Backtesting
    # ─────────────────────────────────────────────

    async def get_backtest_data(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
    ) -> Dict[str, Any]:
        """
        Données de backtesting modèle nutrition (prédictions vs GAM réel).
        Appelé par GET /predictions/backtest/{region_id}?modele=nutrition.
        """
        try:
            from src.database.models import MLPrediction
            import numpy as np

            stmt_pred = (
                select(MLPrediction)
                .where(
                    and_(
                        MLPrediction.region_id == region_id,
                        MLPrediction.modele_nom == "nutrition",
                        MLPrediction.date_prediction >= datetime.combine(date_debut, datetime.min.time()),
                        MLPrediction.date_prediction <= datetime.combine(date_fin, datetime.max.time()),
                        MLPrediction.valeur_reelle.is_not(None),
                    )
                )
                .order_by(MLPrediction.date_prediction)
            )
            result_pred = await self._db.execute(stmt_pred)
            predictions = result_pred.scalars().all()

            if not predictions:
                return {
                    "region_id": region_id, "periode_debut": str(date_debut),
                    "periode_fin": str(date_fin), "modele": "nutrition",
                    "mae": 0.0, "rmse": 0.0, "mape_pct": 0.0,
                    "correlation": 0.0, "biais": 0.0,
                    "nb_predictions": 0, "predictions_vs_reel": [],
                }

            pred_values = [float(p.score_nutrition or 0) for p in predictions]
            real_values = [float(p.valeur_reelle or 0) for p in predictions]

            errors     = [abs(p - r) for p, r in zip(pred_values, real_values)]
            sq_errors  = [(p - r) ** 2 for p, r in zip(pred_values, real_values)]
            pct_errors = [abs(p - r) / max(r, 0.01) * 100 for p, r in zip(pred_values, real_values)]

            mae  = round(float(np.mean(errors)), 4)
            rmse = round(float(np.sqrt(np.mean(sq_errors))), 4)
            mape = round(float(np.mean(pct_errors)), 2)
            biais = round(float(np.mean([p - r for p, r in zip(pred_values, real_values)])), 4)
            corr = round(float(np.corrcoef(pred_values, real_values)[0, 1]), 4) \
                if len(pred_values) > 2 else 0.0

            return {
                "region_id":    region_id,
                "periode_debut": str(date_debut),
                "periode_fin":   str(date_fin),
                "modele":       "nutrition",
                "mae":          mae,
                "rmse":         rmse,
                "mape_pct":     mape,
                "correlation":  corr,
                "biais":        biais,
                "nb_predictions": len(predictions),
                "predictions_vs_reel": [
                    {
                        "date":           p.date_prediction.isoformat(),
                        "score_predit":   float(p.score_nutrition or 0),
                        "valeur_reelle":  float(p.valeur_reelle or 0),
                        "erreur_absolue": round(abs(float(p.score_nutrition or 0) - float(p.valeur_reelle or 0)), 4),
                    }
                    for p in predictions[:100]
                ],
            }

        except Exception as exc:
            logger.error("NutritionRepo.get_backtest_data {} : {}", region_id, exc)
            return {"region_id": region_id, "erreur": str(exc)}

    # ─────────────────────────────────────────────
    # WRITE
    # ─────────────────────────────────────────────

    async def save_statut(
        self, region_id: str, data: Dict[str, Any]
    ) -> NutritionStatus:
        """Upsert statut nutritionnel. Appelé par le scheduler Celery."""
        try:
            obs_date = data.get("date_observation")
            if isinstance(obs_date, str):
                obs_date = date.fromisoformat(obs_date[:10])
            obs_date = obs_date or date.today()

            stmt = select(NutritionStatus).where(
                and_(
                    NutritionStatus.region_id == region_id,
                    NutritionStatus.date_observation == obs_date,
                )
            )
            result = await self._db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                for field in ["gam_pct", "sam_pct", "mam_pct", "stunting_pct",
                               "underweight_pct", "classification_who", "source"]:
                    val = data.get(field)
                    if val is not None:
                        setattr(existing, field, val)
                obj = existing
            else:
                gam = data.get("gam_pct", 0)
                obj = NutritionStatus(
                    region_id=region_id,
                    region_name=data.get("region_name"),
                    date_observation=obs_date,
                    date_enquete=data.get("date_enquete"),
                    gam_pct=gam,
                    sam_pct=data.get("sam_pct", round(gam * 0.28, 2)),
                    mam_pct=data.get("mam_pct", round(gam * 0.72, 2)),
                    stunting_pct=data.get("stunting_pct"),
                    underweight_pct=data.get("underweight_pct"),
                    enfants_5ans_affectes=data.get("enfants_5ans_affectes"),
                    femmes_enceintes_malnutries=data.get("femmes_enceintes_malnutries"),
                    classification_who=data.get("classification_who",
                                                 self._classifier_gam(gam)),
                    tendance_vs_periode_prec=data.get("tendance_vs_periode_prec", "stable"),
                    fiabilite_donnees=data.get("fiabilite_donnees", "estimée"),
                    source=data.get("source", "collecte auto"),
                    raw_json=data,
                )
                self._db.add(obj)

            await self._db.flush()
            return obj

        except Exception as exc:
            logger.error("NutritionRepo.save_statut {} : {}", region_id, exc)
            raise

    async def save_food_security(
        self, region_id: str, data: Dict[str, Any]
    ) -> NutritionFoodSecurity:
        """Upsert données FCS/HDDS/rCSI."""
        try:
            obs_date = data.get("date_observation")
            if isinstance(obs_date, str):
                obs_date = date.fromisoformat(obs_date[:10])
            obs_date = obs_date or date.today()

            stmt = select(NutritionFoodSecurity).where(
                and_(
                    NutritionFoodSecurity.region_id == region_id,
                    NutritionFoodSecurity.date_observation == obs_date,
                )
            )
            result = await self._db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                for field in ["score_fcs", "hdds", "rcsi", "classification_fcs",
                               "disponibilite_cereales", "disponibilite_legumineuses",
                               "disponibilite_proteines_animales", "disponibilite_legumes",
                               "disponibilite_fruits"]:
                    val = data.get(field)
                    if val is not None:
                        setattr(existing, field, val)
                obj = existing
            else:
                fcs = data.get("score_fcs", 35)
                obj = NutritionFoodSecurity(
                    region_id=region_id,
                    date_observation=obs_date,
                    score_fcs=fcs,
                    classification_fcs=data.get("classification_fcs",
                                                  self._classifier_fcs(fcs)),
                    hdds=data.get("hdds"),
                    rcsi=data.get("rcsi"),
                    disponibilite_cereales=data.get("disponibilite_cereales", 2),
                    disponibilite_legumineuses=data.get("disponibilite_legumineuses", 2),
                    disponibilite_proteines_animales=data.get("disponibilite_proteines_animales", 1),
                    disponibilite_legumes=data.get("disponibilite_legumes", 2),
                    disponibilite_fruits=data.get("disponibilite_fruits", 2),
                    source=data.get("source", "WFP VAM"),
                )
                self._db.add(obj)

            await self._db.flush()
            return obj

        except Exception as exc:
            logger.error("NutritionRepo.save_food_security {} : {}", region_id, exc)
            raise

    async def save_stocks(
        self, region_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Insère un inventaire de stocks humanitaires.
        Appelé par POST /nutrition/stocks/{region_id}.
        Retourne un dict avec l'id créé.
        """
        try:
            inv_date = data.get("date_inventaire")
            if isinstance(inv_date, str):
                inv_date = date.fromisoformat(inv_date[:10])
            inv_date = inv_date or date.today()

            stock = HumanitarianStock(
                region_id=region_id,
                date_inventaire=inv_date,
                rutf_sachets=data.get("rutf_sachets", 0),
                rusf_sachets=data.get("rusf_sachets", 0),
                plumpy_nut_sachets=data.get("plumpy_nut_sachets", 0),
                spiruline_kg=data.get("spiruline_kg", 0),
                sel_iode_kg=data.get("sel_iode_kg", 0),
                vitamine_a_capsules=data.get("vitamine_a_capsules", 0),
                fer_folate_comprimes=data.get("fer_folate_comprimes", 0),
                zinc_comprimes=data.get("zinc_comprimes", 0),
                jours_couverture_sam=data.get("jours_couverture_sam", 0),
                jours_couverture_mam=data.get("jours_couverture_mam", 0),
                statut_stock=data.get("statut_stock"),
                derniere_livraison=data.get("derniere_livraison"),
                prochaine_livraison_prevue=data.get("prochaine_livraison_prevue"),
            )
            self._db.add(stock)
            await self._db.flush()

            return {"id": stock.id, "region_id": region_id, "date": str(inv_date)}

        except Exception as exc:
            logger.error("NutritionRepo.save_stocks {} : {}", region_id, exc)
            raise

    async def save_prix(
        self, region_id: str, data: Dict[str, Any]
    ) -> FoodPrice:
        """Upsert prix alimentaires."""
        try:
            obs_date = data.get("date_observation")
            if isinstance(obs_date, str):
                obs_date = date.fromisoformat(obs_date[:10])
            obs_date = obs_date or date.today()

            stmt = select(FoodPrice).where(
                and_(
                    FoodPrice.region_id == region_id,
                    FoodPrice.date_observation == obs_date,
                )
            )
            result = await self._db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                for field in ["prix_riz_kg", "prix_manioc_kg", "prix_mais_kg",
                               "prix_haricots_kg", "prix_huile_litre", "variation_prix_pct_1m"]:
                    val = data.get(field)
                    if val is not None:
                        setattr(existing, field, val)
                obj = existing
            else:
                obj = FoodPrice(
                    region_id=region_id,
                    date_observation=obs_date,
                    prix_riz_kg=data.get("prix_riz_kg"),
                    prix_manioc_kg=data.get("prix_manioc_kg"),
                    prix_mais_kg=data.get("prix_mais_kg"),
                    prix_haricots_kg=data.get("prix_haricots_kg"),
                    prix_huile_litre=data.get("prix_huile_litre"),
                    variation_prix_pct_1m=data.get("variation_prix_pct_1m"),
                    source=data.get("source", "WFP VAM"),
                )
                self._db.add(obj)

            await self._db.flush()
            return obj

        except Exception as exc:
            logger.error("NutritionRepo.save_prix {} : {}", region_id, exc)
            raise

    async def save_alerte(self, data: Dict[str, Any]) -> EpidemioAlert:
        """Insère une alerte nutrition (idempotent)."""
        try:
            alerte_id = data.get("alerte_id", str(uuid.uuid4()))
            stmt = select(EpidemioAlert).where(EpidemioAlert.alerte_id == alerte_id)
            result = await self._db.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing:
                return existing

            date_detection = data.get("date_detection")
            if isinstance(date_detection, str):
                date_detection = datetime.fromisoformat(date_detection.replace("Z", "+00:00"))

            alert = EpidemioAlert(
                alerte_id=alerte_id,
                region_id=data["region_id"],
                region_name=data.get("region_name"),
                type_alerte=data["type_alerte"],
                severite=data["severite"],
                domaine="nutrition",
                seuil_depasse=data.get("seuil_alerte"),
                valeur_actuelle=data.get("valeur_actuelle"),
                indicateur_declencheur=data.get("indicateur_declencheur"),
                population_affectee=data.get("population_affectee"),
                enfants_a_risque=data.get("enfants_a_risque"),
                date_detection=date_detection or datetime.utcnow(),
                statut=data.get("statut", "active"),
                description=data.get("description"),
                actions_requises=data.get("actions_requises"),
            )
            self._db.add(alert)
            await self._db.flush()
            return alert

        except Exception as exc:
            logger.error("NutritionRepo.save_alerte : {}", exc)
            raise

    # ─────────────────────────────────────────────
    # Helpers privés
    # ─────────────────────────────────────────────

    async def _estimer_statut(self, region_id: str) -> Optional[Dict[str, Any]]:
        """Estimation du statut nutritionnel quand la DB est vide."""
        try:
            from src.data_collection.nutrition_fetcher import NutritionFetcher
            fetcher = NutritionFetcher()
            return fetcher._estimer_statut_nutritionnel(region_id)
        except Exception:
            return None

    async def _estimer_disponibilite(self, region_id: str) -> Optional[Dict[str, Any]]:
        """Estimation disponibilité quand la DB est vide."""
        try:
            from src.data_collection.nutrition_fetcher import NutritionFetcher
            fetcher = NutritionFetcher()
            return await fetcher.get_disponibilite_complete(region_id)
        except Exception as exc:
            logger.debug("Estimation disponibilite {} : {}", region_id, exc)
            return None

    async def _generer_trend_synthetique(
        self, region_id: str, date_debut: date, date_fin: date
    ) -> List[Dict[str, Any]]:
        """Génère une courbe GAM synthétique quand la DB est vide."""
        try:
            from src.data_collection.nutrition_fetcher import (
                REGION_FOOD_PROFILE, DEFAULT_FOOD_PROFILE
            )
            import math

            profile = REGION_FOOD_PROFILE.get(region_id, DEFAULT_FOOD_PROFILE)
            vuln    = profile.get("indice_vulnerabilite", 0.5)
            gam_base = 5 + vuln * 12

            records = []
            current = date_debut
            step    = timedelta(days=30)
            i = 0
            while current <= date_fin:
                # Variation saisonnière synthétique
                mois = current.month
                en_soudure = mois in profile.get("mois_soudure", [11, 12])
                gam = gam_base * (1.3 if en_soudure else 1.0)
                gam += 1.5 * math.sin(2 * math.pi * i / 6)  # cycle semi-annuel
                gam = max(1.0, min(25.0, gam))

                records.append({
                    "date_observation": str(current),
                    "gam_pct":          round(gam, 2),
                    "sam_pct":          round(gam * 0.28, 2),
                    "mam_pct":          round(gam * 0.72, 2),
                    "stunting_pct":     round(gam * 2.3, 1),
                    "classification_who": self._classifier_gam(gam),
                    "source":           "Estimation synthétique",
                    "region_id":        region_id,
                })
                current += step
                i += 1
            return records
        except Exception:
            return []

    @staticmethod
    def _status_to_dict(s: NutritionStatus) -> Dict[str, Any]:
        return {
            "region_id":            s.region_id,
            "region_name":          s.region_name or s.region_id,
            "date_enquete":         str(s.date_enquete) if s.date_enquete else str(s.date_observation),
            "source":               s.source or "DB interne",
            "gam_pct":              float(s.gam_pct) if s.gam_pct is not None else 0.0,
            "sam_pct":              float(s.sam_pct) if s.sam_pct is not None else 0.0,
            "mam_pct":              float(s.mam_pct) if s.mam_pct is not None else 0.0,
            "stunting_pct":         float(s.stunting_pct) if s.stunting_pct is not None else 0.0,
            "underweight_pct":      float(s.underweight_pct) if s.underweight_pct is not None else 0.0,
            "enfants_5ans_affectes":      s.enfants_5ans_affectes or 0,
            "femmes_enceintes_malnutries": s.femmes_enceintes_malnutries or 0,
            "classification_who":   s.classification_who or "acceptable",
            "tendance_vs_periode_prec": s.tendance_vs_periode_prec or "stable",
            "fiabilite_donnees":    s.fiabilite_donnees or "estimée",
        }

    @staticmethod
    def _stocks_to_dict(s: HumanitarianStock) -> Dict[str, Any]:
        return {
            "region_id":               s.region_id,
            "date_inventaire":         str(s.date_inventaire),
            "rutf_sachets":            s.rutf_sachets or 0,
            "rusf_sachets":            s.rusf_sachets or 0,
            "plumpy_nut_sachets":      s.plumpy_nut_sachets or 0,
            "spiruline_kg":            float(s.spiruline_kg) if s.spiruline_kg else 0.0,
            "sel_iode_kg":             float(s.sel_iode_kg) if s.sel_iode_kg else 0.0,
            "vitamine_a_capsules":     s.vitamine_a_capsules or 0,
            "fer_folate_comprimes":    s.fer_folate_comprimes or 0,
            "zinc_comprimes":          s.zinc_comprimes or 0,
            "jours_couverture_sam":    float(s.jours_couverture_sam) if s.jours_couverture_sam else 0.0,
            "jours_couverture_mam":    float(s.jours_couverture_mam) if s.jours_couverture_mam else 0.0,
            "statut_stock":            s.statut_stock or "adéquat",
            "derniere_livraison":      str(s.derniere_livraison) if s.derniere_livraison else None,
            "prochaine_livraison_prevue": str(s.prochaine_livraison_prevue) if s.prochaine_livraison_prevue else None,
        }

    @staticmethod
    def _recipe_to_dict(r: Recipe) -> Dict[str, Any]:
        return {
            "recette_id":           r.recette_id,
            "nom":                  r.nom,
            "nom_malgache":         r.nom_malgache,
            "region_adaptee":       r.regions_adaptees or [],
            "saison":               r.saison or [],
            "calories_kcal":        float(r.calories_kcal) if r.calories_kcal else 0.0,
            "proteines_g":          float(r.proteines_g) if r.proteines_g else 0.0,
            "glucides_g":           float(r.glucides_g) if r.glucides_g else 0.0,
            "lipides_g":            float(r.lipides_g) if r.lipides_g else 0.0,
            "fer_mg":               float(r.fer_mg) if r.fer_mg else 0.0,
            "vitamine_a_ug":        float(r.vitamine_a_ug) if r.vitamine_a_ug else 0.0,
            "zinc_mg":              float(r.zinc_mg) if r.zinc_mg else 0.0,
            "score_nutritionnel":   float(r.score_nutritionnel) if r.score_nutritionnel else 0.0,
            "ingredients":          r.ingredients or [],
            "instructions":         r.instructions or "",
            "temps_preparation_min":r.temps_preparation_min or 30,
            "cout_estime_ariary":   float(r.cout_estime_ariary) if r.cout_estime_ariary else None,
            "cible":                r.cible or [],
            "image_url":            r.image_url,
        }

    @staticmethod
    def _classifier_gam(gam: float) -> str:
        if gam < 5:   return "acceptable"
        if gam < 10:  return "alerte"
        if gam < 15:  return "urgence"
        return "crise"

    @staticmethod
    def _classifier_fcs(fcs: float) -> str:
        if fcs < 21:  return "pauvre"
        if fcs < 35:  return "limite"
        return "acceptable"

    @staticmethod
    def _stocks_synthetiques(region_id: str) -> Dict[str, Any]:
        """Stocks synthétiques par défaut quand aucune donnée disponible."""
        return {
            "region_id": region_id, "date_inventaire": str(date.today()),
            "rutf_sachets": 0, "rusf_sachets": 0, "plumpy_nut_sachets": 0,
            "spiruline_kg": 0.0, "sel_iode_kg": 0.0,
            "vitamine_a_capsules": 0, "fer_folate_comprimes": 0, "zinc_comprimes": 0,
            "jours_couverture_sam": 0.0, "jours_couverture_mam": 0.0,
            "statut_stock": "données non disponibles",
            "derniere_livraison": None, "prochaine_livraison_prevue": None,
        }

    @staticmethod
    def _recettes_synthetiques(
        region_id: Optional[str], cible: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """Retourne des recettes de base si la DB est vide."""
        recettes_base = [
            {
                "recette_id": "synth-001",
                "nom": "Bouillie enrichie au haricot (Misovola)",
                "nom_malgache": "Misovola",
                "region_adaptee": ["toutes"],
                "saison": ["toute_annee"],
                "calories_kcal": 180.0, "proteines_g": 8.5, "glucides_g": 28.0,
                "lipides_g": 3.5, "fer_mg": 3.2, "vitamine_a_ug": 120.0, "zinc_mg": 1.8,
                "score_nutritionnel": 75.0,
                "ingredients": [
                    {"nom": "Farine de riz", "quantite_g": 50, "disponible_localement": True},
                    {"nom": "Haricots rouges cuits", "quantite_g": 40, "disponible_localement": True},
                    {"nom": "Huile végétale", "quantite_g": 10, "disponible_localement": True},
                    {"nom": "Sucre", "quantite_g": 10, "disponible_localement": True},
                ],
                "instructions": "Cuire la farine de riz dans l'eau. Ajouter les haricots écrasés. Incorporer l'huile et le sucre. Remuer 10 min à feu doux.",
                "temps_preparation_min": 20,
                "cout_estime_ariary": 800.0,
                "cible": ["enfants_6_23m", "enfants_2_5ans"],
                "image_url": None,
            },
            {
                "recette_id": "synth-002",
                "nom": "Soupe de légumes au poisson séché",
                "nom_malgache": "Laoka trondro sy anana",
                "region_adaptee": ["toutes"],
                "saison": ["toute_annee"],
                "calories_kcal": 220.0, "proteines_g": 18.0, "glucides_g": 15.0,
                "lipides_g": 7.0, "fer_mg": 5.5, "vitamine_a_ug": 350.0, "zinc_mg": 2.5,
                "score_nutritionnel": 82.0,
                "ingredients": [
                    {"nom": "Poisson séché", "quantite_g": 50, "disponible_localement": True},
                    {"nom": "Légumes verts (anana)", "quantite_g": 100, "disponible_localement": True},
                    {"nom": "Tomate", "quantite_g": 50, "disponible_localement": True},
                    {"nom": "Huile", "quantite_g": 10, "disponible_localement": True},
                ],
                "instructions": "Faire revenir oignon et tomate. Ajouter eau et poisson. Cuire 15 min. Ajouter légumes verts et cuire 5 min.",
                "temps_preparation_min": 25,
                "cout_estime_ariary": 1200.0,
                "cible": ["famille", "femmes_enceintes"],
                "image_url": None,
            },
        ]

        # Filtrage basique par cible
        if cible:
            recettes_base = [
                r for r in recettes_base
                if cible in r["cible"] or "famille" in r["cible"]
            ]

        return recettes_base[:limit]