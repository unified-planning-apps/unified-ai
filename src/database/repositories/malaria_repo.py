"""
Repository pour toutes les opérations DB liées au paludisme.

Méthodes publiques (contrat avec les routers malaria.py et predictions.py) :
  get_cas_by_region(region_id, date_debut, date_fin, district, limit, offset)
  get_alertes(region_id, severite, statut)
  get_seasonal_stats(region_id)
  get_weekly_trend(region_id, date_debut, date_fin)
  acquitter_alerte(alerte_id, user_id, commentaire)
  get_national_stats()
  get_backtest_data(region_id, date_debut, date_fin)

Méthodes internes (scheduler, feature_engineering) :
  save_case(data)
  save_cases_batch(data_list)
  save_alert(data)
  save_alerts_batch(alerts)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import and_, desc, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import EpidemioAlert, MalariaCase


class MalariaRepository:
    """Repository async pour les données épidémiologiques paludisme."""

    def __init__(self, db: AsyncSession):
        self._db = db

    # ─────────────────────────────────────────────
    # READ — Cas
    # ─────────────────────────────────────────────

    async def get_cas_by_region(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
        district: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les cas de paludisme pour une région sur une période.
        Appelé par GET /paludisme/cas/{region_id}.
        """
        try:
            conditions = [
                MalariaCase.region_id == region_id,
                MalariaCase.date_rapport >= date_debut,
                MalariaCase.date_rapport <= date_fin,
            ]
            if district:
                conditions.append(MalariaCase.district == district)

            stmt = (
                select(MalariaCase)
                .where(and_(*conditions))
                .order_by(MalariaCase.annee, MalariaCase.semaine_epidemio)
                .limit(limit)
                .offset(offset)
            )
            result = await self._db.execute(stmt)
            rows   = result.scalars().all()

            return [self._case_to_dict(r) for r in rows]

        except Exception as exc:
            logger.error("MalariaRepo.get_cas_by_region {} : {}", region_id, exc)
            return []

    async def get_weekly_trend(
        self,
        region_id: str,
        date_debut: date,
        date_fin: date,
    ) -> List[Dict[str, Any]]:
        """
        Retourne la tendance hebdomadaire des cas.
        Appelé par GET /paludisme/tendance/{region_id}.
        """
        try:
            stmt = (
                select(
                    MalariaCase.annee,
                    MalariaCase.semaine_epidemio,
                    MalariaCase.date_rapport,
                    func.sum(MalariaCase.cas_confirmes).label("cas_confirmes"),
                    func.sum(MalariaCase.cas_confirmes_mixte).label("cas_confirmes_mixte"),
                    func.sum(MalariaCase.deces).label("deces"),
                    func.sum(MalariaCase.hospitalisations).label("hospitalisations"),
                    func.avg(MalariaCase.taux_positivite_tdr_pct).label("taux_positivite_tdr_pct"),
                    func.avg(MalariaCase.taux_incidence_pour_mille).label("taux_incidence_pour_mille"),
                )
                .where(
                    and_(
                        MalariaCase.region_id == region_id,
                        MalariaCase.date_rapport >= date_debut,
                        MalariaCase.date_rapport <= date_fin,
                    )
                )
                .group_by(
                    MalariaCase.annee,
                    MalariaCase.semaine_epidemio,
                    MalariaCase.date_rapport,
                )
                .order_by(MalariaCase.annee, MalariaCase.semaine_epidemio)
            )
            result = await self._db.execute(stmt)
            rows   = result.fetchall()

            return [
                {
                    "annee":            row.annee,
                    "semaine_epidemio": row.semaine_epidemio,
                    "date_rapport":     str(row.date_rapport),
                    "cas_confirmes":    int(row.cas_confirmes or 0),
                    "cas_confirmes_mixte":     int(row.cas_confirmes_mixte or 0),
                    "deces":            int(row.deces or 0),
                    "hospitalisations": int(row.hospitalisations or 0),
                    "taux_positivite_tdr_pct": round(float(row.taux_positivite_tdr_pct or 0), 2),
                    "taux_incidence_pour_mille": round(float(row.taux_incidence_pour_mille or 0), 4),
                    "region_id":        region_id,
                }
                for row in rows
            ]

        except Exception as exc:
            logger.error("MalariaRepo.get_weekly_trend {} : {}", region_id, exc)
            return []

    async def get_seasonal_stats(self, region_id: str) -> Dict[str, Any]:
        """
        Calcule les statistiques saisonnières pour une région.
        Appelé par GET /paludisme/saisonnalite/{region_id}.
        """
        try:
            today   = date.today()
            annee   = today.year
            semaine = today.isocalendar()[1]
            mois    = today.month

            # Saison courante
            from src.utils.constants import get_saison_courante
            saison = get_saison_courante(mois)

            # Cas cumulés de la saison courante (13 dernières semaines)
            date_debut_saison = today - timedelta(weeks=13)
            stmt_cumul = (
                select(func.sum(MalariaCase.cas_confirmes))
                .where(
                    and_(
                        MalariaCase.region_id == region_id,
                        MalariaCase.date_rapport >= date_debut_saison,
                        MalariaCase.date_rapport <= today,
                    )
                )
            )
            result_cumul = await self._db.execute(stmt_cumul)
            cas_cumules  = int(result_cumul.scalar() or 0)

            # Même période de l'année précédente
            date_debut_prec = date_debut_saison.replace(year=date_debut_saison.year - 1)
            date_fin_prec   = today.replace(year=today.year - 1)
            stmt_prec = (
                select(func.sum(MalariaCase.cas_confirmes))
                .where(
                    and_(
                        MalariaCase.region_id == region_id,
                        MalariaCase.date_rapport >= date_debut_prec,
                        MalariaCase.date_rapport <= date_fin_prec,
                    )
                )
            )
            result_prec = await self._db.execute(stmt_prec)
            cas_prec    = int(result_prec.scalar() or 0)

            variation = (
                round((cas_cumules - cas_prec) / max(1, cas_prec) * 100, 1)
                if cas_prec > 0 else 0.0
            )

            # Semaine de pic historique (moyenne sur 3 ans)
            annee_debut_hist = annee - 3
            stmt_hist = (
                select(
                    MalariaCase.semaine_epidemio,
                    func.avg(MalariaCase.cas_confirmes).label("moy_cas"),
                )
                .where(
                    and_(
                        MalariaCase.region_id == region_id,
                        MalariaCase.annee >= annee_debut_hist,
                    )
                )
                .group_by(MalariaCase.semaine_epidemio)
                .order_by(desc("moy_cas"))
                .limit(1)
            )
            result_hist = await self._db.execute(stmt_hist)
            row_hist    = result_hist.fetchone()
            semaine_pic = int(row_hist.semaine_epidemio) if row_hist else 10

            semaines_avant_pic = (semaine_pic - semaine) % 52

            tendance = (
                "hausse" if variation > 15
                else "baisse" if variation < -15
                else "stable"
            )

            return {
                "region_id":                    region_id,
                "saison_courante":              saison.value,
                "semaine_dans_saison":          semaine,
                "pic_historique_semaine":       semaine_pic,
                "semaines_avant_pic_estime":    semaines_avant_pic,
                "cas_cumules_saison":           cas_cumules,
                "cas_cumules_saison_precedente": cas_prec,
                "variation_pct":                variation,
                "tendance":                     tendance,
            }

        except Exception as exc:
            logger.error("MalariaRepo.get_seasonal_stats {} : {}", region_id, exc)
            return {"region_id": region_id, "tendance": "données insuffisantes"}

    async def get_national_stats(self) -> Dict[str, Any]:
        """
        KPIs nationaux paludisme.
        Appelé par GET /paludisme/statistiques/national.
        """
        try:
            today      = date.today()
            semaine    = today.isocalendar()[1]
            annee      = today.year
            date_4sem  = today - timedelta(weeks=4)

            # Cas 4 dernières semaines
            stmt_recent = (
                select(
                    func.sum(MalariaCase.cas_confirmes).label("total_cas"),
                    func.sum(MalariaCase.deces).label("total_deces"),
                    func.avg(MalariaCase.taux_positivite_tdr_pct).label("tdr_moyen"),
                    func.count(func.distinct(MalariaCase.region_id)).label("nb_regions"),
                )
                .where(MalariaCase.date_rapport >= date_4sem)
            )
            result_recent = await self._db.execute(stmt_recent)
            row_recent    = result_recent.fetchone()

            # Cas 4 semaines précédentes (pour comparaison)
            date_8sem = today - timedelta(weeks=8)
            stmt_prec = (
                select(func.sum(MalariaCase.cas_confirmes))
                .where(
                    and_(
                        MalariaCase.date_rapport >= date_8sem,
                        MalariaCase.date_rapport < date_4sem,
                    )
                )
            )
            result_prec = await self._db.execute(stmt_prec)
            cas_prec    = int(result_prec.scalar() or 0)

            cas_recent = int(row_recent.total_cas or 0)
            variation  = (
                round((cas_recent - cas_prec) / max(1, cas_prec) * 100, 1)
                if cas_prec > 0 else 0.0
            )

            # Alertes actives
            stmt_alertes = (
                select(func.count())
                .select_from(EpidemioAlert)
                .where(
                    and_(
                        EpidemioAlert.statut == "active",
                        EpidemioAlert.domaine == "paludisme",
                    )
                )
            )
            result_alertes = await self._db.execute(stmt_alertes)
            nb_alertes     = int(result_alertes.scalar() or 0)

            # Régions en urgence
            stmt_urgence = (
                select(func.count(func.distinct(EpidemioAlert.region_id)))
                .where(
                    and_(
                        EpidemioAlert.statut == "active",
                        EpidemioAlert.severite.in_(["urgence", "crise"]),
                        EpidemioAlert.domaine == "paludisme",
                    )
                )
            )
            result_urgence = await self._db.execute(stmt_urgence)
            regions_urgence = int(result_urgence.scalar() or 0)

            return {
                "semaine_reference": f"S{semaine:02d}-{annee}",
                "total_cas_4semaines": cas_recent,
                "total_deces_4semaines": int(row_recent.total_deces or 0),
                "taux_positivite_tdr_moyen_pct": round(float(row_recent.tdr_moyen or 0), 2),
                "variation_cas_vs_4sem_prec_pct": variation,
                "tendance": (
                    "hausse" if variation > 10
                    else "baisse" if variation < -10
                    else "stable"
                ),
                "alertes_actives": nb_alertes,
                "regions_en_urgence": regions_urgence,
                "nb_regions_avec_donnees": int(row_recent.nb_regions or 0),
                "date_calcul": today.isoformat(),
            }

        except Exception as exc:
            logger.error("MalariaRepo.get_national_stats : {}", exc)
            return {"erreur": str(exc)}

    # ─────────────────────────────────────────────
    # READ — Alertes
    # ─────────────────────────────────────────────

    async def get_alertes(
        self,
        region_id: Optional[str] = None,
        severite: Optional[str] = None,
        statut: str = "active",
    ) -> List[Dict[str, Any]]:
        """
        Retourne les alertes épidémiologiques paludisme.
        Appelé par GET /paludisme/alertes.
        """
        try:
            conditions = [EpidemioAlert.domaine == "paludisme"]

            if region_id:
                conditions.append(EpidemioAlert.region_id == region_id)
            if severite:
                conditions.append(EpidemioAlert.severite == severite)
            if statut and statut != "all":
                conditions.append(EpidemioAlert.statut == statut)

            stmt = (
                select(EpidemioAlert)
                .where(and_(*conditions))
                .order_by(
                    # Priorité : crise > urgence > alerte > surveillance
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
            rows   = result.scalars().all()

            return [self._alert_to_dict(r) for r in rows]

        except Exception as exc:
            logger.error("MalariaRepo.get_alertes : {}", exc)
            return []

    async def acquitter_alerte(
        self,
        alerte_id: str,
        user_id: int,
        commentaire: Optional[str] = None,
    ) -> bool:
        """
        Acquitte une alerte (statut → resolue).
        Appelé par POST /paludisme/alertes/{alerte_id}/acquitter.
        Retourne True si acquittée, False si introuvable.
        """
        try:
            stmt = (
                update(EpidemioAlert)
                .where(EpidemioAlert.alerte_id == alerte_id)
                .values(
                    statut="resolue",
                    acquittee_par=user_id,
                    acquittee_le=datetime.utcnow(),
                    commentaire_acquittement=commentaire,
                )
            )
            result = await self._db.execute(stmt)
            await self._db.flush()
            return result.rowcount > 0

        except Exception as exc:
            logger.error("MalariaRepo.acquitter_alerte {} : {}", alerte_id, exc)
            return False

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
        Retourne les données de backtesting (prédictions vs réel).
        Appelé par GET /predictions/backtest/{region_id}?modele=paludisme.
        """
        try:
            from src.database.models import MLPrediction

            # Prédictions historiques
            stmt_pred = (
                select(MLPrediction)
                .where(
                    and_(
                        MLPrediction.region_id == region_id,
                        MLPrediction.modele_nom == "malaria",
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
                return self._empty_backtest(region_id, date_debut, date_fin, "paludisme")

            pred_values = [float(p.score_paludisme or 0) for p in predictions]
            real_values = [float(p.valeur_reelle or 0) for p in predictions]

            import numpy as np
            errors   = [abs(p - r) for p, r in zip(pred_values, real_values)]
            sq_errors = [(p - r)**2 for p, r in zip(pred_values, real_values)]
            pct_errors = [
                abs(p - r) / max(r, 0.01) * 100
                for p, r in zip(pred_values, real_values)
            ]

            mae     = round(float(np.mean(errors)), 4)
            rmse    = round(float(np.sqrt(np.mean(sq_errors))), 4)
            mape    = round(float(np.mean(pct_errors)), 2)
            biais   = round(float(np.mean([p - r for p, r in zip(pred_values, real_values)])), 4)
            corr    = round(float(np.corrcoef(pred_values, real_values)[0, 1]), 4) \
                if len(pred_values) > 2 else 0.0

            pvr = [
                {
                    "date": p.date_prediction.isoformat(),
                    "score_predit": float(p.score_paludisme or 0),
                    "valeur_reelle": float(p.valeur_reelle or 0),
                    "erreur_absolue": round(abs(float(p.score_paludisme or 0) - float(p.valeur_reelle or 0)), 4),
                }
                for p in predictions
            ]

            return {
                "region_id":    region_id,
                "periode_debut": str(date_debut),
                "periode_fin":   str(date_fin),
                "modele":       "paludisme",
                "mae":          mae,
                "rmse":         rmse,
                "mape_pct":     mape,
                "correlation":  corr,
                "biais":        biais,
                "nb_predictions": len(predictions),
                "predictions_vs_reel": pvr[:100],  # Max 100 points
            }

        except Exception as exc:
            logger.error("MalariaRepo.get_backtest_data {} : {}", region_id, exc)
            return self._empty_backtest(region_id, date_debut, date_fin, "paludisme")

    # ─────────────────────────────────────────────
    # WRITE
    # ─────────────────────────────────────────────

    async def save_case(self, data: Dict[str, Any]) -> MalariaCase:
        """
        Upsert un cas de paludisme hebdomadaire.
        Appelé par le scheduler Celery (version async).
        """
        try:
            stmt = select(MalariaCase).where(
                and_(
                    MalariaCase.region_id == data["region_id"],
                    MalariaCase.annee == data["annee"],
                    MalariaCase.semaine_epidemio == data["semaine_epidemio"],
                )
            )
            result   = await self._db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                for field in [
                    "cas_confirmes", "cas_confirmes_mixte", "deces", "hospitalisations",
                    "taux_incidence_pour_mille", "taux_positivite_tdr_pct",
                    "tests_malaria", "tdr_positifs", "fiabilite_donnees",
                ]:
                    val = data.get(field)
                    if val is not None:
                        setattr(existing, field, val)
                case = existing
            else:
                date_rapport = data.get("date_rapport")
                if isinstance(date_rapport, str):
                    date_rapport = date.fromisoformat(date_rapport[:10])

                case = MalariaCase(
                    region_id=data["region_id"],
                    district=data.get("district"),
                    annee=data["annee"],
                    semaine_epidemio=data["semaine_epidemio"],
                    date_rapport=date_rapport,
                    cas_confirmes=data.get("cas_confirmes", 0),
                    cas_confirmes_mixte=data.get("cas_confirmes_mixte", 0),
                    deces=data.get("deces", 0),
                    hospitalisations=data.get("hospitalisations", 0),
                    tests_malaria=data.get("tests_malaria", 0),
                    tdr_positifs=data.get("tdr_positifs", 0),
                    taux_incidence_pour_mille=data.get("taux_incidence_pour_mille", 0),
                    taux_positivite_tdr_pct=data.get("taux_positivite_tdr_pct", 0),
                    population_a_risque=data.get("population_a_risque"),
                    source=data.get("source", "DHIS2"),
                    fiabilite_donnees=data.get("fiabilite_donnees", "confirmée"),
                    period_dhis2=data.get("period_dhis2"),
                )
                self._db.add(case)

            await self._db.flush()
            return case

        except Exception as exc:
            logger.error("MalariaRepo.save_case : {}", exc)
            raise

    async def save_cases_batch(self, data_list: List[Dict[str, Any]]) -> int:
        """Upsert batch de cas paludisme."""
        count = 0
        for data in data_list:
            try:
                await self.save_case(data)
                count += 1
            except Exception as exc:
                logger.warning("Batch malaria skip : {}", exc)
        await self._db.flush()
        return count

    async def save_alert(self, data: Dict[str, Any]) -> EpidemioAlert:
        """Insère une alerte épidémiologique (idempotent sur alerte_id)."""
        try:
            alerte_id = data.get("alerte_id", str(uuid.uuid4()))
            stmt = select(EpidemioAlert).where(EpidemioAlert.alerte_id == alerte_id)
            result   = await self._db.execute(stmt)
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
                domaine="paludisme",
                seuil_depasse=data.get("seuil_depasse"),
                valeur_actuelle=data.get("valeur_actuelle"),
                date_detection=date_detection or datetime.utcnow(),
                statut=data.get("statut", "active"),
                description=data.get("description"),
                actions_requises=data.get("actions_requises"),
                responsable_notification=data.get("responsable_notification"),
            )
            self._db.add(alert)
            await self._db.flush()
            return alert

        except Exception as exc:
            logger.error("MalariaRepo.save_alert : {}", exc)
            raise

    async def save_alerts_batch(self, alerts: List[Dict[str, Any]]) -> int:
        """Insère un batch d'alertes."""
        count = 0
        for a in alerts:
            try:
                await self.save_alert(a)
                count += 1
            except Exception:
                pass
        await self._db.flush()
        return count

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _case_to_dict(case: MalariaCase) -> Dict[str, Any]:
        return {
            "region_id":                 case.region_id,
            "region_name":               case.region_id,  # enrichi via metadata
            "district":                  case.district,
            "semaine_epidemio":          case.semaine_epidemio,
            "annee":                     case.annee,
            "date_rapport":              str(case.date_rapport) if case.date_rapport else None,
            "cas_confirmes":             case.cas_confirmes or 0,
            "cas_confirmes_mixte":              case.cas_confirmes_mixte or 0,
            "deces":                     case.deces or 0,
            "hospitalisations":          case.hospitalisations or 0,
            "taux_incidence_pour_mille": float(case.taux_incidence_pour_mille or 0),
            "taux_positivite_tdr_pct":   float(case.taux_positivite_tdr_pct or 0),
            "population_a_risque":       case.population_a_risque or 0,
            "source":                    case.source,
            "fiabilite_donnees":         case.fiabilite_donnees,
        }

    @staticmethod
    def _alert_to_dict(alert: EpidemioAlert) -> Dict[str, Any]:
        return {
            "alerte_id":             alert.alerte_id,
            "region_id":             alert.region_id,
            "region_name":           alert.region_name or alert.region_id,
            "type_alerte":           alert.type_alerte,
            "severite":              alert.severite,
            "seuil_depasse":         float(alert.seuil_depasse) if alert.seuil_depasse else None,
            "valeur_actuelle":       float(alert.valeur_actuelle) if alert.valeur_actuelle else None,
            "date_detection":        alert.date_detection.isoformat() if alert.date_detection else None,
            "statut":                alert.statut,
            "description":           alert.description,
            "actions_requises":      alert.actions_requises or [],
            "responsable_notification": alert.responsable_notification,
        }

    @staticmethod
    def _empty_backtest(
        region_id: str, date_debut: date, date_fin: date, modele: str
    ) -> Dict[str, Any]:
        return {
            "region_id": region_id, "periode_debut": str(date_debut),
            "periode_fin": str(date_fin), "modele": modele,
            "mae": 0.0, "rmse": 0.0, "mape_pct": 0.0,
            "correlation": 0.0, "biais": 0.0,
            "nb_predictions": 0, "predictions_vs_reel": [],
        }