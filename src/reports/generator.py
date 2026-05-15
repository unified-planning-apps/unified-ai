"""
Orchestrateur de génération de rapports UNICEF Madagascar.

Interface publique (contrat avec reports.py router et scheduler.py) :

    generator = ReportGenerator()
    chemin = await generator.generate(
        rapport_id, type_rapport, format_rapport, langue,
        region_id, date_debut, date_fin, options
    )
    # → Path vers le fichier PDF/HTML généré

Types de rapports :
  paludisme_hebdomadaire  → malaria_weekly.html → PDF
  nutrition_hebdomadaire  → nutrition_weekly.html → PDF
  combine_hebdomadaire    → Les deux combinés
  urgence                 → Rapport d'urgence simplifié prioritaire
  mensuel                 → Rapport mensuel détaillé

Pipeline de génération :
  1. Collecte des données (DB + API)
  2. Calcul prédictions ML
  3. Génération visualisations (cartes, graphiques)
  4. Rendu template Jinja2 → HTML
  5. Conversion HTML → PDF (ReportLab / WeasyPrint)
  6. Stockage fichier + upload MinIO (optionnel)
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger

from config.settings import settings


class ReportGenerator:
    """
    Générateur de rapports UNICEF — orchestrateur principal.
    Produit des rapports PDF ou HTML selon le type demandé.
    """

    TEMPLATES_DIR = Path(__file__).parent / "templates"
    OUTPUT_DIR    = settings.report_output_dir

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(self.TEMPLATES_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        # Filtres Jinja2 personnalisés
        self._jinja_env.filters["date_fr"]    = self._filtre_date_fr
        self._jinja_env.filters["niveau_css"] = self._filtre_niveau_css
        self._jinja_env.filters["pourcentage"]= lambda v: f"{v:.1f}%" if v is not None else "N/A"
        self._jinja_env.filters["nombre_fr"]  = lambda v: f"{v:,.0f}".replace(",", " ") if v else "0"

    # ─────────────────────────────────────────────
    # Interface publique — contrat avec router + scheduler
    # ─────────────────────────────────────────────

    async def generate(
        self,
        rapport_id: str,
        type_rapport: str,
        format_rapport: str = "pdf",
        langue: str = "fr",
        region_id: Optional[str] = None,
        date_debut: Optional[date] = None,
        date_fin: Optional[date] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Génère un rapport et retourne le chemin du fichier créé.

        Args:
            rapport_id    : ID unique du rapport (pour nommage fichier)
            type_rapport  : paludisme_hebdomadaire | nutrition_hebdomadaire |
                            combine_hebdomadaire | urgence | mensuel
            format_rapport: pdf | html | json
            langue        : fr | mg
            region_id     : ID région ou None pour rapport national
            date_debut    : Début de la période analysée
            date_fin      : Fin de la période analysée
            options       : Dict avec flags inclure_cartes, inclure_shap, etc.

        Returns:
            Path vers le fichier généré.
        """
        options    = options or {}
        date_fin   = date_fin or date.today()
        date_debut = date_debut or date_fin - timedelta(days=7)

        logger.info(
            "Génération rapport — id={} type={} format={} region={}",
            rapport_id, type_rapport, format_rapport, region_id or "national"
        )

        # 1. Routage vers le bon type de rapport
        if type_rapport in ("paludisme_hebdomadaire", "alerte_epidemique"):
            context = await self._build_malaria_context(
                region_id, date_debut, date_fin, options
            )
            template_name = "malaria_weekly.html"

        elif type_rapport == "nutrition_hebdomadaire":
            context = await self._build_nutrition_context(
                region_id, date_debut, date_fin, options
            )
            template_name = "nutrition_weekly.html"

        elif type_rapport in ("combine_hebdomadaire", "mensuel"):
            context_mal = await self._build_malaria_context(
                region_id, date_debut, date_fin, options
            )
            context_nut = await self._build_nutrition_context(
                region_id, date_debut, date_fin, options
            )
            context = {**context_mal, **context_nut, "combine": True}
            template_name = "malaria_weekly.html"  # template combine intégré

        elif type_rapport == "urgence":
            context = await self._build_urgence_context(
                region_id, date_debut, date_fin, options
            )
            template_name = "malaria_weekly.html"

        else:
            raise ValueError(f"Type de rapport inconnu : {type_rapport}")

        # Enrichissement contexte commun
        context.update(
            self._build_common_context(
                rapport_id, type_rapport, langue, region_id, date_debut, date_fin
            )
        )

        # 2. Génération visualisations
        if options.get("inclure_cartes", True):
            context["carte_risque_url"] = await self._generer_carte(
                region_id=region_id,
                type_carte="paludisme" if "paludisme" in type_rapport else "combine",
                rapport_id=rapport_id,
            )

        # 3. Rendu HTML via Jinja2
        html_content = self._render_template(template_name, context)

        # 4. Sauvegarde ou conversion
        output_path = self._get_output_path(rapport_id, format_rapport)

        if format_rapport == "html":
            output_path.write_text(html_content, encoding="utf-8")
        elif format_rapport == "pdf":
            output_path = await self._html_to_pdf(html_content, output_path)
        elif format_rapport == "json":
            import json
            output_path.write_text(
                json.dumps(context, default=str, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        logger.info("✅ Rapport généré → {}", output_path)
        return output_path

    # ─────────────────────────────────────────────
    # Contextes de données pour les templates
    # ─────────────────────────────────────────────

    async def _build_malaria_context(
        self,
        region_id: Optional[str],
        date_debut: date,
        date_fin: date,
        options: Dict,
    ) -> Dict[str, Any]:
        """Collecte et structure les données paludisme pour le template."""
        context: Dict[str, Any] = {
            "section": "paludisme",
            "alertes_malaria": [],
            "cas_semaine": {},
            "tendance_data": [],
            "prediction_malaria": {},
            "facteurs_risque": {},
            "shap_data": None,
        }

        try:
            # Données épidémiologiques
            if region_id:
                from src.data_collection.malaria_fetcher import MalariaFetcher
                fetcher = MalariaFetcher()
                records = await fetcher.get_cas_dhis2(region_id, date_debut, date_fin)
                await fetcher.close()

                if records:
                    last = records[-1]
                    context["cas_semaine"] = {
                        "cas_confirmes":    last.get("cas_confirmes", 0),
                        "cas_confirmes_mixte":     last.get("cas_confirmes_mixte", 0),
                        "deces":            last.get("deces", 0),
                        "hospitalisations": last.get("hospitalisations", 0),
                        "taux_positivite_tdr_pct": last.get("taux_positivite_tdr_pct", 0),
                        "taux_incidence_pour_mille": last.get("taux_incidence_pour_mille", 0),
                        "semaine_epidemio": last.get("semaine_epidemio", 0),
                        "annee":            last.get("annee", date.today().year),
                    }

                    # Tendance (lissée)
                    from src.preprocessing.health_processor import HealthProcessor
                    processor = HealthProcessor()
                    cas_list  = [r.get("cas_confirmes", 0) for r in records]
                    cas_lisse = processor.smooth_time_series(cas_list, window=3)
                    context["tendance_data"] = [
                        {**r, "cas_lisse": round(v, 1)}
                        for r, v in zip(records, cas_lisse)
                    ]

                    # Alertes
                    alertes = fetcher.calculer_alertes(records, region_id)
                    context["alertes_malaria"] = [
                        a for a in alertes if a.get("severite") in ("urgence", "crise")
                    ]

            # Prédiction ML (si modèle disponible)
            try:
                from src.models.malaria_predictor import MalariaPredictor
                from src.preprocessing.feature_engineering import FeatureEngineer

                model = MalariaPredictor.load_latest()
                if model and region_id:
                    engineer = FeatureEngineer()
                    features = await engineer.build_malaria_features(region_id)
                    pred = model.predict(features, horizon_days=14)
                    context["prediction_malaria"] = pred
                    context["facteurs_risque"] = features

                    # SHAP si demandé
                    if options.get("inclure_shap") and model:
                        from src.models.explainability import SHAPExplainer
                        explainer = SHAPExplainer(model)
                        context["shap_data"] = explainer.explain(
                            features, region_id=region_id, generate_plots=True
                        )
            except Exception as exc:
                logger.debug("Prédiction ML rapport : {}", exc)

        except Exception as exc:
            logger.warning("Contexte malaria rapport : {}", exc)

        return context

    async def _build_nutrition_context(
        self,
        region_id: Optional[str],
        date_debut: date,
        date_fin: date,
        options: Dict,
    ) -> Dict[str, Any]:
        """Collecte et structure les données nutrition pour le template."""
        context: Dict[str, Any] = {
            "section": "nutrition",
            "statut_nutritionnel": {},
            "disponibilite": {},
            "stocks": {},
            "recettes": [],
            "alertes_nutrition": [],
            "soudure": {},
            "prediction_nutrition": {},
        }

        try:
            from src.data_collection.nutrition_fetcher import NutritionFetcher
            fetcher = NutritionFetcher()

            if region_id:
                # Statut nutritionnel
                statut = await fetcher.get_statut_nutritionnel(region_id)
                context["statut_nutritionnel"] = statut

                # Disponibilité alimentaire
                dispo = await fetcher.get_disponibilite_complete(region_id)
                context["disponibilite"] = dispo

                # Période de soudure
                soudure = await fetcher.get_statut_soudure(region_id)
                context["soudure"] = soudure[0] if soudure else {}

            await fetcher.close()

            # Recettes si demandé
            if options.get("inclure_recettes", True) and region_id:
                from src.reports.recipe_selector import RecipeSelector
                selector = RecipeSelector()
                context["recettes"] = await selector.generer_recettes_optimales(
                    region_id=region_id,
                    cible="enfants_6_23m",
                    nombre=3,
                )

            # Stocks si demandé
            if options.get("inclure_stocks", False) and region_id:
                from src.data_collection.nutrition_fetcher import NutritionFetcher as NF
                try:
                    nf = NF()
                    # Placeholder — stocks depuis DB via repo
                    context["stocks"] = {}
                except Exception:
                    pass

            # Prédiction nutrition ML
            try:
                from src.models.nutrition_predictor import NutritionPredictor
                from src.preprocessing.feature_engineering import FeatureEngineer

                model = NutritionPredictor.load_latest()
                if model and region_id:
                    engineer = FeatureEngineer()
                    features = await engineer.build_nutrition_features(region_id)
                    pred = model.predict(features, horizon_days=30)
                    context["prediction_nutrition"] = pred
            except Exception as exc:
                logger.debug("Prédiction nutrition rapport : {}", exc)

        except Exception as exc:
            logger.warning("Contexte nutrition rapport : {}", exc)

        return context

    async def _build_urgence_context(
        self,
        region_id: Optional[str],
        date_debut: date,
        date_fin: date,
        options: Dict,
    ) -> Dict[str, Any]:
        """Contexte simplifié pour rapport d'urgence (génération rapide)."""
        context = await self._build_malaria_context(
            region_id, date_debut, date_fin,
            {**options, "inclure_shap": False}
        )
        context_nut = await self._build_nutrition_context(
            region_id, date_debut, date_fin,
            {**options, "inclure_recettes": False}
        )
        context.update(context_nut)
        context["urgence"]          = True
        context["type_crise"]       = options.get("type_crise", "non spécifié")
        context["description_crise"]= options.get("description_crise", "")
        return context

    def _build_common_context(
        self,
        rapport_id: str,
        type_rapport: str,
        langue: str,
        region_id: Optional[str],
        date_debut: date,
        date_fin: date,
    ) -> Dict[str, Any]:
        """Contexte commun à tous les rapports."""
        import json
        from pathlib import Path as P

        # Métadonnées région
        region_meta = {}
        if region_id:
            try:
                with P("config/regions_metadata.json").open() as f:
                    meta = json.load(f)
                region_meta = next(
                    (r for r in meta["regions"] if r["id"] == region_id), {}
                )
            except Exception:
                pass

        return {
            "rapport_id":     rapport_id,
            "type_rapport":   type_rapport,
            "langue":         langue,
            "region_id":      region_id,
            "region_name":    region_meta.get("name", "National"),
            "region_chef_lieu": region_meta.get("chef_lieu", ""),
            "date_debut":     date_debut,
            "date_fin":       date_fin,
            "date_generation":datetime.utcnow(),
            "semaine":        f"S{date_fin.isocalendar()[1]:02d}-{date_fin.year}",
            "organisation":   "UNICEF Madagascar",
            "logo_url":       "/static/unicef_logo.png",
            "est_national":   region_id is None,
            "seuils_oms": {
                "gam_acceptable": 5.0,
                "gam_alerte":    10.0,
                "gam_urgence":   15.0,
            },
        }

    # ─────────────────────────────────────────────
    # Rendu template et conversion
    # ─────────────────────────────────────────────

    def _render_template(self, template_name: str, context: Dict) -> str:
        """Rend un template Jinja2 avec le contexte fourni."""
        try:
            template = self._jinja_env.get_template(template_name)
            return template.render(**context)
        except Exception as exc:
            logger.error("Erreur rendu template {} : {}", template_name, exc)
            # Fallback HTML minimal
            return self._render_fallback_html(context)

    async def _html_to_pdf(self, html_content: str, output_path: Path) -> Path:
        """Convertit du HTML en PDF via WeasyPrint (prioritaire) ou ReportLab."""
        pdf_path = output_path.with_suffix(".pdf")

        # Tentative WeasyPrint
        try:
            import weasyprint
            weasyprint.HTML(string=html_content, base_url=str(self.TEMPLATES_DIR)).write_pdf(
                str(pdf_path),
                stylesheets=[weasyprint.CSS(string=self._get_print_css())],
            )
            logger.debug("PDF généré via WeasyPrint → {}", pdf_path)
            return pdf_path
        except ImportError:
            logger.debug("WeasyPrint non disponible — fallback ReportLab")
        except Exception as exc:
            logger.warning("WeasyPrint échoué : {} — fallback ReportLab", exc)

        # Fallback ReportLab
        try:
            return await self._reportlab_pdf(html_content, pdf_path)
        except Exception as exc:
            logger.warning("ReportLab échoué : {} — sauvegarde HTML uniquement", exc)
            html_path = output_path.with_suffix(".html")
            html_path.write_text(html_content, encoding="utf-8")
            return html_path

    async def _reportlab_pdf(self, html_content: str, pdf_path: Path) -> Path:
        """Génère un PDF minimaliste via ReportLab."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        from reportlab.lib.units import cm
        import re

        # Extraction texte brut depuis HTML
        clean_text = re.sub(r"<[^>]+>", " ", html_content)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        doc    = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story  = []

        # Titre
        story.append(Paragraph("UNICEF Madagascar — Rapport Épidémiologique", styles["Title"]))
        story.append(Spacer(1, 0.5 * cm))

        # Contenu par paragraphes
        for para in clean_text[:5000].split(". "):
            if para.strip():
                story.append(Paragraph(para.strip() + ".", styles["Normal"]))
                story.append(Spacer(1, 0.2 * cm))

        doc.build(story)
        logger.debug("PDF généré via ReportLab → {}", pdf_path)
        return pdf_path

    async def _generer_carte(
        self,
        region_id: Optional[str],
        type_carte: str,
        rapport_id: str,
    ) -> Optional[str]:
        """Génère une carte choroplèthe et retourne son URL relative."""
        try:
            from src.reports.visualizations import Visualizer
            viz = Visualizer()
            carte_path = viz.generate_risk_map(
                region_id=region_id,
                type_carte=type_carte,
                output_dir=self.OUTPUT_DIR / "maps",
                rapport_id=rapport_id,
            )
            return str(carte_path) if carte_path else None
        except Exception as exc:
            logger.debug("Génération carte : {}", exc)
            return None

    def _get_output_path(self, rapport_id: str, format_rapport: str) -> Path:
        """Construit le chemin de sortie du fichier rapport."""
        ext = {"pdf": ".pdf", "html": ".html", "json": ".json"}.get(format_rapport, ".html")
        return self.OUTPUT_DIR / f"rapport_{rapport_id}{ext}"

    def _render_fallback_html(self, context: Dict) -> str:
        """HTML minimal de fallback si le template Jinja2 échoue."""
        return f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><title>Rapport UNICEF {context.get('rapport_id','')}</title></head>
<body>
<h1>UNICEF Madagascar — Rapport Épidémiologique</h1>
<p>Région : {context.get('region_name', 'Nationale')}</p>
<p>Période : {context.get('date_debut')} — {context.get('date_fin')}</p>
<p>Généré le : {context.get('date_generation', datetime.utcnow())}</p>
<p><em>Le rapport complet sera disponible dès la prochaine génération.</em></p>
</body>
</html>"""

    @staticmethod
    def _get_print_css() -> str:
        """CSS pour l'impression PDF."""
        return """
        @page { size: A4; margin: 2cm; }
        body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10pt; color: #222; }
        h1 { color: #00AEEF; font-size: 18pt; }
        h2 { color: #374EA2; font-size: 13pt; border-bottom: 1px solid #ccc; }
        table { width: 100%; border-collapse: collapse; margin: 1em 0; }
        th { background: #00AEEF; color: white; padding: 6px; }
        td { padding: 5px; border-bottom: 1px solid #eee; }
        .rouge { color: #D32F2F; font-weight: bold; }
        .orange { color: #F57C00; }
        .jaune { color: #F9A825; }
        .vert { color: #388E3C; }
        .alert-box { background: #FFF3E0; border-left: 4px solid #F57C00; padding: 10px; margin: 10px 0; }
        .crisis-box { background: #FFEBEE; border-left: 4px solid #D32F2F; padding: 10px; margin: 10px 0; }
        """

    # ─────────────────────────────────────────────
    # Filtres Jinja2
    # ─────────────────────────────────────────────

    @staticmethod
    def _filtre_date_fr(d) -> str:
        """Formate une date en français."""
        if d is None:
            return "N/A"
        mois_fr = [
            "janvier", "février", "mars", "avril", "mai", "juin",
            "juillet", "août", "septembre", "octobre", "novembre", "décembre"
        ]
        if isinstance(d, (date, datetime)):
            return f"{d.day} {mois_fr[d.month - 1]} {d.year}"
        return str(d)

    @staticmethod
    def _filtre_niveau_css(niveau: str) -> str:
        """Retourne la classe CSS correspondant au niveau de risque."""
        mapping = {
            "faible":     "vert",
            "moyen":      "jaune",
            "élevé":      "orange",
            "très élevé": "rouge",
            "acceptable": "vert",
            "alerte":     "jaune",
            "urgence":    "orange",
            "crise":      "rouge",
            "vert":       "vert",
            "jaune":      "jaune",
            "orange":     "orange",
            "rouge":      "rouge",
        }
        return mapping.get(niveau, "")