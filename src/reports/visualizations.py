"""
Génération de visualisations pour les rapports UNICEF Madagascar.

Couvre :
  - Cartes choroplèthes (Folium → PNG statique via Selenium ou export direct)
  - Graphiques épidémiologiques (Matplotlib/Seaborn)
  - Graphiques nutritionnels (GAM trend, FCS, stocks)
  - Graphiques SHAP pour explicabilité

Interface publique (contrat avec generator.py) :
  Visualizer().generate_risk_map(region_id, type_carte, output_dir, rapport_id)
    → Optional[Path]

Toutes les méthodes retournent des Path vers des fichiers PNG/SVG
sauvegardés dans output_dir, prêts pour inclusion dans les rapports PDF.
"""

from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Backend non-interactif — obligatoire en mode serveur
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────────────
# Palette couleurs UNICEF (charte graphique officielle)
# ─────────────────────────────────────────────────────────────────
UNICEF_BLUE    = "#00AEEF"
UNICEF_DARK    = "#374EA2"
UNICEF_YELLOW  = "#FFCC00"
UNICEF_GREEN   = "#80BD41"
UNICEF_ORANGE  = "#F26A21"
UNICEF_RED     = "#E2231A"

RISK_COLORS = {
    "vert":   "#4CAF50",
    "jaune":  "#FFC107",
    "orange": "#FF9800",
    "rouge":  "#F44336",
}

RISK_SCORE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "unicef_risk",
    ["#4CAF50", "#FFC107", "#FF9800", "#F44336"],
    N=256,
)


class Visualizer:
    """
    Générateur de visualisations statiques pour rapports UNICEF.
    Toutes les sorties sont des fichiers PNG/SVG sur disque.
    """

    DEFAULT_FIGSIZE  = (10, 6)
    DEFAULT_DPI      = 150
    FONT_FAMILY      = "DejaVu Sans"

    def __init__(self):
        # Style global Matplotlib
        plt.rcParams.update({
            "font.family":      self.FONT_FAMILY,
            "font.size":        10,
            "axes.titlesize":   12,
            "axes.labelsize":   10,
            "xtick.labelsize":  9,
            "ytick.labelsize":  9,
            "axes.grid":        True,
            "grid.alpha":       0.3,
            "figure.dpi":       self.DEFAULT_DPI,
            "savefig.dpi":      self.DEFAULT_DPI,
            "savefig.bbox_inches": "tight",
            "savefig.transparent": False,
        })

    # ─────────────────────────────────────────────
    # Carte de risque — interface avec generator.py
    # ─────────────────────────────────────────────

    def generate_risk_map(
        self,
        region_id: Optional[str],
        type_carte: str,
        output_dir: Path,
        rapport_id: str,
    ) -> Optional[Path]:
        """
        Génère une carte choroplèthe du risque pour les 22 régions.
        Appelé par generator.py._generer_carte().

        Args:
            region_id   : si fourni → met en surbrillance cette région
            type_carte  : "paludisme" | "nutrition" | "combine"
            output_dir  : répertoire de sortie
            rapport_id  : identifiant du rapport (pour nommage)

        Returns:
            Path vers le fichier PNG généré, ou None si échec.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"carte_{type_carte}_{rapport_id}.png"

        try:
            # Tentative Folium (carte interactive → PNG)
            result = self._generate_folium_map(
                region_id=region_id,
                type_carte=type_carte,
                output_path=output_path,
            )
            if result:
                return result
        except Exception as exc:
            logger.debug("Folium carte échoué : {} — fallback matplotlib", exc)

        # Fallback : carte schématique Matplotlib
        try:
            return self._generate_schematic_map(
                region_id=region_id,
                type_carte=type_carte,
                output_path=output_path,
            )
        except Exception as exc:
            logger.error("Génération carte complètement échouée : {}", exc)
            return None

    def _generate_folium_map(
        self,
        region_id: Optional[str],
        type_carte: str,
        output_path: Path,
    ) -> Optional[Path]:
        """Carte choroplèthe via Folium + données de risque synthétiques."""
        import folium
        import json
        from pathlib import Path as P

        # Données de risque (depuis cache Redis si dispo, sinon synthétiques)
        risk_data = self._get_risk_data_for_map(type_carte)

        # Centre Madagascar
        m = folium.Map(
            location=[-20.0, 47.0],
            zoom_start=6,
            tiles="CartoDB positron",
            prefer_canvas=True,
        )

        # Ajout titre
        title_html = f"""
        <div style="position:fixed;top:10px;left:50px;z-index:1000;
                    background:white;padding:10px;border-radius:5px;
                    border:2px solid {UNICEF_BLUE};font-family:Arial;">
            <b style="color:{UNICEF_BLUE}">UNICEF Madagascar</b><br>
            Carte de risque — {type_carte.upper()}<br>
            <small>{datetime.utcnow().strftime('%d/%m/%Y')}</small>
        </div>"""
        m.get_root().html.add_child(folium.Element(title_html))

        # Chargement shapefiles des régions (si disponibles)
        shapefile_path = P("data/external/madagascar_regions.geojson")
        if shapefile_path.exists():
            with shapefile_path.open() as f:
                geojson = json.load(f)

            folium.Choropleth(
                geo_data=geojson,
                data=risk_data,
                columns=["region_id", "score_risque"],
                key_on="feature.properties.region_id",
                fill_color="RdYlGn_r",
                fill_opacity=0.75,
                line_opacity=0.5,
                legend_name=f"Score de risque {type_carte}",
                name="Risque",
                nan_fill_color="lightgray",
            ).add_to(m)
        else:
            # Sans GeoJSON → marqueurs ponctuels
            self._add_region_markers(m, risk_data, region_id)

        # Légende UNICEF
        self._add_folium_legend(m, type_carte)

        # Sauvegarde HTML puis conversion PNG via screenshot
        html_path = output_path.with_suffix(".html")
        m.save(str(html_path))

        # Tentative screenshot headless (Selenium/Playwright)
        png_path = self._screenshot_folium(html_path, output_path)
        return png_path if png_path else None

    def _generate_schematic_map(
        self,
        region_id: Optional[str],
        type_carte: str,
        output_path: Path,
    ) -> Path:
        """
        Carte schématique Matplotlib (fallback sans GeoJSON/Selenium).
        Représente les 22 régions comme des rectangles colorés par risque.
        """
        risk_data = self._get_risk_data_for_map(type_carte)
        risk_by_id = {r["region_id"]: r for r in risk_data}

        # Disposition approximative des 22 régions Madagascar (coordonnées simplifiées)
        REGIONS_POS = [
            ("MDG-DIA",  0.85, 0.92, "Diana"),
            ("MDG-SAV",  0.90, 0.78, "Sava"),
            ("MDG-SOF",  0.60, 0.80, "Sofia"),
            ("MDG-BOE",  0.30, 0.72, "Boeny"),
            ("MDG-MEN",  0.15, 0.60, "Melaky"),
            ("MDG-ANA2", 0.90, 0.65, "Analanjirofo"),
            ("MDG-ALA",  0.75, 0.62, "Alaotra-Mangoro"),
            ("MDG-BMT",  0.35, 0.55, "Bongolava"),
            ("MDG-ANA",  0.60, 0.50, "Analamanga"),
            ("MDG-ITM",  0.45, 0.52, "Itasy"),
            ("MDG-VAK",  0.58, 0.43, "Vakinankaratra"),
            ("MDG-ATS",  0.92, 0.52, "Atsinanana"),
            ("MDG-MEN2", 0.20, 0.45, "Menabe"),
            ("MDG-ATI",  0.65, 0.38, "Amoron'i Mania"),
            ("MDG-MAT",  0.62, 0.32, "Matsiatra Ambony"),
            ("MDG-VAT",  0.82, 0.32, "Vatovavy"),
            ("MDG-FIT",  0.88, 0.42, "Fitovinany"),
            ("MDG-ANO",  0.78, 0.22, "Atsimo-Atsinanana"),
            ("MDG-IHO",  0.52, 0.22, "Ihorombe"),
            ("MDG-ASO",  0.22, 0.18, "Atsimo-Andrefana"),
            ("MDG_AND",  0.62, 0.10, "Androy"),
            ("MDG-AAN",  0.82, 0.10, "Anosy"),
        ]

        fig, ax = plt.subplots(figsize=(8, 12))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor("#E8F4FD")
        ax.axis("off")

        # Fond Madagascar approximatif (ellipse)
        ellipse = mpatches.Ellipse(
            (0.55, 0.50), 0.90, 0.98,
            facecolor="#D4E8D4", edgecolor="#888", linewidth=2, alpha=0.4
        )
        ax.add_patch(ellipse)

        # Dessin des régions
        for rid, x, y, name in REGIONS_POS:
            d = risk_by_id.get(rid, {})
            score = d.get("score_risque", 0.3)
            color = RISK_SCORE_CMAP(score)
            edge  = "#D32F2F" if rid == region_id else "#555"
            lw    = 3 if rid == region_id else 1

            # Bulle de région
            circle = mpatches.Circle(
                (x, y), 0.04,
                facecolor=color, edgecolor=edge,
                linewidth=lw, zorder=3
            )
            ax.add_patch(circle)

            # Label
            ax.text(
                x, y - 0.06, name,
                ha="center", va="top",
                fontsize=6.5,
                fontweight="bold" if rid == region_id else "normal",
                color="#222",
                zorder=4,
            )

            # Score
            ax.text(
                x, y, f"{score:.2f}",
                ha="center", va="center",
                fontsize=6, color="white" if score > 0.4 else "#333",
                fontweight="bold", zorder=5,
            )

        # Colorbar légende
        sm = plt.cm.ScalarMappable(
            cmap=RISK_SCORE_CMAP,
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.4, aspect=15, pad=0.02)
        cbar.set_label("Score de risque", fontsize=9)
        cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
        cbar.set_ticklabels(["Faible", "", "Moyen", "", "Très élevé"], fontsize=7)

        # Titre
        titre = {
            "paludisme":  "Carte du Risque Paludisme",
            "nutrition":  "Carte du Risque Malnutrition",
            "combine":    "Carte du Risque Combiné (Paludisme + Nutrition)",
        }.get(type_carte, "Carte de Risque")

        ax.set_title(
            f"UNICEF Madagascar\n{titre}\n{date.today().strftime('%d/%m/%Y')}",
            fontsize=11, fontweight="bold", color=UNICEF_DARK,
            pad=12
        )

        # Logo UNICEF simulé (barre bleue en haut)
        ax.axhline(y=0.985, color=UNICEF_BLUE, linewidth=6, zorder=6)
        ax.text(
            0.5, 0.992, "UNICEF Madagascar",
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=7,
        )

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=self.DEFAULT_DPI, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

        logger.debug("Carte schématique {} générée → {}", type_carte, output_path)
        return output_path

    # ─────────────────────────────────────────────
    # Graphiques épidémiologiques
    # ─────────────────────────────────────────────

    def plot_malaria_trend(
        self,
        tendance_data: List[Dict],
        region_name: str,
        output_path: Path,
    ) -> Path:
        """
        Graphique courbe temporelle des cas de paludisme avec moyenne mobile.
        Utilisé dans les templates HTML via balise <img>.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not tendance_data:
            return self._placeholder_chart(output_path, "Données épidémiologiques non disponibles")

        semaines = [f"S{d.get('semaine_epidemio', i+1):02d}" for i, d in enumerate(tendance_data)]
        cas      = [d.get("cas_confirmes", 0) for d in tendance_data]
        cas_lisse= [d.get("cas_lisse", c) for d, c in zip(tendance_data, cas)]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[3, 1])

        # ── Axe 1 : Courbe cas + moyenne mobile ──
        ax1 = axes[0]
        x   = range(len(semaines))

        ax1.bar(x, cas, color=UNICEF_BLUE, alpha=0.5, label="Cas confirmés", width=0.7)
        ax1.plot(x, cas_lisse, color=UNICEF_RED, linewidth=2.5,
                 marker="o", markersize=4, label="Moyenne mobile (3 sem.)", zorder=3)

        # Zone de seuil alerte
        max_cas = max(cas) if cas else 1
        ax1.axhline(y=max_cas * 0.7, color=UNICEF_ORANGE, linestyle="--",
                    alpha=0.7, linewidth=1.5, label="Seuil alerte")

        ax1.set_title(f"Évolution des cas de paludisme — {region_name}", fontsize=12,
                      fontweight="bold", color=UNICEF_DARK)
        ax1.set_ylabel("Cas confirmés", fontsize=10)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(semaines, rotation=45, ha="right", fontsize=8)
        ax1.legend(fontsize=9, loc="upper left")
        ax1.set_facecolor("#FAFAFA")

        # Annotation pic
        if cas:
            peak_idx = cas.index(max(cas))
            ax1.annotate(
                f"Pic : {max(cas)} cas",
                xy=(peak_idx, max(cas)),
                xytext=(peak_idx, max(cas) * 1.08),
                arrowprops=dict(arrowstyle="->", color=UNICEF_RED),
                fontsize=8, color=UNICEF_RED, ha="center",
            )

        # ── Axe 2 : Taux de positivité TDR ──
        ax2 = axes[1]
        tdr = [d.get("taux_positivite_tdr_pct", 0) for d in tendance_data]
        colors_tdr = [
            UNICEF_RED if t > 40 else UNICEF_ORANGE if t > 20 else UNICEF_GREEN
            for t in tdr
        ]
        ax2.bar(x, tdr, color=colors_tdr, alpha=0.8, width=0.7)
        ax2.axhline(y=40, color=UNICEF_RED, linestyle="--", linewidth=1,
                    label="Seuil TDR (40%)")
        ax2.set_ylabel("TDR (%)", fontsize=9)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(semaines, rotation=45, ha="right", fontsize=7)
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=8)
        ax2.set_facecolor("#FAFAFA")

        # Bande UNICEF en haut
        fig.patch.set_facecolor("white")
        plt.suptitle("", y=1.0)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=self.DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_risk_gauge(
        self,
        score: float,
        niveau: str,
        titre: str,
        output_path: Path,
    ) -> Path:
        """
        Jauge circulaire (demi-cercle) affichant le score de risque.
        Inclus dans les rapports comme visuel synthétique.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"))
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.2, 1.3)
        ax.axis("off")

        # Arcs de fond (gris)
        theta = np.linspace(np.pi, 0, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#E0E0E0", linewidth=25, solid_capstyle="round")

        # Arc coloré selon le score
        color = RISK_SCORE_CMAP(score)
        angle_score = np.pi - score * np.pi
        theta_fill  = np.linspace(np.pi, angle_score, 200)
        ax.plot(
            np.cos(theta_fill), np.sin(theta_fill),
            color=color, linewidth=25, solid_capstyle="round"
        )

        # Texte central
        ax.text(0, 0.35, f"{score:.2f}", ha="center", va="center",
                fontsize=28, fontweight="bold", color=UNICEF_DARK)
        ax.text(0, 0.10, niveau.upper(), ha="center", va="center",
                fontsize=11, color=color, fontweight="bold")

        # Graduation
        for val, label in [(0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1, "1")]:
            ang = np.pi - val * np.pi
            ax.text(
                1.18 * np.cos(ang), 1.18 * np.sin(ang),
                label, ha="center", va="center", fontsize=7, color="#666"
            )

        ax.set_title(titre, fontsize=10, fontweight="bold", color=UNICEF_DARK, pad=5)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=120, bbox_inches="tight",
                    facecolor="white", transparent=False)
        plt.close(fig)
        return output_path

    # ─────────────────────────────────────────────
    # Graphiques nutritionnels
    # ─────────────────────────────────────────────

    def plot_gam_trend(
        self,
        gam_data: List[Dict],
        region_name: str,
        output_path: Path,
    ) -> Path:
        """
        Courbe d'évolution du taux GAM avec seuils OMS.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not gam_data:
            return self._placeholder_chart(output_path, "Données GAM non disponibles")

        dates = [d.get("date_observation", f"M{i}") for i, d in enumerate(gam_data)]
        gam   = [float(d.get("gam_pct", 0)) for d in gam_data]
        sam   = [float(d.get("sam_pct", 0)) for d in gam_data]

        fig, ax = plt.subplots(figsize=(12, 5))
        x = range(len(dates))

        # Zones seuils OMS en fond
        ax.axhspan(0, 5,   alpha=0.08, color=UNICEF_GREEN,  label="Acceptable (<5%)")
        ax.axhspan(5, 10,  alpha=0.08, color=UNICEF_YELLOW, label="Alerte (5-10%)")
        ax.axhspan(10, 15, alpha=0.08, color=UNICEF_ORANGE, label="Urgence (10-15%)")
        ax.axhspan(15, 40, alpha=0.08, color=UNICEF_RED,    label="Crise (≥15%)")

        # Lignes seuils OMS
        for seuil, couleur, libelle in [
            (5,  UNICEF_GREEN,  "Seuil alerte"),
            (10, UNICEF_ORANGE, "Seuil urgence"),
            (15, UNICEF_RED,    "Seuil crise"),
        ]:
            ax.axhline(y=seuil, color=couleur, linestyle="--",
                       linewidth=1.2, alpha=0.8)
            ax.text(len(dates) - 0.5, seuil + 0.3, libelle,
                    fontsize=7.5, color=couleur, ha="right")

        # Courbe GAM principale
        ax.fill_between(x, gam, alpha=0.25, color=UNICEF_ORANGE)
        ax.plot(x, gam, color=UNICEF_ORANGE, linewidth=2.5,
                marker="o", markersize=5, label="GAM (%)", zorder=3)

        # Courbe SAM
        ax.plot(x, sam, color=UNICEF_RED, linewidth=1.5,
                linestyle="--", marker="s", markersize=3,
                label="SAM (%)", zorder=3, alpha=0.8)

        ax.set_title(
            f"Évolution du taux de malnutrition aiguë (GAM) — {region_name}",
            fontsize=12, fontweight="bold", color=UNICEF_DARK
        )
        ax.set_ylabel("Taux GAM / SAM (%)", fontsize=10)
        ax.set_xticks(list(x))
        step = max(1, len(dates) // 12)
        ax.set_xticklabels(
            [d[:7] if i % step == 0 else "" for i, d in enumerate(dates)],
            rotation=45, ha="right", fontsize=8
        )
        ax.set_ylim(0, max(max(gam) * 1.2, 20))
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.set_facecolor("#FAFAFA")

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=self.DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_food_security_radar(
        self,
        disponibilite: Dict[str, Any],
        region_name: str,
        output_path: Path,
    ) -> Path:
        """
        Radar chart de la sécurité alimentaire (FCS, HDDS, disponibilités).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        categories = [
            "Céréales", "Légumineuses", "Protéines", "Légumes", "Fruits",
            "Score FCS\n(/ 42)", "Diversité\nHDDS"
        ]
        values = [
            float(disponibilite.get("disponibilite_cereales", 2)) / 3 * 100,
            float(disponibilite.get("disponibilite_legumineuses", 2)) / 3 * 100,
            float(disponibilite.get("disponibilite_proteines_animales", 1)) / 3 * 100,
            float(disponibilite.get("disponibilite_legumes", 2)) / 3 * 100,
            float(disponibilite.get("disponibilite_fruits", 2)) / 3 * 100,
            float(disponibilite.get("score_fcs", 35)) / 42 * 100,
            float(disponibilite.get("hdds", 5)) / 12 * 100,
        ]
        values = [min(100, max(0, v)) for v in values]

        N = len(categories)
        angles = [n / N * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values_plot = values + values[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # Remplissage
        ax.fill(angles, values_plot, color=UNICEF_BLUE, alpha=0.25)
        ax.plot(angles, values_plot, color=UNICEF_BLUE, linewidth=2, marker="o", markersize=6)

        # Référence optimale (100%)
        ax.plot(angles, [100] * len(angles), color="#CCC", linewidth=1,
                linestyle="--", alpha=0.5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="#888")
        ax.grid(True, alpha=0.3)

        ax.set_title(
            f"Sécurité alimentaire — {region_name}",
            fontsize=11, fontweight="bold", color=UNICEF_DARK, pad=20
        )

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=self.DEFAULT_DPI, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        return output_path

    def plot_stocks_bar(
        self,
        stocks: Dict[str, Any],
        region_name: str,
        output_path: Path,
    ) -> Path:
        """Graphique barres des stocks humanitaires (jours de couverture)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        items = {
            "RUTF (MAG sévère)":    float(stocks.get("jours_couverture_sam", 0)),
            "RUSF (MAG modérée)":   float(stocks.get("jours_couverture_mam", 0)),
        }

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(
            list(items.keys()), list(items.values()),
            color=[UNICEF_RED if v < 30 else UNICEF_ORANGE if v < 60 else UNICEF_GREEN
                   for v in items.values()],
            edgecolor="white", height=0.5,
        )

        # Seuil 30 jours
        ax.axvline(x=30, color=UNICEF_RED, linestyle="--", linewidth=1.5,
                   label="Seuil critique (30j)")
        ax.axvline(x=60, color=UNICEF_ORANGE, linestyle="--", linewidth=1,
                   label="Seuil alerte (60j)", alpha=0.7)

        for bar, val in zip(bars, items.values()):
            ax.text(
                val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} jours", va="center", fontsize=9,
                color=UNICEF_DARK, fontweight="bold"
            )

        ax.set_xlabel("Jours de couverture", fontsize=10)
        ax.set_title(f"Stocks humanitaires — {region_name}", fontsize=11,
                     fontweight="bold", color=UNICEF_DARK)
        ax.legend(fontsize=8)
        ax.set_facecolor("#FAFAFA")
        ax.set_xlim(0, max(max(items.values()) * 1.2, 90))

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=self.DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_regional_comparison(
        self,
        regions_data: List[Dict],
        metric: str,
        titre: str,
        output_path: Path,
    ) -> Path:
        """
        Graphique comparaison inter-régionale (barres horizontales triées).
        metric : "score_risque" | "gam_pct" | "score_composite"
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not regions_data:
            return self._placeholder_chart(output_path, "Données comparatives non disponibles")

        sorted_data = sorted(regions_data, key=lambda r: r.get(metric, 0), reverse=True)
        names  = [r.get("region_name", r.get("region_id", "?"))[:18] for r in sorted_data]
        values = [float(r.get(metric, 0)) for r in sorted_data]

        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.5)))

        colors = [RISK_SCORE_CMAP(v if v <= 1 else v / 20) for v in values]
        bars   = ax.barh(names, values, color=colors, edgecolor="white", height=0.7)

        # Labels valeurs
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.01 if val <= 1 else val + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}" if val <= 1 else f"{val:.1f}%",
                va="center", fontsize=8, color=UNICEF_DARK
            )

        ax.set_title(titre, fontsize=11, fontweight="bold", color=UNICEF_DARK)
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=9)
        ax.set_facecolor("#FAFAFA")
        ax.invert_yaxis()  # Plus élevé en haut

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=self.DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # ─────────────────────────────────────────────
    # Helpers internes
    # ─────────────────────────────────────────────

    def _get_risk_data_for_map(self, type_carte: str) -> List[Dict]:
        """
        Récupère les données de risque depuis Redis ou génère des données synthétiques.
        """
        try:
            import redis
            import json
            from config.settings import settings
            from src.utils.constants import REGIONS_MADAGASCAR

            r = redis.Redis.from_url(settings.redis.url, decode_responses=True)
            data = []
            for rid in REGIONS_MADAGASCAR:
                key = f"unicef:mdg:predictions:combinee:{rid}:14"
                cached = r.get(key)
                if cached:
                    pred = json.loads(cached)
                    score_key = {
                        "paludisme": "score_paludisme",
                        "nutrition": "score_nutrition",
                        "combine":   "score_composite",
                    }.get(type_carte, "score_composite")
                    data.append({
                        "region_id":   rid,
                        "score_risque": float(pred.get(score_key, 0.3)),
                        "niveau_risque": pred.get("niveau_risque", "moyen"),
                    })
            if data:
                return data
        except Exception as exc:
            logger.debug("Redis risque carte : {}", exc)

        # Données synthétiques si Redis vide
        return self._synthetic_risk_data()

    @staticmethod
    def _synthetic_risk_data() -> List[Dict]:
        """Données de risque synthétiques pour carte de démonstration."""
        from src.utils.constants import REGIONS_MADAGASCAR
        import random
        random.seed(42)  # Reproductible

        ENDEMICITE = {
            "MDG-ANA": 0.2, "MDG-VAK": 0.2, "MDG-ITM": 0.35, "MDG-BMT": 0.55,
            "MDG-MAT": 0.35, "MDG-ATI": 0.25, "MDG-VAT": 0.75, "MDG-FIT": 0.70,
            "MDG-ANO": 0.72, "MDG-ATS": 0.65, "MDG-ANA2": 0.62, "MDG-ALA": 0.45,
            "MDG-BOE": 0.68, "MDG-SOF": 0.60, "MDG-MEN": 0.55, "MDG-MEN2": 0.58,
            "MDG-DIA": 0.62, "MDG-SAV": 0.75, "MDG-IHO": 0.30, "MDG-ASO": 0.48,
            "MDG_AND": 0.25, "MDG-AAN": 0.38,
        }
        return [
            {
                "region_id":    rid,
                "score_risque": round(min(1, ENDEMICITE.get(rid, 0.4)
                                          + random.gauss(0, 0.05)), 3),
                "niveau_risque": "élevé" if ENDEMICITE.get(rid, 0.4) > 0.5 else "moyen",
            }
            for rid in REGIONS_MADAGASCAR
        ]

    @staticmethod
    def _add_region_markers(m, risk_data: List[Dict], highlight_id: Optional[str]) -> None:
        """Ajoute des marqueurs circulaires colorés pour chaque région sur Folium."""
        try:
            import folium
            from src.data_collection.weather_fetcher import REGION_COORDS, REGION_NAMES

            risk_by_id = {r["region_id"]: r for r in risk_data}
            for rid, coords in REGION_COORDS.items():
                d = risk_by_id.get(rid, {})
                score = d.get("score_risque", 0.3)

                if score >= 0.75:   color = "#D32F2F"
                elif score >= 0.5:  color = "#F57C00"
                elif score >= 0.25: color = "#FBC02D"
                else:               color = "#388E3C"

                folium.CircleMarker(
                    location=[coords["lat"], coords["lon"]],
                    radius=10 + score * 10,
                    color="#FFF",
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    weight=3 if rid == highlight_id else 1,
                    popup=folium.Popup(
                        f"<b>{REGION_NAMES.get(rid, rid)}</b><br>"
                        f"Score risque : {score:.2f}<br>"
                        f"Niveau : {d.get('niveau_risque', 'N/A')}",
                        max_width=200,
                    ),
                    tooltip=REGION_NAMES.get(rid, rid),
                ).add_to(m)
        except Exception as exc:
            logger.debug("Markers Folium : {}", exc)

    @staticmethod
    def _add_folium_legend(m, type_carte: str) -> None:
        """Ajoute une légende HTML à la carte Folium."""
        try:
            import folium
            legend = f"""
            <div style="position:fixed;bottom:20px;left:20px;z-index:1000;
                        background:white;padding:12px;border-radius:8px;
                        border:1px solid #ccc;font-size:12px;">
                <b>Niveau de risque</b><br>
                <span style="color:#388E3C">■</span> Faible (< 0.25)<br>
                <span style="color:#FBC02D">■</span> Moyen (0.25-0.5)<br>
                <span style="color:#F57C00">■</span> Élevé (0.5-0.75)<br>
                <span style="color:#D32F2F">■</span> Très élevé (> 0.75)<br>
                <hr style="margin:4px 0">
                <small>UNICEF Madagascar | {type_carte}</small>
            </div>"""
            m.get_root().html.add_child(folium.Element(legend))
        except Exception:
            pass

    @staticmethod
    def _screenshot_folium(html_path: Path, output_path: Path) -> Optional[Path]:
        """
        Convertit une carte Folium HTML en PNG via Selenium headless.
        Retourne None si Selenium non disponible.
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            import time

            opts = Options()
            opts.add_argument("--headless")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--window-size=1200,900")

            driver = webdriver.Chrome(options=opts)
            driver.get(f"file://{html_path.resolve()}")
            time.sleep(2)  # Attente chargement tiles

            png_path = output_path.with_suffix(".png")
            driver.save_screenshot(str(png_path))
            driver.quit()

            logger.debug("Screenshot Folium → {}", png_path)
            return png_path

        except Exception as exc:
            logger.debug("Screenshot Selenium : {} — PNG non généré", exc)
            return None

    @staticmethod
    def _placeholder_chart(output_path: Path, message: str) -> Path:
        """Génère un graphique placeholder quand les données sont absentes."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, message,
            ha="center", va="center",
            fontsize=13, color="#888",
            transform=ax.transAxes
        )
        ax.set_facecolor("#F5F5F5")
        ax.axis("off")
        ax.set_title("UNICEF Madagascar", fontsize=10, color=UNICEF_BLUE)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path