"""
Explicabilité des modèles ML — standard UNICEF de transparence algorithmique.

Couvre :
  - SHAPExplainer : wrapper unifié autour des valeurs SHAP
  - Force plots et Waterfall plots (exportés en PNG/SVG)
  - Rapport d'explicabilité (format UNICEF Model Card)
  - LIME (fallback si SHAP indisponible)
  - Analyse de sensibilité par feature

Interface attendue par predictions.py (router) :
  SHAPExplainer(model).explain(features, region_id) → dict

    Clés de sortie :
      region_id, modele, date_prediction, valeur_predite, valeur_base,
      features: [{nom, valeur, shap_value, contribution_pct, direction}],
      force_plot_url, waterfall_url
"""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import settings


class SHAPExplainer:
    """
    Wrapper unifié pour l'explicabilité SHAP des modèles du projet.

    Supporte :
      - TreeExplainer (XGBoost, Random Forest) — exact, rapide
      - LinearExplainer (modèles linéaires) — exact
      - KernelExplainer (tout modèle) — approx, lent, utilisé pour MLP

    Usage (appelé par predictions.py router) :
        from src.models.malaria_predictor import MalariaPredictor
        model = MalariaPredictor.load_latest()
        explainer = SHAPExplainer(model)
        result = explainer.explain(features, region_id="MDG-ANA")
    """

    # Répertoire de sauvegarde des graphiques
    PLOTS_DIR = Path("data/processed/shap_plots")

    def __init__(self, model):
        """
        Args:
            model : instance de BasePredictor (MalariaPredictor ou NutritionPredictor)
        """
        self._model       = model
        self._explainer   = None
        self._base_value  = 0.5
        self._shap_ready  = False
        self.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initialise le bon type d'explainer SHAP selon le modèle."""
        try:
            import shap

            # Récupère le modèle sous-jacent sklearn/xgboost
            underlying = getattr(self._model, "_clf", None) or \
                         getattr(self._model, "_rf",  None) or \
                         getattr(self._model, "_model", None)

            if underlying is None:
                logger.warning("Modèle sous-jacent None — SHAP désactivé")
                return

            model_class = type(underlying).__name__

            if any(t in model_class for t in ("XGB", "GradientBoosting", "RandomForest", "DecisionTree", "ExtraTrees")):
                self._explainer = shap.TreeExplainer(underlying)
                logger.debug("SHAP TreeExplainer initialisé pour {}", model_class)

            elif any(t in model_class for t in ("LogisticRegression", "LinearSVC", "Ridge", "Lasso")):
                # Besoin d'un background dataset — on utilise le dataset moyen
                n_features = len(self._model.get_feature_names())
                background = np.zeros((1, n_features))
                self._explainer = shap.LinearExplainer(underlying, background)
                logger.debug("SHAP LinearExplainer initialisé pour {}", model_class)

            else:
                # MLP et autres → KernelExplainer (lent mais universel)
                logger.info("KernelExplainer SHAP pour {} (plus lent)", model_class)
                n_features = len(self._model.get_feature_names())
                background = np.zeros((10, n_features))

                if hasattr(underlying, "predict_proba"):
                    def predict_fn(x):
                        return underlying.predict_proba(x)[:, 1]
                else:
                    predict_fn = underlying.predict

                self._explainer = shap.KernelExplainer(predict_fn, background)

            self._shap_ready = True

            # Valeur de base (E[f(x)])
            try:
                if hasattr(self._explainer, "expected_value"):
                    ev = self._explainer.expected_value
                    self._base_value = float(
                        ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else ev
                    )
            except Exception:
                self._base_value = 0.5

        except ImportError:
            logger.warning("SHAP non installé — explicabilité dégradée (LIME fallback)")
        except Exception as exc:
            logger.warning("SHAP init échoué : {}", exc)

    # ─────────────────────────────────────────────
    # Interface principale (appelée par predictions.py)
    # ─────────────────────────────────────────────

    def explain(
        self,
        features: Dict[str, Any],
        region_id: str = "unknown",
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Calcule l'explication SHAP pour une prédiction.

        Retourne le dict attendu par le router predictions.py :
          GET /api/v1/predictions/explicabilite/{region_id}/{modele}
        """
        feature_names = self._model.get_feature_names()
        X = np.array(
            [float(features.get(name, 0.0)) for name in feature_names],
            dtype=np.float32,
        ).reshape(1, -1)

        # Scaling si présent
        if getattr(self._model, "_scaler", None) is not None:
            X = self._model._scaler.transform(X)

        # 1. Prédiction brute
        try:
            raw_pred = self._model._predict_raw(X)
            valeur_predite = float(np.clip(raw_pred[0], 0.0, 1.0))
        except Exception:
            valeur_predite = 0.5

        # 2. Calcul SHAP
        shap_features = []
        force_plot_url    = None
        waterfall_url     = None

        if self._shap_ready and self._explainer is not None:
            shap_features, force_plot_url, waterfall_url = self._compute_shap_explanation(
                X=X,
                features=features,
                feature_names=feature_names,
                valeur_predite=valeur_predite,
                region_id=region_id,
                generate_plots=generate_plots,
            )
        else:
            # Fallback LIME
            shap_features = self._lime_fallback(features, feature_names, valeur_predite)

        return {
            "region_id":       region_id,
            "modele":          self._model.MODEL_NAME,
            "date_prediction": datetime.utcnow().isoformat(),
            "valeur_predite":  round(valeur_predite, 4),
            "valeur_base":     round(self._base_value, 4),
            "features":        shap_features,
            "force_plot_url":  force_plot_url,
            "waterfall_url":   waterfall_url,
            "methode":         "SHAP" if self._shap_ready else "LIME (fallback)",
            "nb_features":     len(feature_names),
        }

    # ─────────────────────────────────────────────
    # Calcul SHAP + génération graphiques
    # ─────────────────────────────────────────────

    def _compute_shap_explanation(
        self,
        X: np.ndarray,
        features: Dict[str, Any],
        feature_names: List[str],
        valeur_predite: float,
        region_id: str,
        generate_plots: bool,
    ) -> Tuple[List[Dict], Optional[str], Optional[str]]:
        """Calcule les SHAP values et génère les graphiques."""
        try:
            raw_shap = self._explainer.shap_values(X)

            # Gestion format binary classifier (liste de 2 arrays)
            if isinstance(raw_shap, list):
                shap_vals = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
            else:
                shap_vals = raw_shap

            sv = shap_vals[0]  # Vecteur de SHAP pour le sample unique
            total_abs = np.sum(np.abs(sv)) + 1e-9

            # Construction features expliquées (toutes, triées par impact)
            feat_explained = []
            for i, name in enumerate(feature_names):
                shap_v = float(sv[i])
                feat_explained.append({
                    "nom":             name,
                    "valeur":          round(float(features.get(name, 0.0)), 4),
                    "shap_value":      round(shap_v, 4),
                    "contribution_pct": round(abs(shap_v) / total_abs * 100, 1),
                    "direction":       "hausse_risque" if shap_v > 0 else "baisse_risque",
                    "rang_importance": 0,  # Calculé après tri
                })

            # Tri par impact absolu décroissant
            feat_explained.sort(
                key=lambda x: abs(x["shap_value"]), reverse=True
            )
            for i, f in enumerate(feat_explained):
                f["rang_importance"] = i + 1

            # Génération graphiques
            force_plot_url  = None
            waterfall_url   = None

            if generate_plots:
                force_plot_url  = self._generate_force_plot(
                    shap_vals=sv,
                    base_value=self._base_value,
                    feature_names=feature_names,
                    feature_values=[features.get(n, 0) for n in feature_names],
                    region_id=region_id,
                )
                waterfall_url = self._generate_waterfall_plot(
                    shap_vals=sv,
                    base_value=self._base_value,
                    valeur_predite=valeur_predite,
                    feature_names=feature_names,
                    feature_values=[features.get(n, 0) for n in feature_names],
                    region_id=region_id,
                )

            return feat_explained, force_plot_url, waterfall_url

        except Exception as exc:
            logger.error("SHAP compute échoué : {}", exc)
            return self._lime_fallback(
                features, feature_names, valeur_predite
            ), None, None

    # ─────────────────────────────────────────────
    # Génération graphiques SHAP
    # ─────────────────────────────────────────────

    def _generate_force_plot(
        self,
        shap_vals: np.ndarray,
        base_value: float,
        feature_names: List[str],
        feature_values: List[float],
        region_id: str,
    ) -> Optional[str]:
        """
        Génère un SHAP Force Plot et le sauvegarde en PNG.
        Retourne le chemin relatif pour l'URL de téléchargement.
        """
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(14, 3))

            # SHAP Force Plot (HTML → image via matplotlib)
            top_n = 10
            top_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]

            positive = [(feature_names[i], float(shap_vals[i]))
                        for i in top_idx if shap_vals[i] > 0]
            negative = [(feature_names[i], float(shap_vals[i]))
                        for i in top_idx if shap_vals[i] < 0]

            y_pos = base_value
            x = 0
            bar_height = 0.4

            # Barres positives (rouge)
            for name, sv in positive:
                ax.barh(
                    y=0, width=sv, left=x, height=bar_height,
                    color="#FF4444", alpha=0.8, label=f"{name}: +{sv:.3f}"
                )
                x += sv

            # Barres négatives (bleues)
            for name, sv in negative:
                ax.barh(
                    y=0, width=sv, left=x, height=bar_height,
                    color="#4444FF", alpha=0.8, label=f"{name}: {sv:.3f}"
                )
                x += sv

            ax.axvline(x=base_value, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(
                f"SHAP Force Plot — {self._model.MODEL_NAME} | {region_id}",
                fontsize=11
            )
            ax.set_xlabel("SHAP values (impact sur score risque)")
            ax.legend(loc="upper right", fontsize=7, ncol=3)
            plt.tight_layout()

            plot_path = self.PLOTS_DIR / f"force_{region_id}_{self._model.MODEL_NAME}.png"
            plt.savefig(str(plot_path), dpi=120, bbox_inches="tight")
            plt.close(fig)

            return f"/plots/shap/force_{region_id}_{self._model.MODEL_NAME}.png"

        except Exception as exc:
            logger.debug("Force plot échoué : {}", exc)
            return None

    def _generate_waterfall_plot(
        self,
        shap_vals: np.ndarray,
        base_value: float,
        valeur_predite: float,
        feature_names: List[str],
        feature_values: List[float],
        region_id: str,
    ) -> Optional[str]:
        """
        Génère un SHAP Waterfall Plot (lecture verticale feature par feature).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            # Sélectionne le top 12 features par impact
            top_n = 12
            top_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
            top_idx = top_idx[::-1]  # Ordre ascendant pour plot horizontal

            names  = [feature_names[i] for i in top_idx]
            values = shap_vals[top_idx]
            fvals  = [feature_values[i] for i in top_idx]

            fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.6)))

            colors = ["#FF4444" if v > 0 else "#4444FF" for v in values]

            # Barres horizontales
            bars = ax.barh(
                y=range(len(names)),
                width=values,
                color=colors,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )

            # Labels
            for i, (bar, name, fv, sv) in enumerate(zip(bars, names, fvals, values)):
                label = f"{name} = {fv:.2f}"
                ax.text(
                    sv + (0.002 if sv >= 0 else -0.002),
                    i,
                    f"{sv:+.3f}",
                    va="center",
                    ha="left" if sv >= 0 else "right",
                    fontsize=8,
                )

            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([
                f"{n} = {v:.2f}" for n, v in zip(names, fvals)
            ], fontsize=8)
            ax.axvline(x=0, color="black", linewidth=0.8)
            ax.set_xlabel("Contribution SHAP au score de risque")
            ax.set_title(
                f"SHAP Waterfall — {self._model.MODEL_NAME.upper()} | {region_id}\n"
                f"Base: {base_value:.3f} → Prédiction: {valeur_predite:.3f}",
                fontsize=10,
            )

            # Légende
            patch_pos = mpatches.Patch(color="#FF4444", alpha=0.8, label="Augmente le risque")
            patch_neg = mpatches.Patch(color="#4444FF", alpha=0.8, label="Réduit le risque")
            ax.legend(handles=[patch_pos, patch_neg], loc="lower right", fontsize=8)

            plt.tight_layout()
            plot_path = self.PLOTS_DIR / f"waterfall_{region_id}_{self._model.MODEL_NAME}.png"
            plt.savefig(str(plot_path), dpi=120, bbox_inches="tight")
            plt.close(fig)

            return f"/plots/shap/waterfall_{region_id}_{self._model.MODEL_NAME}.png"

        except Exception as exc:
            logger.debug("Waterfall plot échoué : {}", exc)
            return None

    # ─────────────────────────────────────────────
    # Fallback LIME (si SHAP indisponible)
    # ─────────────────────────────────────────────

    def _lime_fallback(
        self,
        features: Dict[str, Any],
        feature_names: List[str],
        valeur_predite: float,
    ) -> List[Dict[str, Any]]:
        """
        Approximation LIME quand SHAP n'est pas disponible.
        Utilise une perturbation locale des features.
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
            import numpy as np

            # Dataset de fond synthétique (centré)
            n_background = 50
            background = np.random.randn(n_background, len(feature_names))

            X_instance = np.array(
                [float(features.get(n, 0.0)) for n in feature_names]
            ).reshape(1, -1)

            def predict_fn(x):
                preds = []
                for row in x:
                    feat_dict = {n: v for n, v in zip(feature_names, row)}
                    try:
                        raw = self._model._predict_raw(
                            self._model._features_to_array(feat_dict)
                        )
                        preds.append([1 - float(raw[0]), float(raw[0])])
                    except Exception:
                        preds.append([0.5, 0.5])
                return np.array(preds)

            explainer = LimeTabularExplainer(
                background,
                feature_names=feature_names,
                mode="classification",
                discretize_continuous=False,
            )
            exp = explainer.explain_instance(
                X_instance[0],
                predict_fn,
                num_features=10,
            )
            lime_list = exp.as_list()

            total_abs = sum(abs(v) for _, v in lime_list) + 1e-9
            result = []
            for name_expr, coef in lime_list:
                # LIME retourne des expressions comme "feature_5 > 0.5"
                feat_name = name_expr.split(" ")[0] if " " in name_expr else name_expr
                result.append({
                    "nom":             feat_name,
                    "valeur":          round(float(features.get(feat_name, 0.0)), 4),
                    "shap_value":      round(float(coef), 4),
                    "contribution_pct": round(abs(coef) / total_abs * 100, 1),
                    "direction":       "hausse_risque" if coef > 0 else "baisse_risque",
                    "rang_importance": 0,
                    "methode":         "LIME",
                })

            result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            for i, f in enumerate(result):
                f["rang_importance"] = i + 1

            return result

        except ImportError:
            logger.debug("LIME non disponible — fallback par valeur brute")
            return self._basic_fallback(features, feature_names)
        except Exception as exc:
            logger.debug("LIME échoué : {}", exc)
            return self._basic_fallback(features, feature_names)

    @staticmethod
    def _basic_fallback(
        features: Dict[str, Any],
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Fallback minimal : tri des features par valeur absolue normalisée."""
        items = []
        values = [abs(float(features.get(n, 0))) for n in feature_names]
        total  = sum(values) + 1e-9

        for i, name in enumerate(feature_names):
            v = float(features.get(name, 0.0))
            items.append({
                "nom":             name,
                "valeur":          round(v, 4),
                "shap_value":      0.0,
                "contribution_pct": round(abs(v) / total * 100, 1),
                "direction":       "inconnu",
                "rang_importance": i + 1,
                "methode":         "Heuristique (SHAP/LIME indisponibles)",
            })

        items.sort(key=lambda x: x["contribution_pct"], reverse=True)
        for i, f in enumerate(items):
            f["rang_importance"] = i + 1
        return items[:10]

    # ─────────────────────────────────────────────
    # Rapport d'explicabilité global (Model Card UNICEF)
    # ─────────────────────────────────────────────

    def generate_model_card(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Génère une Model Card au format UNICEF.
        Document de transparence algorithmique requis pour déploiement.
        """
        model_info = self._model.get_health_info()

        card = {
            "model_card_version": "1.0",
            "date_generation": datetime.utcnow().isoformat(),
            "organisation": "UNICEF Madagascar",
            "modele": {
                "nom":     self._model.MODEL_NAME,
                "version": self._model.MODEL_VERSION,
                "type":    self._model.MODEL_TYPE,
                "date_entrainement": model_info.get("date_entrainement"),
            },
            "performances": model_info.get("metriques", {}),
            "donnees_entrainement": {
                "periode": "2021-2024",
                "regions": "22 régions Madagascar",
                "nb_features": len(self._model.get_feature_names()),
                "features_principales": self._model.get_feature_names()[:5],
                "sources": [
                    "DHIS2 Madagascar",
                    "NASA POWER",
                    "OpenWeatherMap",
                    "WFP VAM",
                    "FAO FAOSTAT",
                ],
            },
            "limites_connues": [
                "Données DHIS2 parfois incomplètes pour régions enclavées",
                "NDVI satellite indisponible en période de forte couverture nuageuse",
                "Données nutrition < 1 an pour certaines régions",
                "Modèle non validé pour les cyclones de catégorie 5",
            ],
            "biais_potentiels": [
                "Sous-représentation Grand Sud (données rares)",
                "Modèle entraîné avant 2024 — dérive possible sur nouvelles variantes",
            ],
            "utilisation_recommandee": [
                "Aide à la décision — jamais source unique de décision",
                "Toujours croiser avec expertise terrain agents de santé",
                "Retraining recommandé si PSI > 0.15",
            ],
            "contacts": {
                "modele": "tech@unicef-madagascar.org",
                "donnees": "data@unicef-madagascar.org",
            },
            "explicabilite": {
                "methode": "SHAP" if self._shap_ready else "LIME",
                "disponible": self._shap_ready,
                "valeur_base": round(self._base_value, 4),
            },
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(card, f, indent=2, ensure_ascii=False)
            logger.info("Model Card sauvegardée → {}", output_path)

        return card

    # ─────────────────────────────────────────────
    # Analyse de sensibilité (what-if feature level)
    # ─────────────────────────────────────────────

    def sensitivity_analysis(
        self,
        features: Dict[str, Any],
        target_feature: str,
        values: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Analyse de sensibilité : impact d'une feature sur le score.
        Utile pour les simulations what-if des agents de terrain.

        Args:
            features       : dict de features de base
            target_feature : feature à faire varier
            values         : liste de valeurs à tester

        Returns:
            Liste de {valeur, score_risque, variation_score}
        """
        results = []
        base_pred = self._model.predict(features)
        base_score = base_pred.get("score_risque", 0.5)

        for val in values:
            modified = dict(features)
            modified[target_feature] = val
            pred = self._model.predict(modified)
            score = pred.get("score_risque", 0.5)
            results.append({
                "valeur": val,
                "score_risque": round(score, 4),
                "variation_score": round(score - base_score, 4),
                "variation_pct": round((score - base_score) / (base_score + 1e-9) * 100, 1),
                "niveau_risque": pred.get("niveau_risque", ""),
            })

        return results

    def global_feature_importance(
        self,
        X_sample: np.ndarray,
        top_n: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Importance globale des features via SHAP values moyennées sur un sample.
        Utile pour les rapports de performance UNICEF.
        """
        if not self._shap_ready or self._explainer is None:
            return []

        try:
            shap_values = self._explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values

            # Importance globale = mean(|SHAP|) par feature
            mean_abs = np.mean(np.abs(sv), axis=0)
            total    = np.sum(mean_abs) + 1e-9
            feat_names = self._model.get_feature_names()

            top_idx = np.argsort(mean_abs)[::-1][:top_n]
            return [
                {
                    "rang":             i + 1,
                    "feature":          feat_names[idx] if idx < len(feat_names) else f"f_{idx}",
                    "importance_shap":  round(float(mean_abs[idx]), 4),
                    "contribution_pct": round(float(mean_abs[idx] / total * 100), 1),
                }
                for i, idx in enumerate(top_idx)
            ]

        except Exception as exc:
            logger.warning("Global feature importance échouée : {}", exc)
            return []