"""
Modèle de prévision météo basé sur LSTM (TensorFlow/Keras) + Prophet.

Rôle dans le pipeline :
  Affine les prévisions météo au-delà des 7j d'OpenWeatherMap,
  détecte les anomalies climatiques précoces, et génère des scénarios.

Architecture hybride :
  1. LSTM Bidirectionnel (séquences 30j → prévision 14j)
     Capte les patterns temporels complexes (cyclones, ENSO)
  2. Prophet (Facebook/Meta) en fallback
     Robuste pour la saisonnalité et les tendances Madagascar

Variables prédites (multi-output) :
  temperature_moy_c     (°C)
  precipitations_mm     (mm/j)
  humidite_moy_pct      (%)

Séquence d'entrée : 30 jours de données météo historiques
Horizon de prédiction : 1-30 jours

Utilisé par :
  - src/preprocessing/feature_engineering.py (enrichissement features météo)
  - src/api/routers/weather.py (prévisions longue portée)
  - Affinement des features pour MalariaPredictor
"""

from __future__ import annotations

import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from config.settings import settings


# Variables météo prédites par le LSTM
WEATHER_TARGET_VARS = ["temperature_moy_c", "precipitations_mm", "humidite_moy_pct"]

# Variables d'entrée pour le LSTM (features par pas de temps)
WEATHER_INPUT_VARS = [
    "temperature_moy_c",
    "temperature_min_c",
    "temperature_max_c",
    "precipitations_mm",
    "humidite_moy_pct",
    "vent_kmh",
    "rayonnement_solaire_mj",
    "humidite_sol_fraction",
]

# Paramètres de séquence
SEQ_LEN         = 30   # 30 jours d'historique en entrée
FORECAST_HORIZON = 14  # 14 jours de prévision en sortie
N_FEATURES      = len(WEATHER_INPUT_VARS)


class WeatherForecaster:
    """
    Prévisionniste météo hybride LSTM + Prophet.

    Interface cohérente avec l'écosystème existant (pas d'héritage BasePredictor
    car tâche de régression multi-output différente de classification).

    Usage :
        forecaster = WeatherForecaster.load_latest()
        previsions = forecaster.predict_sequence(historical_data, horizon=14)
        anomalies  = forecaster.detect_anomalies(historical_data)
    """

    MODEL_NAME:    ClassVar[str] = "weather_forecaster"
    MODEL_VERSION: ClassVar[str] = "1.0.0"

    def __init__(self):
        self._lstm_model  = None   # Keras Sequential
        self._prophet_models: Dict[str, Any] = {}  # {var: Prophet}
        self._scaler_X    = None   # MinMaxScaler pour features d'entrée
        self._scaler_y    = None   # MinMaxScaler pour targets
        self._trained_at: Optional[datetime] = None
        self._metrics: Dict[str, float] = {}
        self._model_dir = Path(settings.ml.model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # Interface publique
    # ─────────────────────────────────────────────

    def predict_sequence(
        self,
        historical_data: List[Dict[str, Any]],
        horizon: int = 14,
    ) -> List[Dict[str, Any]]:
        """
        Prédit les conditions météo sur `horizon` jours.

        Args:
            historical_data : liste de dicts journaliers (30 derniers jours minimum)
                              Clés requises : les WEATHER_INPUT_VARS
            horizon         : nombre de jours à prédire (1-30)

        Returns:
            Liste de dicts journaliers avec les prévisions
        """
        if len(historical_data) < SEQ_LEN:
            logger.warning(
                "Historique insuffisant ({} j < {} j requis) — fallback Prophet",
                len(historical_data), SEQ_LEN
            )
            return self._predict_prophet(historical_data, horizon)

        if self._lstm_model is None:
            logger.info("LSTM non entraîné — utilisation Prophet")
            return self._predict_prophet(historical_data, horizon)

        try:
            return self._predict_lstm(historical_data, horizon)
        except Exception as exc:
            logger.warning("LSTM predict échoué : {} — fallback Prophet", exc)
            return self._predict_prophet(historical_data, horizon)

    def detect_anomalies(
        self,
        historical_data: List[Dict[str, Any]],
        zscore_threshold: float = 2.5,
    ) -> List[Dict[str, Any]]:
        """
        Détecte les anomalies climatiques dans une série historique.
        Méthode : Z-score sur fenêtre glissante 30 jours.

        Returns:
            Liste d'anomalies détectées avec sévérité et impact estimé.
        """
        if len(historical_data) < 14:
            return []

        anomalies = []

        for var in ["temperature_moy_c", "precipitations_mm"]:
            values = np.array([
                float(d.get(var, 0)) for d in historical_data
            ])

            # Z-score glissant sur 30j
            for i in range(len(values)):
                window_start = max(0, i - 30)
                window = values[window_start:i + 1]
                if len(window) < 7:
                    continue

                mean = np.mean(window)
                std  = np.std(window) + 1e-9
                z    = (values[i] - mean) / std

                if abs(z) > zscore_threshold:
                    record = historical_data[i]
                    date_str = record.get("date", str(date.today()))

                    # Classification de l'anomalie
                    type_anomalie, impact = self._classifier_anomalie(var, z, values[i])

                    anomalies.append({
                        "date": date_str,
                        "variable": var,
                        "valeur_observee": round(float(values[i]), 2),
                        "valeur_normale": round(float(mean), 2),
                        "z_score": round(float(z), 2),
                        "type_anomalie": type_anomalie,
                        "severite": "extreme" if abs(z) > 4 else "severe" if abs(z) > 3 else "moderate",
                        "impact_paludisme": impact["paludisme"],
                        "impact_nutrition": impact["nutrition"],
                    })

        # Tri par date et déduplication
        seen = set()
        result = []
        for a in sorted(anomalies, key=lambda x: x["date"]):
            key = f"{a['date']}_{a['type_anomalie']}"
            if key not in seen:
                seen.add(key)
                result.append(a)

        logger.debug(
            "Anomalies détectées : {} sur {} jours de données",
            len(result), len(historical_data)
        )
        return result

    def get_health_info(self) -> Dict[str, Any]:
        """Compatible avec le endpoint /sante-modeles."""
        return {
            "modele": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "date_entrainement": self._trained_at.isoformat() if self._trained_at else None,
            "metriques": self._metrics,
            "backend": "LSTM" if self._lstm_model else "Prophet",
            "prophet_variables": list(self._prophet_models.keys()),
            "lstm_disponible": self._lstm_model is not None,
        }

    # ─────────────────────────────────────────────
    # Prédiction LSTM
    # ─────────────────────────────────────────────

    def _predict_lstm(
        self,
        historical_data: List[Dict],
        horizon: int,
    ) -> List[Dict[str, Any]]:
        """Prédiction via le modèle LSTM Keras."""
        # Vectorisation des SEQ_LEN derniers jours
        X = self._prepare_lstm_input(historical_data[-SEQ_LEN:])

        # Prédiction récursive (1 pas à la fois)
        previsions = []
        input_seq = X.copy()

        for step in range(min(horizon, FORECAST_HORIZON)):
            pred_scaled = self._lstm_model.predict(
                input_seq[np.newaxis, :, :], verbose=0
            )[0]

            # Inverse-transform
            if self._scaler_y:
                pred = self._scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
            else:
                pred = pred_scaled

            dt = date.today() + timedelta(days=step + 1)
            previsions.append({
                "date": str(dt),
                "temperature_moy_c":  round(float(np.clip(pred[0], -5, 45)), 2),
                "precipitations_mm":  round(float(np.clip(pred[1], 0, 500)), 2),
                "humidite_moy_pct":   round(float(np.clip(pred[2], 0, 100)), 1),
                "source":             "LSTM",
                "confiance":          self._lstm_confidence(step),
            })

            # Mise à jour fenêtre (rolling)
            new_row = self._dict_to_row(previsions[-1])
            if self._scaler_X:
                new_row_scaled = self._scaler_X.transform(new_row.reshape(1, -1))[0]
            else:
                new_row_scaled = new_row
            input_seq = np.vstack([input_seq[1:], new_row_scaled])

        # Si horizon > FORECAST_HORIZON → compléter avec Prophet
        if horizon > FORECAST_HORIZON:
            extra = self._predict_prophet(historical_data, horizon - FORECAST_HORIZON)
            for i, p in enumerate(extra):
                p["date"] = str(date.today() + timedelta(days=FORECAST_HORIZON + i + 1))
                p["source"] = "LSTM+Prophet"
            previsions.extend(extra)

        return previsions[:horizon]

    def _prepare_lstm_input(self, data: List[Dict]) -> np.ndarray:
        """Vectorise les données historiques pour le LSTM."""
        X = np.zeros((len(data), N_FEATURES), dtype=np.float32)
        for i, row in enumerate(data):
            for j, var in enumerate(WEATHER_INPUT_VARS):
                X[i, j] = float(row.get(var, 0.0))

        if self._scaler_X:
            X = self._scaler_X.transform(X)
        return X

    def _dict_to_row(self, pred: Dict) -> np.ndarray:
        """Convertit une prévision en vecteur de features pour step suivant."""
        row = np.zeros(N_FEATURES, dtype=np.float32)
        for i, var in enumerate(WEATHER_INPUT_VARS):
            row[i] = float(pred.get(var, 0.0))
        return row

    @staticmethod
    def _lstm_confidence(step: int) -> float:
        """Confiance décroissante avec l'horizon."""
        return round(max(0.5, 0.95 - step * 0.03), 2)

    # ─────────────────────────────────────────────
    # Prédiction Prophet (fallback)
    # ─────────────────────────────────────────────

    def _predict_prophet(
        self,
        historical_data: List[Dict],
        horizon: int,
    ) -> List[Dict[str, Any]]:
        """
        Prédiction via Prophet (Facebook/Meta).
        Robuste pour saisonnalité tropicale et tendances.
        """
        if not historical_data:
            return self._climatologie_fallback(horizon)

        # Entraîne Prophet à la volée sur les données disponibles
        previsions_par_var: Dict[str, List[float]] = {}

        for var in WEATHER_TARGET_VARS:
            try:
                from prophet import Prophet
                import pandas as pd

                df = pd.DataFrame([
                    {
                        "ds": pd.to_datetime(d.get("date", str(date.today()))),
                        "y":  float(d.get(var, 0)),
                    }
                    for d in historical_data
                    if d.get(var) is not None
                ])

                if len(df) < 7:
                    raise ValueError("Données insuffisantes pour Prophet")

                # Configuration Prophet pour Madagascar
                m = Prophet(
                    seasonality_mode="multiplicative",
                    yearly_seasonality=True,
                    weekly_seasonality=False,  # Pas de saisonnalité hebdo pour météo
                    daily_seasonality=False,
                    changepoint_prior_scale=0.15,  # Flexibilité
                    interval_width=0.80,
                )

                # Saison cyclones (Nov-Avr) — important pour Madagascar
                m.add_seasonality(
                    name="cyclone_season",
                    period=365.25 / 2,
                    fourier_order=5,
                )

                m.fit(df)

                future = m.make_future_dataframe(periods=horizon)
                forecast = m.predict(future)

                # Récupère les prévisions futures uniquement
                future_forecast = forecast.tail(horizon)
                previsions_par_var[var] = future_forecast["yhat"].tolist()

            except ImportError:
                logger.debug("Prophet non disponible pour var {} — fallback climatologie", var)
                previsions_par_var[var] = self._moyenne_mobile_fallback(
                    historical_data, var, horizon
                )
            except Exception as exc:
                logger.debug("Prophet échoué pour {} : {} — fallback", var, exc)
                previsions_par_var[var] = self._moyenne_mobile_fallback(
                    historical_data, var, horizon
                )

        # Assemble les prévisions multi-variables
        result = []
        for step in range(horizon):
            dt = date.today() + timedelta(days=step + 1)
            result.append({
                "date": str(dt),
                "temperature_moy_c": round(float(
                    previsions_par_var.get("temperature_moy_c", [25.0] * horizon)[step]
                ), 2),
                "precipitations_mm": round(max(0, float(
                    previsions_par_var.get("precipitations_mm", [3.0] * horizon)[step]
                )), 2),
                "humidite_moy_pct": round(float(np.clip(
                    previsions_par_var.get("humidite_moy_pct", [75.0] * horizon)[step],
                    0, 100
                )), 1),
                "source": "Prophet",
                "confiance": round(max(0.5, 0.85 - step * 0.02), 2),
            })

        return result

    @staticmethod
    def _moyenne_mobile_fallback(
        data: List[Dict], var: str, horizon: int, window: int = 14
    ) -> List[float]:
        """Moyenne mobile sur les `window` derniers jours comme prévision."""
        recent = [float(d.get(var, 0)) for d in data[-window:] if d.get(var) is not None]
        mean   = sum(recent) / len(recent) if recent else 0.0
        return [mean] * horizon

    def _climatologie_fallback(self, horizon: int) -> List[Dict[str, Any]]:
        """Valeurs climatologiques moyennes Madagascar si toutes les sources échouent."""
        result = []
        for step in range(horizon):
            dt = date.today() + timedelta(days=step + 1)
            mois = dt.month
            # Température moyenne selon saison
            temp = 24 if 5 <= mois <= 10 else 27
            pluie = 1.5 if 5 <= mois <= 10 else 8.0
            result.append({
                "date": str(dt),
                "temperature_moy_c": temp,
                "precipitations_mm": pluie,
                "humidite_moy_pct": 72.0,
                "source": "Climatologie par défaut",
                "confiance": 0.4,
            })
        return result

    # ─────────────────────────────────────────────
    # Entraînement LSTM
    # ─────────────────────────────────────────────

    def fit_lstm(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.15,
    ) -> "WeatherForecaster":
        """
        Entraîne le modèle LSTM.

        Args:
            sequences : array shape (n_samples, SEQ_LEN, N_FEATURES)
            targets   : array shape (n_samples, len(WEATHER_TARGET_VARS))
            epochs    : nombre d'époques
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, callbacks

            logger.info(
                "Entraînement LSTM WeatherForecaster — {} séquences, {} epochs",
                len(sequences), epochs
            )

            # Architecture LSTM bidirectionnel
            model = tf.keras.Sequential([
                layers.Input(shape=(SEQ_LEN, N_FEATURES)),
                layers.Bidirectional(
                    layers.LSTM(128, return_sequences=True, dropout=0.2)
                ),
                layers.Bidirectional(
                    layers.LSTM(64, return_sequences=False, dropout=0.2)
                ),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dense(len(WEATHER_TARGET_VARS)),  # Output : 3 variables météo
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="huber",
                metrics=["mae"],
            )

            early_stop = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            )
            checkpoint_path = self._model_dir / f"{self.MODEL_NAME}_best.keras"
            checkpoint = callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
            )

            history = model.fit(
                sequences, targets,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop, reduce_lr, checkpoint],
                verbose=1,
            )

            self._lstm_model = model
            self._trained_at = datetime.utcnow()

            # Métriques finales
            val_mae = float(min(history.history.get("val_mae", [999])))
            val_loss = float(min(history.history.get("val_loss", [999])))
            self._metrics = {
                "val_mae":  round(val_mae, 4),
                "val_loss": round(val_loss, 4),
                "epochs_completed": len(history.history["loss"]),
            }

            logger.info(
                "LSTM entraîné — val_mae={:.3f} val_loss={:.4f}",
                val_mae, val_loss
            )

        except ImportError:
            logger.warning("TensorFlow non disponible — LSTM désactivé")
        except Exception as exc:
            logger.error("Erreur entraînement LSTM : {}", exc)

        return self

    # ─────────────────────────────────────────────
    # Calcul SPI et autres indices climatiques
    # ─────────────────────────────────────────────

    @staticmethod
    def compute_spi(
        precipitations: List[float],
        scale: int = 30,
    ) -> float:
        """
        Standardized Precipitation Index.
        SPI < -1 : sécheresse modérée.
        SPI < -2 : sécheresse sévère.
        """
        if len(precipitations) < scale:
            return 0.0
        window = np.array(precipitations[-scale:])
        mean = np.mean(window)
        std  = np.std(window) + 1e-9
        return round(float((np.sum(window) / scale - mean) / std), 3)

    @staticmethod
    def compute_consecutive_dry_days(
        precipitations: List[float],
        threshold_mm: float = 1.0,
    ) -> int:
        """Nombre de jours consécutifs sans pluie (< threshold_mm)."""
        count = 0
        for p in reversed(precipitations):
            if p < threshold_mm:
                count += 1
            else:
                break
        return count

    def generate_scenarios(
        self,
        historical_data: List[Dict],
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Génère des scénarios climatiques pour le what-if (predictions.py).

        Scénarios disponibles :
          - normal    : prévision standard
          - cyclone   : +300% pluies + vent fort
          - secheresse: -70% pluies
          - canicule  : +3°C température
        """
        scenarios = scenarios or ["normal", "cyclone", "secheresse", "canicule"]
        results: Dict[str, List[Dict]] = {}

        # Baseline
        baseline = self.predict_sequence(historical_data, horizon=14)
        results["normal"] = baseline

        if "cyclone" in scenarios:
            cyclone = []
            for p in baseline:
                c = dict(p)
                c["precipitations_mm"] = min(400, p["precipitations_mm"] * 4.0)
                c["humidite_moy_pct"]  = min(100, p["humidite_moy_pct"] * 1.15)
                c["source"] = "Scénario Cyclone"
                cyclone.append(c)
            results["cyclone"] = cyclone

        if "secheresse" in scenarios:
            seche = []
            for p in baseline:
                s = dict(p)
                s["precipitations_mm"] = p["precipitations_mm"] * 0.3
                s["humidite_moy_pct"]  = max(20, p["humidite_moy_pct"] * 0.7)
                s["source"] = "Scénario Sécheresse"
                seche.append(s)
            results["secheresse"] = seche

        if "canicule" in scenarios:
            canicule = []
            for p in baseline:
                k = dict(p)
                k["temperature_moy_c"] = min(45, p["temperature_moy_c"] + 3.0)
                k["humidite_moy_pct"]  = max(30, p["humidite_moy_pct"] - 10)
                k["source"] = "Scénario Canicule"
                canicule.append(k)
            results["canicule"] = canicule

        return results

    # ─────────────────────────────────────────────
    # Persistance
    # ─────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Sauvegarde le modèle (LSTM en format Keras + Prophet en pkl)."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_path = path or (self._model_dir / f"{self.MODEL_NAME}_{timestamp}")
        base_path = Path(str(base_path).replace(".pkl", ""))
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarde LSTM (format natif Keras)
        if self._lstm_model:
            keras_path = Path(f"{base_path}.keras")
            self._lstm_model.save(str(keras_path))
            logger.info("LSTM sauvegardé → {}", keras_path)

        # Sauvegarde Prophet + scalers + méta
        pkl_path = Path(f"{base_path}.pkl")
        payload = {
            "prophet_models":  self._prophet_models,
            "scaler_X":        self._scaler_X,
            "scaler_y":        self._scaler_y,
            "trained_at":      self._trained_at,
            "metrics":         self._metrics,
            "model_version":   self.MODEL_VERSION,
            "lstm_path":       str(base_path) + ".keras",
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("WeatherForecaster sauvegardé → {}", pkl_path)
        return pkl_path

    @classmethod
    def load(cls, path: Path) -> "WeatherForecaster":
        """Charge un WeatherForecaster depuis les fichiers pkl + keras."""
        path = Path(str(path).replace(".keras", "").replace(".pkl", ""))

        instance = cls()

        # Chargement pickle (Prophet + scalers)
        pkl_path = Path(f"{path}.pkl")
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                payload = pickle.load(f)
            instance._prophet_models = payload.get("prophet_models", {})
            instance._scaler_X       = payload.get("scaler_X")
            instance._scaler_y       = payload.get("scaler_y")
            instance._trained_at     = payload.get("trained_at")
            instance._metrics        = payload.get("metrics", {})

        # Chargement LSTM (Keras)
        keras_path = Path(f"{path}.keras")
        if keras_path.exists():
            try:
                import tensorflow as tf
                instance._lstm_model = tf.keras.models.load_model(str(keras_path))
                logger.info("LSTM chargé depuis {}", keras_path)
            except ImportError:
                logger.warning("TensorFlow non disponible — LSTM non chargé")
            except Exception as exc:
                logger.warning("Erreur chargement LSTM : {}", exc)

        logger.info(
            "WeatherForecaster chargé — LSTM: {} | Prophet: {}",
            instance._lstm_model is not None,
            list(instance._prophet_models.keys()),
        )
        return instance

    @classmethod
    def load_latest(cls) -> Optional["WeatherForecaster"]:
        """Charge le modèle WeatherForecaster le plus récent."""
        model_dir = Path(settings.ml.model_dir)
        candidates = sorted(
            model_dir.glob(f"{cls.MODEL_NAME}_*.pkl"), reverse=True
        )
        if not candidates:
            logger.warning("Aucun WeatherForecaster trouvé — mode Prophet only")
            instance = cls()  # Instance vide → utilise Prophet à la demande
            return instance

        try:
            # Reconstruit le chemin sans extension
            base = str(candidates[0]).replace(".pkl", "")
            return cls.load(Path(base))
        except Exception as exc:
            logger.error("Erreur load WeatherForecaster : {}", exc)
            return cls()

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _classifier_anomalie(
        variable: str,
        z_score: float,
        valeur: float,
    ) -> Tuple[str, Dict[str, str]]:
        """Classifie une anomalie et estime son impact sur paludisme/nutrition."""
        if variable == "temperature_moy_c":
            if z_score > 0:
                type_a = "chaleur_extreme"
                impact = {
                    "paludisme": "Favorise développement Plasmodium si T > 20°C",
                    "nutrition": "Stress hydrique des cultures, baisse rendements",
                }
            else:
                type_a = "froid_extreme"
                impact = {
                    "paludisme": "Inhibe développement Plasmodium si T < 16°C",
                    "nutrition": "Gel possible sur hautes terres, pertes agricoles",
                }
        else:  # precipitations
            if z_score > 0:
                type_a = "inondation"
                impact = {
                    "paludisme": "Multiplication gîtes larvaires, risque épidémique",
                    "nutrition": "Pertes récoltes, rupture accès marchés",
                }
            else:
                type_a = "secheresse"
                impact = {
                    "paludisme": "Assèchement gîtes larvaires, transmission réduite",
                    "nutrition": "Déficit hydrique cultures, soudure anticipée",
                }

        return type_a, impact