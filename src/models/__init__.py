"""
Package modèles ML — Paludisme, Nutrition, Météo, Explicabilité.
"""

from src.models.base_model import BasePredictor, ModelRegistry, PredictionResult
from src.models.malaria_predictor import MalariaPredictor
from src.models.nutrition_predictor import NutritionPredictor
from src.models.weather_forecaster import WeatherForecaster

__all__ = [
    "BasePredictor",
    "ModelRegistry",
    "PredictionResult",
    "MalariaPredictor",
    "NutritionPredictor",
    "WeatherForecaster",
]