"""
Package de collecte de données temps réel.
Couvre : météo, épidémiologie paludisme, nutrition/alimentation.
"""

from src.data_collection.weather_fetcher import WeatherFetcher
from src.data_collection.malaria_fetcher import MalariaFetcher
from src.data_collection.nutrition_fetcher import NutritionFetcher

__all__ = ["WeatherFetcher", "MalariaFetcher", "NutritionFetcher"]