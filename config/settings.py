"""
config/settings.py
Configuration centralisée avec Pydantic Settings v2.
Charge les variables depuis .env et valide les types automatiquement.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Répertoire racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/malaria_db",
        alias="DATABASE_URL",
    )
    sync_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/malaria_db",
        alias="DATABASE_SYNC_URL",
    )
    pool_size: int = Field(default=20, alias="DB_POOL_SIZE")
    max_overflow: int = Field(default=40, alias="DB_MAX_OVERFLOW")
    echo: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RedisSettings(BaseSettings):
    url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    cache_db: int = Field(default=1, alias="REDIS_CACHE_DB")
    celery_db: int = Field(default=2, alias="REDIS_CELERY_DB")
    cache_ttl_predictions: int = Field(default=86400, alias="CACHE_TTL_PREDICTIONS")
    cache_ttl_weather: int = Field(default=3600, alias="CACHE_TTL_WEATHER")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class WeatherAPISettings(BaseSettings):
    openweather_api_key: str = Field(default="", alias="OPENWEATHER_API_KEY")
    openweather_base_url: str = Field(
        default="https://api.openweathermap.org/data/2.5",
        alias="OPENWEATHER_BASE_URL",
    )
    copernicus_api_key: str = Field(default="", alias="COPERNICUS_API_KEY")
    nasa_power_base_url: str = Field(
        default="https://power.larc.nasa.gov/api/temporal/daily/point",
        alias="NASA_POWER_BASE_URL",
    )
    sentinel_hub_client_id: str = Field(default="", alias="SENTINEL_HUB_CLIENT_ID")
    sentinel_hub_client_secret: str = Field(
        default="", alias="SENTINEL_HUB_CLIENT_SECRET"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class HealthAPISettings(BaseSettings):
    dhis2_base_url: str = Field(
        default="https://dhis.moh.gov.mg/api", alias="DHIS2_BASE_URL"
    )
    dhis2_username: str = Field(default="", alias="DHIS2_USERNAME")
    dhis2_password: str = Field(default="", alias="DHIS2_PASSWORD")
    who_gho_base_url: str = Field(
        default="https://ghoapi.azureedge.net/api", alias="WHO_GHO_BASE_URL"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class NutritionAPISettings(BaseSettings):
    fao_base_url: str = Field(
        default="http://www.fao.org/faostat/api/v1", alias="FAO_API_BASE_URL"
    )
    wfp_base_url: str = Field(
        default="https://api.wfp.org/vam-data-bridges/2.0", alias="WFP_API_BASE_URL"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class MLSettings(BaseSettings):
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="malaria-nutrition-madagascar", alias="MLFLOW_EXPERIMENT_NAME"
    )
    model_dir: Path = ROOT_DIR / "data" / "models"
    retrain_frequency_days: int = 30
    drift_threshold: float = 0.15  # PSI threshold for retraining trigger
    malaria_risk_thresholds: Dict[str, float] = {
        "faible": 0.25,
        "moyen": 0.50,
        "eleve": 0.75,
        "tres_eleve": 1.0,
    }
    nutrition_risk_thresholds: Dict[str, float] = {
        "acceptable": 5.0,   # GAM < 5%
        "alerte": 10.0,      # GAM 5-10%
        "urgence": 15.0,     # GAM 10-15%
        "urgence_critique": 100.0,  # GAM > 15%
    }

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class MinIOSettings(BaseSettings):
    endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    bucket_reports: str = Field(default="unicef-reports", alias="MINIO_BUCKET_REPORTS")
    bucket_models: str = Field(default="ml-models", alias="MINIO_BUCKET_MODELS")
    secure: bool = Field(default=False, alias="MINIO_SECURE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class JWTSettings(BaseSettings):
    secret_key: str = Field(default="change-me-in-production", alias="JWT_SECRET_KEY")
    algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=60, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    refresh_token_expire_days: int = Field(
        default=7, alias="JWT_REFRESH_TOKEN_EXPIRE_DAYS"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """Configuration principale de l'application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Malaria-Nutrition Predictor", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    app_secret_key: str = Field(default="change-me", alias="APP_SECRET_KEY")
    api_version: str = Field(default="v1", alias="API_VERSION")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=4, alias="WORKERS")

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/2", alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/3", alias="CELERY_RESULT_BACKEND"
    )
    celery_timezone: str = Field(
        default="Indian/Antananarivo", alias="CELERY_TIMEZONE"
    )

    # Rapports
    report_output_dir: Path = Field(
        default=Path("/tmp/reports"), alias="REPORT_OUTPUT_DIR"
    )
    report_language: str = Field(default="fr", alias="REPORT_LANGUAGE")
    weekly_report_day: int = Field(default=0, alias="WEEKLY_REPORT_DAY")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="/var/log/malaria-predictor/app.log", alias="LOG_FILE")

    # Régions Madagascar
    madagascar_regions_count: int = Field(default=22, alias="MADAGASCAR_REGIONS_COUNT")
    default_region: str = Field(default="Analamanga", alias="DEFAULT_REGION")

    # Sous-configurations (lazy init)
    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings()

    @property
    def redis(self) -> RedisSettings:
        return RedisSettings()

    @property
    def weather_api(self) -> WeatherAPISettings:
        return WeatherAPISettings()

    @property
    def health_api(self) -> HealthAPISettings:
        return HealthAPISettings()

    @property
    def nutrition_api(self) -> NutritionAPISettings:
        return NutritionAPISettings()

    @property
    def ml(self) -> MLSettings:
        return MLSettings()

    @property
    def minio(self) -> MinIOSettings:
        return MinIOSettings()

    @property
    def jwt(self) -> JWTSettings:
        return JWTSettings()

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"APP_ENV must be one of {allowed}")
        return v

    @field_validator("report_language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        allowed = {"fr", "mg"}
        if v not in allowed:
            raise ValueError(f"REPORT_LANGUAGE must be one of {allowed}")
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def allowed_origins(self) -> List[str]:
        if self.is_production:
            return [
                "https://unicef-dashboard.mg",
                "https://api.unicef-madagascar.org",
            ]
        return ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retourne l'instance singleton des settings (cachée)."""
    return Settings()


# Instance globale
settings = get_settings()