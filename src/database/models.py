"""
Modèles SQLAlchemy ORM — toutes les tables PostgreSQL du projet.

Tables :
  Météo      : weather_observations, geo_ndvi, geo_wetlands
  Paludisme  : malaria_cases, epidemio_alerts
  Nutrition  : nutrition_status, nutrition_food_security, food_prices,
               humanitarian_stocks, nutrition_alerts, recipes, recipe_ingredients
  ML         : ml_predictions, model_registry
  Système    : users, audit_log

Conventions :
  - Toutes les tables ont created_at / updated_at (TimestampMixin)
  - UUIDs pour les identifiants des alertes, rapports
  - PostGIS via geoalchemy2 pour les champs géométriques
  - JSONB pour les payloads semi-structurés (features ML, métadonnées)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Float, ForeignKey,
    Index, Integer, Numeric, SmallInteger, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from src.database import Base
from geoalchemy2 import Geometry


# ─────────────────────────────────────────────────────────────────
# Mixins
# ─────────────────────────────────────────────────────────────────

class TimestampMixin:
    """Ajoute created_at et updated_at à toutes les tables."""
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ─────────────────────────────────────────────────────────────────
# Météo
# ─────────────────────────────────────────────────────────────────

class WeatherObservation(TimestampMixin, Base):
    __tablename__ = "weather_observations"
    __table_args__ = (
        Index("ix_weather_region_date", "region_id", "horodatage"),
    )

    id                    = Column(BigInteger, primary_key=True, autoincrement=True)
    region_id             = Column(String(20), nullable=False)   # ← aligné fetcher
    horodatage            = Column(DateTime(timezone=True), nullable=False)  # ← aligné
    temperature_c         = Column(Float)
    temperature_min_c     = Column(Float)
    temperature_max_c     = Column(Float)
    humidite_pct          = Column(Float)
    precipitations_mm     = Column(Float, default=0.0)   # ← aligné fetcher
    vent_kmh              = Column(Float)
    pression_hpa          = Column(Float)
    couverture_nuageuse_pct = Column(Float)
    rayonnement_solaire_mj  = Column(Float)
    humidite_sol_fraction   = Column(Float)
    indice_uv             = Column(Float)
    description           = Column(String(200))
    source                = Column(String(50), default="OpenWeatherMap")
    raw_json              = Column(JSONB)

class GeoNDVI(TimestampMixin, Base):
    """Indice NDVI satellite par région (Sentinel-2)."""
    __tablename__ = "geo_ndvi"
    __table_args__ = (
        UniqueConstraint("region_id", "observation_date", name="uq_ndvi_region_date"),
        Index("ix_ndvi_region_date", "region_id", "observation_date"),
    )

    id               = Column(Integer, primary_key=True, autoincrement=True)
    region_id        = Column(String(20), nullable=False)
    observation_date = Column(Date, nullable=False)
    ndvi_mean        = Column(Float, nullable=False)
    ndvi_std         = Column(Float)
    cloud_cover_pct  = Column(Float)
    satellite        = Column(String(20), default="Sentinel-2")
    source           = Column(String(50))


class GeoWetlands(Base):
    """Couches zones humides (shapefiles importés)."""
    __tablename__ = "geo_wetlands"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    region_id = Column(String(20), nullable=False)
    geom      = Column(Geometry(geometry_type="MULTIPOLYGON", srid=4326))
    type_zone = Column(String(50))  # mare, riziere, mangrove, cours_eau
    superficie_ha = Column(Float)
    source    = Column(String(100))


class Region(Base):
    """Table des 22 régions de Madagascar (géométries PostGIS)."""
    __tablename__ = "regions"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    region_id   = Column(String(20), unique=True, nullable=False)
    name        = Column(String(100), nullable=False)
    chef_lieu   = Column(String(100))
    geom        = Column(Geometry(geometry_type="MULTIPOLYGON", srid=4326))
    latitude    = Column(Float)
    longitude   = Column(Float)
    altitude_mean_m = Column(Float)
    area_km2    = Column(Float)
    population_2023 = Column(Integer)
    climate_zone    = Column(String(50))
    malaria_endemicity = Column(String(20))
    metadata_json   = Column(JSONB)


# ─────────────────────────────────────────────────────────────────
# Paludisme
# ─────────────────────────────────────────────────────────────────

class MalariaCase(TimestampMixin, Base):
    """
    Cas confirmés de paludisme hebdomadaires par région.
    Source : DHIS2 Ministère de la Santé Madagascar, WHO GHO.
    """
    __tablename__ = "malaria_observations"
    __table_args__ = (
        UniqueConstraint(
            "region_id", "annee", "semaine_epidemio",
            name="uq_malaria_region_week"
        ),
        Index("ix_malaria_region_date", "region_id", "date_rapport"),
        Index("ix_malaria_semaine", "annee", "semaine_epidemio"),
    )

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    region_id        = Column(String(20), nullable=False)
    district         = Column(String(100))
    annee            = Column(SmallInteger, nullable=False)
    semaine_epidemio = Column(SmallInteger, nullable=False)
    date_rapport     = Column(Date, nullable=False)

    # Comptages
    cas_confirmes    = Column(Integer, default=0)
    cas_confirmes_mixte     = Column(Integer, default=0)
    deces            = Column(Integer, default=0)
    hospitalisations = Column(Integer, default=0)
    tests_malaria    = Column(Integer, default=0)
    tdr_positifs     = Column(Integer, default=0)
    tdr_negatifs = Column(Integer, default=0)

    # Taux calculés
    taux_incidence_pour_mille  = Column(Numeric(10, 4), default=0)
    taux_positivite_tdr_pct    = Column(Numeric(5, 2), default=0)
    taux_letalite_pct          = Column(Numeric(5, 3), default=0)

    # Contexte
    population_a_risque = Column(Integer)
    source              = Column(String(50), default="DHIS2")
    fiabilite_donnees   = Column(String(20), default="confirmée")
    period_dhis2        = Column(String(10))  # ex: "2024W05"
    raw_json            = Column(JSONB)

    def __repr__(self) -> str:
        return (
            f"<MalariaCase region={self.region_id} "
            f"S{self.semaine_epidemio}-{self.annee} cas={self.cas_confirmes}>"
        )


class EpidemioAlert(TimestampMixin, Base):
    """
    Alertes épidémiologiques (paludisme et nutrition).
    Générées automatiquement par le système de détection.
    """
    __tablename__ = "epidemio_alerts"
    __table_args__ = (
        Index("ix_alert_region_statut", "region_id", "statut"),
        Index("ix_alert_severite", "severite"),
        Index("ix_alert_date", "date_detection"),
    )

    alerte_id    = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id    = Column(String(20), nullable=False)
    region_name  = Column(String(100))
    type_alerte  = Column(String(50), nullable=False)
    severite     = Column(String(20), nullable=False)  # surveillance|alerte|urgence|crise
    domaine      = Column(String(20), default="paludisme")  # paludisme|nutrition

    # Valeurs déclenchantes
    seuil_depasse    = Column(Float)
    valeur_actuelle  = Column(Float)
    indicateur_declencheur = Column(String(100))

    # Statut et suivi
    date_detection   = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    statut           = Column(String(30), default="active")
    date_resolution  = Column(DateTime(timezone=True))
    acquittee_par    = Column(Integer)  # user_id
    acquittee_le     = Column(DateTime(timezone=True))
    commentaire_acquittement = Column(Text)

    # Données enrichies
    description      = Column(Text)
    actions_requises = Column(JSONB)  # Liste de strings
    responsable_notification = Column(String(200))
    population_affectee = Column(Integer)
    enfants_a_risque    = Column(Integer)

    def __repr__(self) -> str:
        return f"<Alert {self.alerte_id[:8]} region={self.region_id} sev={self.severite}>"


# ─────────────────────────────────────────────────────────────────
# Nutrition
# ─────────────────────────────────────────────────────────────────

class NutritionStatus(TimestampMixin, Base):
    """
    Statut nutritionnel par région (GAM, SAM, MAM, Stunting).
    Source : Enquêtes SMART, MICS, DHIS2.
    """
    __tablename__ = "nutrition_status"
    __table_args__ = (
        UniqueConstraint("region_id", "date_observation", name="uq_nutrition_region_date"),
        Index("ix_nutrition_region_date", "region_id", "date_observation"),
    )

    id                   = Column(BigInteger, primary_key=True, autoincrement=True)
    region_id            = Column(String(20), nullable=False)
    region_name          = Column(String(100))
    date_observation     = Column(Date, nullable=False)
    date_enquete         = Column(Date)

    # Indicateurs anthropométriques (%)
    gam_pct              = Column(Numeric(5, 2))   # Global Acute Malnutrition
    sam_pct              = Column(Numeric(5, 2))   # Severe Acute Malnutrition
    mam_pct              = Column(Numeric(5, 2))   # Moderate Acute Malnutrition
    stunting_pct         = Column(Numeric(5, 2))   # Retard de croissance
    underweight_pct      = Column(Numeric(5, 2))   # Insuffisance pondérale

    # Population affectée
    enfants_5ans_affectes       = Column(Integer)
    femmes_enceintes_malnutries  = Column(Integer)

    # Classification et tendance
    classification_who   = Column(String(20))  # acceptable|alerte|urgence|crise
    tendance_vs_periode_prec = Column(String(20))  # amélioration|stable|dégradation
    fiabilite_donnees    = Column(String(20), default="estimée")
    source               = Column(String(100))
    raw_json             = Column(JSONB)


class NutritionFoodSecurity(TimestampMixin, Base):
    """
    Sécurité alimentaire : FCS, HDDS, rCSI, disponibilités.
    Source : WFP VAM, enquêtes ménages.
    """
    __tablename__ = "nutrition_food_security"
    __table_args__ = (
        UniqueConstraint("region_id", "date_observation", name="uq_foodsec_region_date"),
        Index("ix_foodsec_region_date", "region_id", "date_observation"),
    )

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    region_id        = Column(String(20), nullable=False)
    date_observation = Column(Date, nullable=False)

    # Scores sécurité alimentaire
    score_fcs        = Column(Numeric(6, 2))  # Food Consumption Score (0-112)
    classification_fcs = Column(String(20))   # pauvre|limite|acceptable
    hdds             = Column(Numeric(4, 1))  # Dietary Diversity Score (0-12)
    rcsi             = Column(Numeric(5, 1))  # Reduced Coping Strategies Index

    # Disponibilités par groupe (0-3)
    disponibilite_cereales          = Column(SmallInteger)
    disponibilite_legumineuses      = Column(SmallInteger)
    disponibilite_proteines_animales= Column(SmallInteger)
    disponibilite_legumes           = Column(SmallInteger)
    disponibilite_fruits            = Column(SmallInteger)

    source           = Column(String(50), default="WFP VAM")


class FoodPrice(TimestampMixin, Base):
    """
    Prix des denrées alimentaires par région.
    Source : WFP VAM, marchés locaux.
    """
    __tablename__ = "food_prices"
    __table_args__ = (
        UniqueConstraint("region_id", "date_observation", name="uq_price_region_date"),
        Index("ix_price_region_date", "region_id", "date_observation"),
    )

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    region_id        = Column(String(20), nullable=False)
    date_observation = Column(Date, nullable=False)

    # Prix en MGA (Ariary malgache)
    prix_riz_kg        = Column(Numeric(10, 2))
    prix_manioc_kg     = Column(Numeric(10, 2))
    prix_mais_kg       = Column(Numeric(10, 2))
    prix_haricots_kg   = Column(Numeric(10, 2))
    prix_huile_litre   = Column(Numeric(10, 2))
    prix_sucre_kg      = Column(Numeric(10, 2))

    # Variations
    variation_prix_pct_1m  = Column(Numeric(6, 2))
    variation_prix_pct_3m  = Column(Numeric(6, 2))

    devise             = Column(String(5), default="MGA")
    source             = Column(String(50), default="WFP VAM")


class HumanitarianStock(TimestampMixin, Base):
    """
    Stocks humanitaires nutritionnels (RUTF, RUSF, micronutriments).
    """
    __tablename__ = "humanitarian_stocks"
    __table_args__ = (
        Index("ix_stocks_region_date", "region_id", "date_inventaire"),
    )

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    region_id        = Column(String(20), nullable=False)
    date_inventaire  = Column(Date, nullable=False)

    # Stocks thérapeutiques
    rutf_sachets         = Column(Integer, default=0)
    rusf_sachets         = Column(Integer, default=0)
    plumpy_nut_sachets   = Column(Integer, default=0)

    # Micronutriments
    spiruline_kg         = Column(Numeric(8, 2), default=0)
    sel_iode_kg          = Column(Numeric(8, 2), default=0)
    vitamine_a_capsules  = Column(Integer, default=0)
    fer_folate_comprimes = Column(Integer, default=0)
    zinc_comprimes       = Column(Integer, default=0)

    # Couverture calculée
    jours_couverture_sam = Column(Numeric(6, 1), default=0)
    jours_couverture_mam = Column(Numeric(6, 1), default=0)
    statut_stock         = Column(String(30))  # adéquat|alerte|rupture_imminente|rupture

    # Logistique
    derniere_livraison         = Column(Date)
    prochaine_livraison_prevue = Column(Date)
    inventaire_par             = Column(String(100))
    notes                      = Column(Text)


class Recipe(TimestampMixin, Base):
    """
    Base de données des recettes nutritionnelles adaptées Madagascar.
    """
    __tablename__ = "recipes"
    __table_args__ = (
        Index("ix_recipe_score", "score_nutritionnel"),
        Index("ix_recipe_region", "regions_adaptees"),
    )

    recette_id           = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    nom                  = Column(String(200), nullable=False)
    nom_malgache         = Column(String(200))

    # Contexte
    regions_adaptees     = Column(JSONB)   # Liste region_ids
    saison               = Column(JSONB)   # ["saison_pluies", "saison_seche", ...]
    cible                = Column(JSONB)   # ["enfants_6_23m", "femmes_enceintes", ...]

    # Valeurs nutritionnelles par portion
    calories_kcal        = Column(Numeric(7, 1))
    proteines_g          = Column(Numeric(6, 2))
    glucides_g           = Column(Numeric(6, 2))
    lipides_g            = Column(Numeric(6, 2))
    fer_mg               = Column(Numeric(6, 2))
    vitamine_a_ug        = Column(Numeric(6, 2))
    zinc_mg              = Column(Numeric(6, 2))
    score_nutritionnel   = Column(Numeric(5, 1))  # 0-100

    # Recette
    ingredients          = Column(JSONB)   # [{nom, quantite_g, disponible_localement}]
    instructions         = Column(Text)
    temps_preparation_min = Column(Integer)
    cout_estime_ariary   = Column(Numeric(10, 2))
    image_url            = Column(String(500))
    actif                = Column(Boolean, default=True)

    def __repr__(self) -> str:
        return f"<Recipe {self.recette_id[:8]} nom={self.nom}>"


# ─────────────────────────────────────────────────────────────────
# ML — Prédictions et modèles
# ─────────────────────────────────────────────────────────────────

class MLPrediction(TimestampMixin, Base):
    """
    Historique des prédictions ML (pour backtesting et audit).
    """
    __tablename__ = "ml_predictions"
    __table_args__ = (
        Index("ix_pred_region_date", "region_id", "date_prediction"),
        Index("ix_pred_modele", "modele_nom"),
    )

    id                   = Column(BigInteger, primary_key=True, autoincrement=True)
    region_id            = Column(String(20), nullable=False)
    modele_nom           = Column(String(50), nullable=False)
    modele_version       = Column(String(20))
    date_prediction      = Column(DateTime(timezone=True), nullable=False)
    horizon_jours        = Column(SmallInteger, nullable=False)

    # Scores
    score_paludisme      = Column(Numeric(6, 4))
    score_nutrition      = Column(Numeric(6, 4))
    score_composite      = Column(Numeric(6, 4))
    niveau_alerte_global = Column(String(20))

    # Valeurs réelles (remplies après observation — backtesting)
    valeur_reelle        = Column(Numeric(6, 4))
    erreur_prediction    = Column(Numeric(8, 4))

    payload_json         = Column(JSONB)   # Prédiction complète sérialisée
    features_json        = Column(JSONB)   # Features utilisées (pour audit)


# ─────────────────────────────────────────────────────────────────
# Système — Utilisateurs et audit
# ─────────────────────────────────────────────────────────────────

class User(TimestampMixin, Base):
    """Utilisateurs de l'API  ."""
    __tablename__ = "users"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    username     = Column(String(50), unique=True, nullable=False)
    email        = Column(String(200), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role         = Column(String(20), default="viewer")  # admin|national|regional|viewer
    region_id    = Column(String(20))  # Restriction région pour rôle REGIONAL
    is_active    = Column(Boolean, default=True)
    last_login   = Column(DateTime(timezone=True))
    full_name    = Column(String(200))
    organisation = Column(String(200))

    def __repr__(self) -> str:
        return f"<User {self.username} role={self.role}>"


class AuditLog(Base):
    """Journal d'audit des actions sensibles."""
    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_user", "user_id"),
        Index("ix_audit_action", "action"),
        Index("ix_audit_date", "timestamp"),
    )

    id         = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp  = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    user_id    = Column(Integer, ForeignKey("users.id"))
    username   = Column(String(50))
    action     = Column(String(100), nullable=False)
    resource   = Column(String(200))
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    payload    = Column(JSONB)
    status_code = Column(SmallInteger)