"""
Validateurs pour l'ensemble du projet.

Ce module centralise TOUTE la logique de validation :
  1. Modèles Pydantic (entrées API, payloads)
  2. Fonctions de validation métier standalone (utilisables sans Pydantic)
  3. Validators de qualité données (pipelines preprocessing)

Exports principaux :
  ── Pydantic Schemas ──────────────────────────────────────────────
  CoordGPS                   → (lat, lon) validés dans Madagascar
  WeatherDataInput           → données météo brutes d'une API
  MalariaObservationInput    → cas paludisme signalés (DHIS2 / terrain)
  NutritionDataInput         → données nutrition (GAM, MAM, anthropométrie)
  PredictionRequest          → requête de prédiction ML via API
  RegionFilterParams         → paramètres de filtrage régional (query params)
  AlerteInput                → création d'une alerte épidémiologique
  RecipeRequest              → requête de génération de recette

  ── Validators Standalone ─────────────────────────────────────────
  validate_region_code(code) → bool + log
  validate_score(score)      → float clampé ou ValidationError
  validate_gam_rate(rate)    → float validé
  validate_date_range(start, end, max_days) → bool
  validate_weather_payload(data) → WeatherDataInput | None

  ── Qualité Données ───────────────────────────────────────────────
  check_missing_rate(series, threshold) → bool (True si acceptable)
  check_value_bounds(value, low, high)  → bool
  DataQualityReport                     → dataclass rapport de qualité
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)

from src.utils.constants import (
    REGIONS_MADAGASCAR,
    SEUILS_RISQUE_MALARIA,
    SEUILS_GAM,
    NiveauRisqueMalaria,
    ClassificationNutritionOMS,
    TypeRapportEnum,
    GroupeCibleNutrition,
)
from src.utils.logger import get_logger

log = get_logger("validators")


# ─────────────────────────────────────────────────────────────────
# Constantes de validation
# ─────────────────────────────────────────────────────────────────

# Bornes physiques raisonnables pour Madagascar
_TEMP_MIN_C        = -5.0    # Gel exceptionnel hauts plateaux
_TEMP_MAX_C        = 45.0    # Canicule côte ouest
_HUMIDITY_MIN      = 0.0
_HUMIDITY_MAX      = 100.0
_PRECIP_MIN_MM     = 0.0
_PRECIP_MAX_MM_DAY = 500.0   # Cyclone intense : jusqu'à 450 mm/24h
_WIND_MIN_KMS      = 0.0
_WIND_MAX_KMS      = 300.0   # Cyclone catégorie 5
_NDVI_MIN          = -1.0
_NDVI_MAX          = 1.0
_ALTITUDE_MIN_M    = 0.0
_ALTITUDE_MAX_M    = 2_876.0  # Pic Boby — point culminant Madagascar

# Âge anthropométrie
_AGE_MOIS_MIN = 0
_AGE_MOIS_MAX = 59  # Enquêtes UNICEF : 0–59 mois

# Score de risque
_SCORE_MIN = 0.0
_SCORE_MAX = 1.0

# Taux GAM plausible (%)
_GAM_MIN =  0.0
_GAM_MAX = 60.0  # Seuil de crise absolue — au-delà, donnée suspecte

# Nombre de cas paludisme plausible par semaine par district
_CAS_MALARIA_MAX_SEMAINE = 50_000

# Horizon de prédiction
_HORIZON_MIN_JOURS = 1
_HORIZON_MAX_JOURS = 90

# Date minimale acceptable (début des données historiques)
_DATE_MIN = date(2010, 1, 1)

# Regex pour identifiants externes
_DHIS2_ID_REGEX = re.compile(r"^[a-zA-Z0-9]{11}$")


# ═════════════════════════════════════════════════════════════════
# 1. PYDANTIC CONFIG PARTAGÉE
# ═════════════════════════════════════════════════════════════════

class _BaseSchema(BaseModel):
    """Config commune à tous les modèles Pydantic du projet."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        populate_by_name=True,
        frozen=False,
    )


# ═════════════════════════════════════════════════════════════════
# 2. SCHEMAS GÉOGRAPHIQUES
# ═════════════════════════════════════════════════════════════════

class CoordGPS(_BaseSchema):
    """
    Coordonnées GPS validées à l'intérieur du bounding box Madagascar.

    Utilisé dans :
        - Requêtes géolocalisées de l'API
        - Enrichissement des données terrain (agents UNICEF)
    """
    latitude:  float = Field(..., ge=-25.61, le=-11.95, description="Latitude (degrés décimaux)")
    longitude: float = Field(..., ge=43.22,  le=50.48,  description="Longitude (degrés décimaux)")

    @property
    def as_tuple(self) -> Tuple[float, float]:
        return (self.latitude, self.longitude)

    @property
    def as_wkt(self) -> str:
        """Format WKT pour PostGIS."""
        return f"POINT({self.longitude} {self.latitude})"


class RegionFilterParams(_BaseSchema):
    """
    Paramètres de filtrage régional (Query Params FastAPI).

    Example endpoint :
        GET /api/v1/predictions?region_code=MDG-ANA&date_debut=2024-01-01
    """
    region_code: Optional[str] = Field(
        None,
        description="Code région MDG-xxx (si absent : toutes les régions)",
        examples=["MDG-ANA", "MDG-BOE"],
    )
    date_debut: Optional[date] = Field(
        None,
        description="Début de la période (inclusif)",
    )
    date_fin: Optional[date] = Field(
        None,
        description="Fin de la période (inclusif)",
    )
    limit: int = Field(100, ge=1, le=1000, description="Nombre max de résultats")
    offset: int = Field(0,   ge=0,        description="Offset pagination")

    @field_validator("region_code")
    @classmethod
    def valider_code_region(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in REGIONS_MADAGASCAR:
            raise ValueError(
                f"Code région invalide : '{v}'. "
                f"Codes valides : {REGIONS_MADAGASCAR}"
            )
        return v

    @model_validator(mode="after")
    def valider_plage_dates(self) -> "RegionFilterParams":
        if self.date_debut and self.date_fin:
            if self.date_debut > self.date_fin:
                raise ValueError(
                    "date_debut doit être antérieure ou égale à date_fin"
                )
            delta = (self.date_fin - self.date_debut).days
            if delta > 366:
                raise ValueError(
                    f"La plage de dates ne peut excéder 366 jours (reçu : {delta} jours). "
                    "Utilisez l'endpoint /reports pour des agrégations longues."
                )
        return self


# ═════════════════════════════════════════════════════════════════
# 3. SCHEMAS MÉTÉO
# ═════════════════════════════════════════════════════════════════

class WeatherDataInput(_BaseSchema):
    """
    Données météorologiques brutes reçues d'une API externe (OpenWeatherMap, NASA POWER…).

    Toutes les valeurs sont validées dans des plages physiquement réalistes pour Madagascar.
    Les champs optionnels correspondent à des capteurs qui peuvent être absents dans les
    régions à faible couverture (ex: vitesse vent, NDVI).
    """
    region_code:          str   = Field(..., description="Code région MDG-xxx")
    timestamp:            datetime = Field(..., description="Horodatage UTC de la mesure")
    temperature_c:        float = Field(...,  description="Température air (°C)")
    temperature_min_c:    float = Field(...,  description="Température min journalière (°C)")
    temperature_max_c:    float = Field(...,  description="Température max journalière (°C)")
    humidite_pct:         float = Field(...,  description="Humidité relative (%)")
    precipitation_mm:     float = Field(0.0, description="Précipitations sur 24h (mm)")
    vitesse_vent_kmh:     Optional[float] = Field(None, description="Vitesse vent (km/h)")
    ndvi:                 Optional[float] = Field(None, description="Indice végétation NDVI [-1,1]")
    altitude_m:           Optional[float] = Field(None, description="Altitude de la station (m)")
    source_api:           str   = Field("unknown", description="Identifiant source (owm, nasa, copernicus…)")

    # ── Validators champs individuels ──────────────────────────────

    @field_validator("region_code")
    @classmethod
    def valider_region(cls, v: str) -> str:
        if v not in REGIONS_MADAGASCAR:
            raise ValueError(f"Code région invalide : '{v}'")
        return v

    @field_validator("temperature_c", "temperature_min_c", "temperature_max_c")
    @classmethod
    def valider_temperature(cls, v: float) -> float:
        if not (_TEMP_MIN_C <= v <= _TEMP_MAX_C):
            raise ValueError(
                f"Température {v}°C hors plage réaliste [{_TEMP_MIN_C}, {_TEMP_MAX_C}]"
            )
        return round(v, 2)

    @field_validator("humidite_pct")
    @classmethod
    def valider_humidite(cls, v: float) -> float:
        if not (_HUMIDITY_MIN <= v <= _HUMIDITY_MAX):
            raise ValueError(f"Humidité {v}% hors plage [0, 100]")
        return round(v, 1)

    @field_validator("precipitation_mm")
    @classmethod
    def valider_precipitation(cls, v: float) -> float:
        if not (_PRECIP_MIN_MM <= v <= _PRECIP_MAX_MM_DAY):
            raise ValueError(
                f"Précipitation {v} mm/24h hors plage plausible "
                f"[{_PRECIP_MIN_MM}, {_PRECIP_MAX_MM_DAY}]"
            )
        return round(v, 2)

    @field_validator("vitesse_vent_kmh")
    @classmethod
    def valider_vent(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (_WIND_MIN_KMS <= v <= _WIND_MAX_KMS):
            raise ValueError(f"Vitesse vent {v} km/h hors plage [{_WIND_MIN_KMS}, {_WIND_MAX_KMS}]")
        return round(v, 1) if v is not None else None

    @field_validator("ndvi")
    @classmethod
    def valider_ndvi(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (_NDVI_MIN <= v <= _NDVI_MAX):
            raise ValueError(f"NDVI {v} hors plage [-1, 1]")
        return round(v, 4) if v is not None else None

    @field_validator("altitude_m")
    @classmethod
    def valider_altitude(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (_ALTITUDE_MIN_M <= v <= _ALTITUDE_MAX_M):
            raise ValueError(
                f"Altitude {v} m hors plage Madagascar [{_ALTITUDE_MIN_M}, {_ALTITUDE_MAX_M}]"
            )
        return v

    @field_validator("timestamp")
    @classmethod
    def valider_timestamp(cls, v: datetime) -> datetime:
        if v.date() < _DATE_MIN:
            raise ValueError(
                f"Timestamp {v.date()} antérieur à la date minimale {_DATE_MIN}"
            )
        # Données futures autorisées jusqu'à 7 jours (prévisions)
        if v.date() > date.today() + timedelta(days=7):
            raise ValueError(
                f"Timestamp {v.date()} trop loin dans le futur (max 7 jours)"
            )
        return v

    # ── Validator cross-champs ──────────────────────────────────────

    @model_validator(mode="after")
    def valider_coherence_temperatures(self) -> "WeatherDataInput":
        if self.temperature_min_c > self.temperature_max_c:
            raise ValueError(
                f"Incohérence : temp_min ({self.temperature_min_c}°C) "
                f"> temp_max ({self.temperature_max_c}°C)"
            )
        if not (self.temperature_min_c <= self.temperature_c <= self.temperature_max_c):
            raise ValueError(
                f"Température moyenne {self.temperature_c}°C hors plage "
                f"[{self.temperature_min_c}, {self.temperature_max_c}]"
            )
        return self


# ═════════════════════════════════════════════════════════════════
# 4. SCHEMAS PALUDISME
# ═════════════════════════════════════════════════════════════════

class MalariaObservationInput(_BaseSchema):
    """
    Observation épidémiologique paludisme (DHIS2, rapports terrain, WHO).

    Représente les données hebdomadaires par région/district.
    """
    region_code:         str      = Field(..., description="Code région MDG-xxx")
    semaine_iso:         int      = Field(..., ge=1, le=53, description="Semaine ISO (1–53)")
    annee:               int      = Field(..., ge=2010, le=2035, description="Année")
    cas_confirmes:       int      = Field(..., ge=0, description="Cas confirmés (TDR ou microscopie)")
    cas_presumes:        Optional[int] = Field(None, ge=0, description="Cas présumés (clinique seul)")
    deces:               Optional[int] = Field(None, ge=0, description="Décès attribués au paludisme")
    tests_realises:      Optional[int] = Field(None, ge=0, description="Nombre de tests effectués")
    source:              str      = Field("dhis2", description="Source des données")
    dhis2_org_unit_id:   Optional[str] = Field(None, description="ID unité organisationnelle DHIS2")
    district:            Optional[str] = Field(None, description="Nom du district (subdivision)")

    @field_validator("region_code")
    @classmethod
    def valider_region(cls, v: str) -> str:
        if v not in REGIONS_MADAGASCAR:
            raise ValueError(f"Code région invalide : '{v}'")
        return v

    @field_validator("cas_confirmes")
    @classmethod
    def valider_cas_max(cls, v: int) -> int:
        if v > _CAS_MALARIA_MAX_SEMAINE:
            raise ValueError(
                f"Nombre de cas {v} dépasse le seuil plausible "
                f"({_CAS_MALARIA_MAX_SEMAINE}/semaine). "
                "Vérifier l'unité (total annuel saisi au lieu d'hebdomadaire ?)"
            )
        return v

    @field_validator("dhis2_org_unit_id")
    @classmethod
    def valider_dhis2_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _DHIS2_ID_REGEX.match(v):
            raise ValueError(
                f"ID DHIS2 invalide : '{v}'. Format attendu : 11 caractères alphanumériques"
            )
        return v

    @model_validator(mode="after")
    def valider_coherence_cas(self) -> "MalariaObservationInput":
        # Taux de positivité implicite
        if self.tests_realises is not None and self.tests_realises > 0:
            taux = self.cas_confirmes / self.tests_realises
            if taux > 1.0:
                raise ValueError(
                    f"Cas confirmés ({self.cas_confirmes}) > tests réalisés "
                    f"({self.tests_realises}) : impossible"
                )
            if taux > 0.95:
                log.warning(
                    "Taux positivité très élevé ({:.1%}) — région={} semaine={}",
                    taux, self.region_code, self.semaine_iso
                )

        if self.deces is not None and self.deces > self.cas_confirmes:
            raise ValueError(
                f"Décès ({self.deces}) > cas confirmés ({self.cas_confirmes}) : impossible"
            )
        return self

    @property
    def taux_positivite(self) -> Optional[float]:
        """Taux de positivité (0–1). None si tests non renseignés."""
        if self.tests_realises and self.tests_realises > 0:
            return round(self.cas_confirmes / self.tests_realises, 4)
        return None


# ═════════════════════════════════════════════════════════════════
# 5. SCHEMAS NUTRITION
# ═════════════════════════════════════════════════════════════════

class NutritionDataInput(_BaseSchema):
    """
    Données nutritionnelles issues d'enquêtes SMART / UNICEF / terrain.

    GAM  = Global Acute Malnutrition (Malnutrition Aiguë Globale)
    MAM  = Moderate Acute Malnutrition
    SAM  = Severe Acute Malnutrition
    Tous exprimés en % de la population cible (enfants 6–59 mois).
    """
    region_code:          str      = Field(..., description="Code région MDG-xxx")
    date_enquete:         date     = Field(..., description="Date de l'enquête")
    gam_pct:              float    = Field(..., ge=0.0, le=60.0, description="Taux GAM (%)")
    mam_pct:              Optional[float] = Field(None, ge=0.0, description="Taux MAM (%)")
    sam_pct:              Optional[float] = Field(None, ge=0.0, description="Taux SAM (%)")
    groupe_cible:         GroupeCibleNutrition = Field(
        GroupeCibleNutrition.ENFANTS_6_23M,
        description="Groupe de population ciblé",
    )
    n_enfants_enquetes:   Optional[int] = Field(
        None, ge=30,
        description="Taille échantillon (min 30 pour être représentatif)"
    )
    score_sca:            Optional[float] = Field(
        None, ge=0.0, le=112.0,
        description="Score de consommation alimentaire (SCA/FCS) — 0 à 112"
    )
    source:               str = Field("unicef_enquete", description="Origine des données")

    @field_validator("region_code")
    @classmethod
    def valider_region(cls, v: str) -> str:
        if v not in REGIONS_MADAGASCAR:
            raise ValueError(f"Code région invalide : '{v}'")
        return v

    @field_validator("date_enquete")
    @classmethod
    def valider_date(cls, v: date) -> date:
        if v < _DATE_MIN:
            raise ValueError(f"Date enquête {v} antérieure à {_DATE_MIN}")
        if v > date.today():
            raise ValueError(f"Date enquête {v} dans le futur")
        return v

    @model_validator(mode="after")
    def valider_coherence_taux(self) -> "NutritionDataInput":
        # MAM + SAM = GAM (approximation — légère divergence acceptée)
        if self.mam_pct is not None and self.sam_pct is not None:
            gam_calc = self.mam_pct + self.sam_pct
            if abs(gam_calc - self.gam_pct) > 3.0:
                raise ValueError(
                    f"Incohérence GAM : MAM({self.mam_pct}%) + SAM({self.sam_pct}%) "
                    f"= {gam_calc:.1f}% ≠ GAM déclaré {self.gam_pct}% (écart > 3%)"
                )
        if self.sam_pct is not None and self.gam_pct > 0:
            if self.sam_pct > self.gam_pct:
                raise ValueError(
                    f"SAM ({self.sam_pct}%) > GAM ({self.gam_pct}%) : impossible"
                )
        return self

    @property
    def classification_oms(self) -> ClassificationNutritionOMS:
        """Classification OMS selon le taux GAM."""
        if self.gam_pct < SEUILS_GAM["acceptable"]:
            return ClassificationNutritionOMS.ACCEPTABLE
        elif self.gam_pct < SEUILS_GAM["alerte"]:
            return ClassificationNutritionOMS.ALERTE
        elif self.gam_pct < SEUILS_GAM["urgence"]:
            return ClassificationNutritionOMS.URGENCE
        return ClassificationNutritionOMS.CRISE


# ═════════════════════════════════════════════════════════════════
# 6. SCHEMAS PRÉDICTION (API Endpoints)
# ═════════════════════════════════════════════════════════════════

class PredictionRequest(_BaseSchema):
    """
    Requête de prédiction ML via l'API FastAPI.

    Endpoint : POST /api/v1/predictions/
    """
    region_code:    str  = Field(..., description="Code région MDG-xxx")
    horizon_jours:  int  = Field(
        7,
        ge=_HORIZON_MIN_JOURS,
        le=_HORIZON_MAX_JOURS,
        description="Horizon de prédiction (1–90 jours)",
    )
    date_reference: Optional[date] = Field(
        None,
        description="Date de référence (défaut : aujourd'hui)",
    )
    inclure_voisins:    bool = Field(
        False,
        description="Inclure les prédictions des régions limitrophes",
    )
    inclure_shap:       bool = Field(
        False,
        description="Inclure les valeurs SHAP (explainabilité — plus lent)",
    )

    @field_validator("region_code")
    @classmethod
    def valider_region(cls, v: str) -> str:
        if v not in REGIONS_MADAGASCAR:
            raise ValueError(f"Code région invalide : '{v}'")
        return v

    @field_validator("date_reference")
    @classmethod
    def valider_date(cls, v: Optional[date]) -> Optional[date]:
        if v is not None and v < _DATE_MIN:
            raise ValueError(f"date_reference {v} trop ancienne (min: {_DATE_MIN})")
        return v

    @model_validator(mode="after")
    def set_date_defaut(self) -> "PredictionRequest":
        if self.date_reference is None:
            object.__setattr__(self, "date_reference", date.today())
        return self


class PredictionResponse(_BaseSchema):
    """Réponse d'une prédiction ML — structure standard de l'API."""
    region_code:       str
    date_prediction:   date
    horizon_jours:     int
    score_malaria:     float = Field(..., ge=0.0, le=1.0)
    niveau_malaria:    NiveauRisqueMalaria
    score_nutrition:   Optional[float] = Field(None, ge=0.0, le=1.0)
    niveau_nutrition:  Optional[ClassificationNutritionOMS] = None
    confiance:         float = Field(..., ge=0.0, le=1.0, description="Niveau de confiance du modèle")
    shap_values:       Optional[Dict[str, float]] = None
    timestamp_calcul:  datetime

    @field_validator("score_malaria", "score_nutrition", "confiance")
    @classmethod
    def arrondir_score(cls, v: Optional[float]) -> Optional[float]:
        return round(v, 4) if v is not None else None


# ═════════════════════════════════════════════════════════════════
# 7. SCHEMAS ALERTES
# ═════════════════════════════════════════════════════════════════

class SeveriteAlerte(str, Enum):
    SURVEILLANCE = "surveillance"
    ALERTE       = "alerte"
    URGENCE      = "urgence"
    CRISE        = "crise"


class TypeAlerte(str, Enum):
    PALUDISME   = "paludisme"
    NUTRITION   = "nutrition"
    CYCLONE     = "cyclone"
    SECHERESSE  = "sécheresse"
    COMBINEE    = "combinée"


class AlerteInput(_BaseSchema):
    """
    Création d'une alerte épidémiologique ou humanitaire.

    Endpoint : POST /api/v1/alerts/
    """
    region_code:  str           = Field(..., description="Région concernée")
    type_alerte:  TypeAlerte    = Field(..., description="Type d'alerte")
    severite:     SeveriteAlerte = Field(..., description="Niveau de sévérité")
    valeur:       float         = Field(..., description="Valeur déclenchante (score, taux…)")
    description:  str           = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Description de l'alerte (10–500 caractères)",
    )
    recommandations: Optional[List[str]] = Field(
        None,
        description="Actions recommandées (liste)",
        max_length=10,
    )
    expire_dans_jours: int = Field(
        7, ge=1, le=30,
        description="Durée de validité de l'alerte (jours)",
    )
    creee_par:    Optional[int] = Field(None, description="ID utilisateur créateur")

    @field_validator("region_code")
    @classmethod
    def valider_region(cls, v: str) -> str:
        if v not in REGIONS_MADAGASCAR:
            raise ValueError(f"Code région invalide : '{v}'")
        return v

    @field_validator("valeur")
    @classmethod
    def valider_valeur(cls, v: float) -> float:
        if v < 0:
            raise ValueError("La valeur déclenchante ne peut être négative")
        return round(v, 4)


# ═════════════════════════════════════════════════════════════════
# 8. SCHEMA RAPPORTS
# ═════════════════════════════════════════════════════════════════

class ReportRequest(_BaseSchema):
    """
    Requête de génération de rapport.

    Endpoint : POST /api/v1/reports/generate
    """
    type_rapport:   TypeRapportEnum = Field(..., description="Type de rapport à générer")
    region_codes:   Optional[List[str]] = Field(
        None,
        description="Régions incluses (None = toutes les 22 régions)",
    )
    date_debut:     date = Field(..., description="Début de la période couverte")
    date_fin:       date = Field(..., description="Fin de la période couverte")
    langue:         str  = Field("fr", description="Langue du rapport (fr, mg, en)")
    inclure_cartes: bool = Field(True,  description="Inclure cartes choroplèthes")
    inclure_recettes: bool = Field(False, description="Inclure suggestions de recettes nutritionnelles")
    email_destinataires: Optional[List[str]] = Field(
        None,
        description="Emails pour envoi automatique du rapport",
    )

    @field_validator("region_codes")
    @classmethod
    def valider_regions(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            invalides = [r for r in v if r not in REGIONS_MADAGASCAR]
            if invalides:
                raise ValueError(f"Codes régions invalides : {invalides}")
            if len(v) == 0:
                raise ValueError("La liste de régions ne peut être vide")
        return v

    @field_validator("langue")
    @classmethod
    def valider_langue(cls, v: str) -> str:
        if v not in {"fr", "mg", "en"}:
            raise ValueError(f"Langue '{v}' non supportée. Valeurs : fr, mg, en")
        return v

    @field_validator("email_destinataires")
    @classmethod
    def valider_emails(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        email_regex = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
        invalides = [e for e in v if not email_regex.match(e)]
        if invalides:
            raise ValueError(f"Adresses email invalides : {invalides}")
        return v

    @model_validator(mode="after")
    def valider_dates(self) -> "ReportRequest":
        if self.date_debut > self.date_fin:
            raise ValueError("date_debut doit être antérieure à date_fin")
        delta = (self.date_fin - self.date_debut).days
        if delta > 366:
            raise ValueError(
                f"Période trop longue : {delta} jours (max 366). "
                "Découper en plusieurs rapports."
            )
        return self


# ═════════════════════════════════════════════════════════════════
# 9. SCHEMA RECETTES NUTRITIONNELLES
# ═════════════════════════════════════════════════════════════════

class RecipeRequest(_BaseSchema):
    """
    Requête de sélection/génération de recettes nutritionnelles.

    Utilisée par recipe_selector.py pour filtrer les recettes adaptées
    à la région, la saison et le groupe cible.
    """
    region_code:     str                  = Field(..., description="Région pour adaptation culturelle")
    groupe_cible:    GroupeCibleNutrition = Field(..., description="Groupe de bénéficiaires")
    n_recettes:      int                  = Field(3, ge=1, le=10, description="Nombre de recettes souhaitées")
    ingredients_disponibles: Optional[List[str]] = Field(
        None,
        description="Ingrédients disponibles localement (filtre)",
    )
    exclure_allergenes: Optional[List[str]] = Field(
        None,
        description="Allergènes à exclure (arachide, gluten…)",
    )
    objectif_nutritionnel: Optional[str] = Field(
        None,
        description="Objectif principal : fer, protéines, vitamine_a, energie…",
    )

    @field_validator("region_code")
    @classmethod
    def valider_region(cls, v: str) -> str:
        if v not in REGIONS_MADAGASCAR:
            raise ValueError(f"Code région invalide : '{v}'")
        return v


# ═════════════════════════════════════════════════════════════════
# 10. VALIDATORS STANDALONE (sans Pydantic)
# ═════════════════════════════════════════════════════════════════

def validate_region_code(code: str) -> bool:
    """
    Valide un code région sans lever d'exception.

    Args:
        code : Code à valider (ex: "MDG-ANA")

    Returns:
        True si valide, False sinon (avec log warning).
    """
    valid = code in REGIONS_MADAGASCAR
    if not valid:
        log.warning("Code région invalide : '{}' — codes valides : {}", code, REGIONS_MADAGASCAR)
    return valid


def validate_score(score: Any, field_name: str = "score") -> float:
    """
    Valide et clamp un score de risque dans [0, 1].

    Args:
        score      : Valeur à valider (peut être str, float, int)
        field_name : Nom du champ pour les messages d'erreur

    Returns:
        Float ∈ [0.0, 1.0]

    Raises:
        ValueError si score non numérique.
    """
    try:
        score_f = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'{field_name}' doit être numérique, reçu : {type(score).__name__}"
        ) from exc

    if not math.isfinite(score_f):
        raise ValueError(f"'{field_name}' ne peut être NaN ou infini")

    if score_f < _SCORE_MIN or score_f > _SCORE_MAX:
        log.warning(
            "{} hors plage [{}, {}] : {} — clamp appliqué",
            field_name, _SCORE_MIN, _SCORE_MAX, score_f
        )
        score_f = max(_SCORE_MIN, min(_SCORE_MAX, score_f))

    return round(score_f, 4)


def validate_gam_rate(rate: Any) -> float:
    """
    Valide un taux GAM (Global Acute Malnutrition).

    Args:
        rate : Taux en pourcentage (ex: 12.5 pour 12.5%)

    Returns:
        Float ∈ [0.0, 60.0]

    Raises:
        ValueError si invalide.
    """
    try:
        rate_f = float(rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Taux GAM doit être numérique, reçu : {rate}") from exc

    if not (_GAM_MIN <= rate_f <= _GAM_MAX):
        raise ValueError(
            f"Taux GAM {rate_f}% hors plage réaliste [{_GAM_MIN}, {_GAM_MAX}%]. "
            "Vérifier l'unité (ratio 0–1 passé au lieu de pourcentage ?)"
        )
    return round(rate_f, 2)


def validate_date_range(
    start: date,
    end: date,
    max_days: int = 366,
) -> bool:
    """
    Valide une plage de dates.

    Args:
        start    : Date de début
        end      : Date de fin
        max_days : Nombre maximum de jours autorisé

    Returns:
        True si valide.

    Raises:
        ValueError avec message explicite si invalide.
    """
    if start > end:
        raise ValueError(
            f"Date début ({start}) doit être ≤ date fin ({end})"
        )
    if start < _DATE_MIN:
        raise ValueError(
            f"Date début ({start}) antérieure aux données disponibles ({_DATE_MIN})"
        )
    delta = (end - start).days
    if delta > max_days:
        raise ValueError(
            f"Plage de {delta} jours dépasse le maximum autorisé ({max_days} jours)"
        )
    return True


def validate_weather_payload(data: Dict[str, Any]) -> Optional[WeatherDataInput]:
    """
    Tente de construire un WeatherDataInput depuis un dict brut.

    Usage : dans weather_fetcher.py, avant insertion en base.

    Args:
        data : Payload brut d'une API météo (OpenWeatherMap, NASA, etc.)

    Returns:
        WeatherDataInput validé ou None si validation échoue (avec log error).
    """
    try:
        return WeatherDataInput(**data)
    except Exception as exc:
        log.error(
            "Payload météo invalide pour région={} : {}",
            data.get("region_code", "?"),
            exc,
        )
        return None


import math  # noqa: E402 — import ici pour validate_score (évite import circulaire)


# ═════════════════════════════════════════════════════════════════
# 11. QUALITÉ DES DONNÉES
# ═════════════════════════════════════════════════════════════════

@dataclass
class DataQualityReport:
    """
    Rapport de qualité d'un dataset — utilisé par feature_engineering.py
    et les scripts de preprocessing.
    """
    source:          str
    region_code:     str
    n_total:         int
    n_manquants:     int
    taux_manquants:  float = field(init=False)
    champs_problematiques: List[str] = field(default_factory=list)
    alertes:         List[str] = field(default_factory=list)
    valide:          bool = field(init=False)

    def __post_init__(self) -> None:
        self.taux_manquants = (
            round(self.n_manquants / self.n_total, 4)
            if self.n_total > 0 else 1.0
        )
        self.valide = self.taux_manquants < 0.30 and len(self.champs_problematiques) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source":               self.source,
            "region_code":          self.region_code,
            "n_total":              self.n_total,
            "n_manquants":          self.n_manquants,
            "taux_manquants_pct":   round(self.taux_manquants * 100, 1),
            "champs_problematiques": self.champs_problematiques,
            "alertes":              self.alertes,
            "valide":               self.valide,
        }


def check_missing_rate(
    values: List[Optional[float]],
    threshold: float = 0.30,
) -> bool:
    """
    Vérifie si le taux de valeurs manquantes est acceptable.

    Args:
        values    : Liste de valeurs (None = manquant)
        threshold : Taux max acceptable (défaut 30%)

    Returns:
        True si le taux de manquants est < threshold.
    """
    if not values:
        return False
    n_missing = sum(1 for v in values if v is None)
    rate = n_missing / len(values)
    if rate >= threshold:
        log.warning(
            "Taux de valeurs manquantes élevé : {:.1%} ({}/{})",
            rate, n_missing, len(values)
        )
        return False
    return True


def check_value_bounds(
    value: float,
    low: float,
    high: float,
    field_name: str = "valeur",
) -> bool:
    """
    Vérifie qu'une valeur est dans une plage attendue.

    Args:
        value      : Valeur à tester
        low, high  : Bornes (inclusives)
        field_name : Nom pour le message de log

    Returns:
        True si dans les bornes.
    """
    in_bounds = low <= value <= high
    if not in_bounds:
        log.warning(
            "{} hors bornes : {} (attendu [{}, {}])",
            field_name, value, low, high
        )
    return in_bounds


def score_to_niveau_malaria(score: float) -> NiveauRisqueMalaria:
    """
    Convertit un score numérique [0,1] en niveau de risque catégoriel.

    Aligné avec SEUILS_RISQUE_MALARIA dans constants.py.

    Args:
        score : Float ∈ [0.0, 1.0]

    Returns:
        NiveauRisqueMalaria correspondant.
    """
    score = validate_score(score, "score_malaria")
    for niveau, (low, high) in SEUILS_RISQUE_MALARIA.items():
        if low <= score < high:
            return NiveauRisqueMalaria(niveau)
    return NiveauRisqueMalaria.TRES_ELEVE  # score == 1.0