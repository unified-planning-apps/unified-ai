"""
Constantes globales, enums et valeurs de référence pour tout le projet.
"""

from __future__ import annotations

from enum import Enum


# ─────────────────────────────────────────────────────────────────
# Rôles utilisateurs
# ─────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    ADMIN    = "admin"     # Accès total, gestion utilisateurs
    NATIONAL = "national"  # Accès toutes régions, pas de gestion users
    REGIONAL = "regional"  # Accès restreint à sa région
    VIEWER   = "viewer"    # Lecture seule, accès limité


# ─────────────────────────────────────────────────────────────────
# Niveaux de risque paludisme
# ─────────────────────────────────────────────────────────────────

class NiveauRisqueMalaria(str, Enum):
    FAIBLE     = "faible"
    MOYEN      = "moyen"
    ELEVE      = "élevé"
    TRES_ELEVE = "très élevé"

SEUILS_RISQUE_MALARIA = {
    "faible":     (0.00, 0.25),
    "moyen":      (0.25, 0.50),
    "élevé":      (0.50, 0.75),
    "très élevé": (0.75, 1.00),
}


# ─────────────────────────────────────────────────────────────────
# Classifications nutrition OMS
# ─────────────────────────────────────────────────────────────────

class ClassificationNutritionOMS(str, Enum):
    ACCEPTABLE = "acceptable"   # GAM < 5%
    ALERTE     = "alerte"       # GAM 5–10%
    URGENCE    = "urgence"      # GAM 10–15%
    CRISE      = "crise"        # GAM > 15%

SEUILS_GAM = {
    "acceptable": 5.0,
    "alerte":    10.0,
    "urgence":   15.0,
}


# ─────────────────────────────────────────────────────────────────
# Saisons Madagascar
# ─────────────────────────────────────────────────────────────────

class Saison(str, Enum):
    SAISON_PLUIES  = "saison_pluies"   # Nov–Avr
    TRANSITION     = "transition"       # Oct & Mai
    SAISON_SECHE   = "saison_seche"    # Juin–Sep

# Mois par saison (Haute Terres — différent pour côte Est/Ouest)
MOIS_SAISON_PLUIES  = {11, 12, 1, 2, 3, 4}
MOIS_SAISON_SECHE   = {6, 7, 8, 9}
MOIS_TRANSITION     = {5, 10}

def get_saison_courante(mois: int) -> Saison:
    if mois in MOIS_SAISON_PLUIES:
        return Saison.SAISON_PLUIES
    elif mois in MOIS_SAISON_SECHE:
        return Saison.SAISON_SECHE
    return Saison.TRANSITION


# ─────────────────────────────────────────────────────────────────
# Zones endémicité paludisme
# ─────────────────────────────────────────────────────────────────

class EndemicitePaludisme(str, Enum):
    FAIBLE    = "low"
    MEDIUM    = "medium"
    ELEVE     = "high"
    TRES_ELEVE = "very_high"


# ─────────────────────────────────────────────────────────────────
# Paramètres Feature Engineering
# ─────────────────────────────────────────────────────────────────

# Fenêtres temporelles pour features météo (jours)
WEATHER_LAG_WINDOWS = [7, 14, 21, 30]

# Température optimale de développement Plasmodium falciparum
TEMP_OPTIMAL_PLASMODIUM_MIN = 20.0   # °C
TEMP_OPTIMAL_PLASMODIUM_MAX = 30.0   # °C
TEMP_SEUIL_ARRET_DEVELOPPEMENT = 16.0  # °C — en dessous: pas de transmission

# Précipitations seuil accumulation favorisant gîtes larvaires (mm/30j)
PLUIE_SEUIL_GITES_LARVAIRES = 100.0

# Seuil NDVI — végétation dense favorable aux moustiques
NDVI_SEUIL_RISQUE = 0.5


# ─────────────────────────────────────────────────────────────────
# Paramètres rapports
# ─────────────────────────────────────────────────────────────────

class TypeRapportEnum(str, Enum):
    PALUDISME_HEBDO   = "paludisme_hebdomadaire"
    NUTRITION_HEBDO   = "nutrition_hebdomadaire"
    COMBINE_HEBDO     = "combine_hebdomadaire"
    URGENCE           = "urgence"
    MENSUEL           = "mensuel"

# Durée de conservation des rapports en jours
RETENTION_RAPPORTS_JOURS = 90


# ─────────────────────────────────────────────────────────────────
# Codes régions Madagascar (22 régions officielles)
# ─────────────────────────────────────────────────────────────────

REGIONS_MADAGASCAR = [
    "MDG-ANA",   # Analamanga
    "MDG-VAK",   # Vakinankaratra
    "MDG-ITM",   # Itasy
    "MDG-BMT",   # Bongolava
    "MDG-MAT",   # Matsiatra Ambony
    "MDG-ATI",   # Amoron'i Mania
    "MDG-VAT",   # Vatovavy
    "MDG-FIT",   # Fitovinany
    "MDG-ANO",   # Atsimo-Atsinanana
    "MDG-ATS",   # Atsinanana
    "MDG-ANA2",  # Analanjirofo
    "MDG-ALA",   # Alaotra-Mangoro
    "MDG-BOE",   # Boeny
    "MDG-SOF",   # Sofia
    "MDG-MEN",   # Melaky
    "MDG-MEN2",  # Menabe
    "MDG-DIA",   # Diana
    "MDG-SAV",   # Sava
    "MDG-IHO",   # Ihorombe
    "MDG-ASO",   # Atsimo-Andrefana
    "MDG_AND",   # Androy
    "MDG-AAN",   # Anosy
]

# ─────────────────────────────────────────────────────────────────
# Couleurs niveaux d'alerte (choroplèthe dashboard)
# ─────────────────────────────────────────────────────────────────

COULEURS_ALERTE = {
    "vert":   "#388E3C",   # Score < 0.25
    "jaune":  "#F9A825",   # 0.25 ≤ score < 0.50
    "orange": "#F57C00",   # 0.50 ≤ score < 0.75
    "rouge":  "#D32F2F",   # Score ≥ 0.75
}

# ─────────────────────────────────────────────────────────────────
# Groupes cibles nutrition
# ─────────────────────────────────────────────────────────────────

class GroupeCibleNutrition(str, Enum):
    ENFANTS_6_23M      = "enfants_6_23m"
    ENFANTS_2_5ANS     = "enfants_2_5ans"
    FEMMES_ENCEINTES   = "femmes_enceintes"
    FEMMES_ALLAITANTES = "femmes_allaitantes"
    FAMILLE            = "famille"

# Pourcentage moyen enfants < 5 ans dans population Madagascar
PCT_ENFANTS_MOINS_5ANS = 0.17  # 17%
PCT_FEMMES_ENCEINTES   = 0.04  # 4%