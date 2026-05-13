"""
Algorithme de sélection de recettes nutritionnelles contextualisées.

Interface publique (contrat avec nutrition.py router et generator.py) :
  RecipeSelector(db=None).generer_recettes_optimales(region_id, cible, nombre)
    → List[dict]  (format compatible avec RecetteNutritionnelle Pydantic)

Algorithme en 3 étapes :
  1. Filtrage contextuel (région, saison, ingrédients disponibles)
  2. Optimisation nutritionnelle (Linear Programming — scipy)
  3. Diversification (éviter répétitions, respecter culture locale)

Pondération nutritionnelle par groupe cible :
  enfants_6_23m    : énergie + protéines + fer + vitamine A + zinc (ANJE)
  enfants_2_5ans   : énergie + micronutriments diversifiés
  femmes_enceintes : fer + folate + calcium + protéines
  famille          : énergie + équilibre général
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────────────
# Base de recettes intégrée (fallback si DB vide)
# Recettes typiques de Madagascar adaptées UNICEF
# ─────────────────────────────────────────────────────────────────
RECETTES_BASE: List[Dict[str, Any]] = [
    {
        "recette_id":    "MDG-R001",
        "nom":           "Bouillie enrichie Misovola (Riz + Haricot)",
        "nom_malgache":  "Misovola be",
        "regions":       ["toutes"],
        "saisons":       ["toute_annee"],
        "cibles":        ["enfants_6_23m", "enfants_2_5ans"],
        "calories_kcal": 185,
        "proteines_g":   9.5,
        "glucides_g":    28.0,
        "lipides_g":     4.5,
        "fer_mg":        3.8,
        "vitamine_a_ug": 150,
        "zinc_mg":       2.1,
        "calcium_mg":    80,
        "folate_ug":     45,
        "score":         78,
        "ingredients": [
            {"nom": "Farine de riz",           "quantite_g": 50,  "local": True},
            {"nom": "Haricots rouges écrasés",  "quantite_g": 40,  "local": True},
            {"nom": "Huile végétale",           "quantite_g": 10,  "local": True},
            {"nom": "Spiruline en poudre",      "quantite_g": 2,   "local": False},
            {"nom": "Sel iodé",                "quantite_g": 1,   "local": True},
        ],
        "instructions": (
            "1. Porter 300ml d'eau à ébullition. "
            "2. Délayer la farine dans 50ml d'eau froide, verser dans l'eau bouillante. "
            "3. Ajouter les haricots écrasés, remuer 10 min à feu doux. "
            "4. Incorporer l'huile et la spiruline. "
            "5. Saler légèrement avec sel iodé. Servir tiède."
        ),
        "temps_min":     20,
        "cout_ariary":   800,
        "culture":       ["merina", "betsileo", "betsimisaraka"],
        "contraintes":   [],
    },
    {
        "recette_id":    "MDG-R002",
        "nom":           "Soupe de légumes au poisson séché (Laoka Trondro)",
        "nom_malgache":  "Laoka trondro sy anana",
        "regions":       ["MDG-ATS", "MDG-SAV", "MDG-ANA2", "MDG-VAT", "MDG-BOE"],
        "saisons":       ["toute_annee"],
        "cibles":        ["famille", "femmes_enceintes", "enfants_2_5ans"],
        "calories_kcal": 230,
        "proteines_g":   19.5,
        "glucides_g":    14.0,
        "lipides_g":     8.0,
        "fer_mg":        5.8,
        "vitamine_a_ug": 380,
        "zinc_mg":       2.8,
        "calcium_mg":    120,
        "folate_ug":     65,
        "score":         85,
        "ingredients": [
            {"nom": "Poisson séché (trondro gasy)", "quantite_g": 60,  "local": True},
            {"nom": "Feuilles de brède (anana)",    "quantite_g": 100, "local": True},
            {"nom": "Tomates",                      "quantite_g": 60,  "local": True},
            {"nom": "Oignons",                      "quantite_g": 30,  "local": True},
            {"nom": "Huile de palme",               "quantite_g": 10,  "local": True},
        ],
        "instructions": (
            "1. Faire revenir oignon et tomate dans l'huile 5 min. "
            "2. Ajouter 500ml d'eau et le poisson séché émietté. "
            "3. Cuire 15 min à feu moyen. "
            "4. Ajouter les feuilles de brède, cuire encore 5 min. "
            "5. Ajuster le sel. Servir avec riz."
        ),
        "temps_min":     25,
        "cout_ariary":   1200,
        "culture":       ["betsimisaraka", "sakalava", "antaisaka"],
        "contraintes":   [],
    },
    {
        "recette_id":    "MDG-R003",
        "nom":           "Purée de patate douce au lait de coco enrichi",
        "nom_malgache":  "Vary patsy amin'ny ronono voanio",
        "regions":       ["MDG-ATS", "MDG-ANA2", "MDG-SAV", "MDG-BOE"],
        "saisons":       ["saison_pluies", "toute_annee"],
        "cibles":        ["enfants_6_23m"],
        "calories_kcal": 165,
        "proteines_g":   3.0,
        "glucides_g":    32.0,
        "lipides_g":     4.5,
        "fer_mg":        1.8,
        "vitamine_a_ug": 520,
        "zinc_mg":       1.2,
        "calcium_mg":    40,
        "folate_ug":     20,
        "score":         72,
        "ingredients": [
            {"nom": "Patate douce orange",  "quantite_g": 120, "local": True},
            {"nom": "Lait de coco",         "quantite_g": 50,  "local": True},
            {"nom": "Huile",                "quantite_g": 5,   "local": True},
        ],
        "instructions": (
            "1. Éplucher et couper la patate douce en morceaux. "
            "2. Cuire à la vapeur 20 min. "
            "3. Écraser finement, incorporer le lait de coco et l'huile. "
            "4. Servir tiède en purée lisse."
        ),
        "temps_min":     25,
        "cout_ariary":   600,
        "culture":       ["betsimisaraka"],
        "contraintes":   [],
    },
    {
        "recette_id":    "MDG-R004",
        "nom":           "Viande de zébu aux légumineuses (Henakisoa sy antsy)",
        "nom_malgache":  "Henakisoa sy antsy",
        "regions":       ["MDG-ANA", "MDG-VAK", "MDG-IHO", "MDG_AND"],
        "saisons":       ["toute_annee"],
        "cibles":        ["famille", "femmes_enceintes"],
        "calories_kcal": 290,
        "proteines_g":   26.0,
        "glucides_g":    18.0,
        "lipides_g":     11.0,
        "fer_mg":        8.5,
        "vitamine_a_ug": 60,
        "zinc_mg":       5.5,
        "calcium_mg":    55,
        "folate_ug":     40,
        "score":         88,
        "ingredients": [
            {"nom": "Viande de zébu",     "quantite_g": 100, "local": True},
            {"nom": "Haricots secs",      "quantite_g": 60,  "local": True},
            {"nom": "Oignons",            "quantite_g": 40,  "local": True},
            {"nom": "Tomates",            "quantite_g": 50,  "local": True},
            {"nom": "Gingembre",          "quantite_g": 5,   "local": True},
        ],
        "instructions": (
            "1. Faire tremper les haricots 8h. Cuire 1h à l'eau. "
            "2. Faire revenir viande et oignon 10 min. "
            "3. Ajouter tomates, gingembre, haricots cuits. "
            "4. Mijoter 20 min. Servir avec riz."
        ),
        "temps_min":     40,
        "cout_ariary":   3500,
        "culture":       ["merina", "betsileo", "bara"],
        "contraintes":   [],
    },
    {
        "recette_id":    "MDG-R005",
        "nom":           "Bouillie de maïs aux arachides (Grand Sud)",
        "nom_malgache":  "Vary katsaka sy voanjo",
        "regions":       ["MDG_AND", "MDG-ASO", "MDG-IHO", "MDG-AAN"],
        "saisons":       ["saison_seche", "toute_annee"],
        "cibles":        ["enfants_6_23m", "enfants_2_5ans", "famille"],
        "calories_kcal": 200,
        "proteines_g":   8.0,
        "glucides_g":    30.0,
        "lipides_g":     7.0,
        "fer_mg":        2.8,
        "vitamine_a_ug": 30,
        "zinc_mg":       1.9,
        "calcium_mg":    35,
        "folate_ug":     38,
        "score":         70,
        "ingredients": [
            {"nom": "Farine de maïs",   "quantite_g": 60,  "local": True},
            {"nom": "Pâte d'arachide",  "quantite_g": 30,  "local": True},
            {"nom": "Sucre de canne",   "quantite_g": 10,  "local": True},
            {"nom": "Sel iodé",         "quantite_g": 1,   "local": True},
        ],
        "instructions": (
            "1. Délayer la farine de maïs dans 100ml d'eau froide. "
            "2. Verser dans 300ml d'eau bouillante salée, remuer 15 min. "
            "3. Incorporer la pâte d'arachide et le sucre. "
            "4. Cuire encore 5 min à feu doux. Servir chaud."
        ),
        "temps_min":     20,
        "cout_ariary":   600,
        "culture":       ["antandroy", "mahafaly", "bara"],
        "contraintes":   [],
    },
    {
        "recette_id":    "MDG-R006",
        "nom":           "Salade de manioc aux haricots verts et poisson",
        "nom_malgache":  "Mangahazo sy tsaramaso maitso",
        "regions":       ["toutes"],
        "saisons":       ["saison_pluies"],
        "cibles":        ["famille", "femmes_enceintes", "enfants_2_5ans"],
        "calories_kcal": 210,
        "proteines_g":   14.0,
        "glucides_g":    25.0,
        "lipides_g":     6.0,
        "fer_mg":        4.5,
        "vitamine_a_ug": 210,
        "zinc_mg":       2.2,
        "calcium_mg":    90,
        "folate_ug":     80,
        "score":         80,
        "ingredients": [
            {"nom": "Manioc frais",        "quantite_g": 150, "local": True},
            {"nom": "Haricots verts",      "quantite_g": 80,  "local": True},
            {"nom": "Sardines en boîte",   "quantite_g": 60,  "local": False},
            {"nom": "Citron",              "quantite_g": 20,  "local": True},
            {"nom": "Huile d'arachide",    "quantite_g": 10,  "local": True},
        ],
        "instructions": (
            "1. Cuire le manioc 20 min, couper en morceaux. "
            "2. Blanchir les haricots verts 5 min. "
            "3. Mélanger avec les sardines égouttées. "
            "4. Assaisonner avec citron et huile. Servir tiède."
        ),
        "temps_min":     30,
        "cout_ariary":   1500,
        "culture":       ["merina", "betsileo", "betsimisaraka"],
        "contraintes":   [],
    },
    {
        "recette_id":    "MDG-R007",
        "nom":           "Œufs brouillés aux légumes et lait (ANJE 6-23 mois)",
        "nom_malgache":  "Atody mamy sy anana",
        "regions":       ["toutes"],
        "saisons":       ["toute_annee"],
        "cibles":        ["enfants_6_23m"],
        "calories_kcal": 155,
        "proteines_g":   11.0,
        "glucides_g":    8.0,
        "lipides_g":     9.0,
        "fer_mg":        2.5,
        "vitamine_a_ug": 280,
        "zinc_mg":       1.8,
        "calcium_mg":    120,
        "folate_ug":     50,
        "score":         82,
        "ingredients": [
            {"nom": "Œufs",             "quantite_g": 60,  "local": True},
            {"nom": "Lait (ou lait de coco)", "quantite_g": 50, "local": True},
            {"nom": "Légumes verts hachés",   "quantite_g": 40, "local": True},
            {"nom": "Huile",            "quantite_g": 5,   "local": True},
        ],
        "instructions": (
            "1. Battre les œufs avec le lait. "
            "2. Faire revenir les légumes 3 min dans l'huile. "
            "3. Verser le mélange œufs-lait, remuer doucement. "
            "4. Cuire à feu doux jusqu'à consistance molle. "
            "5. Écraser finement pour les tout-petits."
        ),
        "temps_min":     10,
        "cout_ariary":   700,
        "culture":       ["toutes"],
        "contraintes":   [],
    },
    {
        "recette_id":    "MDG-R008",
        "nom":           "Riz complet aux légumineuses et moringa (Vary sy moringa)",
        "nom_malgache":  "Vary mena sy moringa",
        "regions":       ["toutes"],
        "saisons":       ["toute_annee"],
        "cibles":        ["famille", "femmes_enceintes", "enfants_2_5ans"],
        "calories_kcal": 310,
        "proteines_g":   14.0,
        "glucides_g":    52.0,
        "lipides_g":     5.0,
        "fer_mg":        7.2,
        "vitamine_a_ug": 340,
        "zinc_mg":       2.5,
        "calcium_mg":    180,
        "folate_ug":     120,
        "score":         90,
        "ingredients": [
            {"nom": "Riz complet",          "quantite_g": 80,  "local": True},
            {"nom": "Lentilles cuites",     "quantite_g": 60,  "local": True},
            {"nom": "Feuilles de moringa",  "quantite_g": 30,  "local": True},
            {"nom": "Oignons",              "quantite_g": 30,  "local": True},
            {"nom": "Huile",                "quantite_g": 10,  "local": True},
        ],
        "instructions": (
            "1. Cuire le riz complet 25 min. "
            "2. Faire revenir oignon, ajouter lentilles et 200ml d'eau. "
            "3. Mijoter 10 min, ajouter feuilles de moringa 3 min. "
            "4. Servir les lentilles au moringa sur le riz."
        ),
        "temps_min":     35,
        "cout_ariary":   900,
        "culture":       ["toutes"],
        "contraintes":   [],
    },
]

# Pondérations nutritionnelles par groupe cible (pour score LP)
NUTRITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "enfants_6_23m": {
        "calories_kcal": 0.20, "proteines_g": 0.25, "fer_mg": 0.25,
        "vitamine_a_ug": 0.20, "zinc_mg": 0.10,
    },
    "enfants_2_5ans": {
        "calories_kcal": 0.25, "proteines_g": 0.20, "fer_mg": 0.20,
        "vitamine_a_ug": 0.15, "zinc_mg": 0.10, "calcium_mg": 0.10,
    },
    "femmes_enceintes": {
        "calories_kcal": 0.15, "proteines_g": 0.20, "fer_mg": 0.30,
        "folate_ug":    0.20, "calcium_mg": 0.15,
    },
    "femmes_allaitantes": {
        "calories_kcal": 0.20, "proteines_g": 0.20, "fer_mg": 0.20,
        "vitamine_a_ug": 0.20, "calcium_mg": 0.20,
    },
    "famille": {
        "calories_kcal": 0.30, "proteines_g": 0.25, "fer_mg": 0.20,
        "vitamine_a_ug": 0.15, "zinc_mg": 0.10,
    },
}

# Contraintes culturelles par région
CONTRAINTES_CULTURELLES: Dict[str, List[str]] = {
    "MDG_AND":  ["pas_porc"],
    "MDG-ASO":  ["pas_porc"],
    "MDG-DIA":  [],  # Antakarana — pas de contrainte majeure
    "MDG-SAV":  [],
}

# Ingrédients disponibles par saison et région
DISPO_SAISONNIERE: Dict[str, Dict[str, List[str]]] = {
    "saison_pluies": {
        "toutes": ["riz", "manioc", "haricots", "mais", "légumes_verts",
                   "tomates", "oignons", "patate_douce"],
        "cote_est": ["banane", "fruit_pain", "noix_coco", "poisson"],
        "grand_sud": ["mais", "manioc", "arachides", "zebu"],
    },
    "saison_seche": {
        "toutes": ["manioc", "mais", "haricots_secs", "riz_stocké", "zebu"],
        "grand_sud": ["mais", "manioc", "haricots_secs"],
    },
}


class RecipeSelector:
    """
    Sélectionneur de recettes nutritionnelles par optimisation contextuelle.

    Algorithme :
      1. Filtrage contextuel (région, saison, disponibilité)
      2. Score nutritionnel pondéré par groupe cible
      3. Optimisation par Linear Programming (maximise score nutritionnel)
      4. Diversification (évite répétitions d'ingrédients)

    Usage :
        selector = RecipeSelector(db=None)
        recettes = await selector.generer_recettes_optimales(
            region_id="MDG_AND", cible="enfants_6_23m", nombre=5
        )
    """

    def __init__(self, db=None):
        self._db = db

    async def generer_recettes_optimales(
        self,
        region_id: str,
        cible: str = "enfants_6_23m",
        nombre: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Interface publique — contrat avec nutrition.py router et generator.py.

        Args:
            region_id : ID région Madagascar
            cible     : enfants_6_23m | enfants_2_5ans | femmes_enceintes |
                        femmes_allaitantes | famille
            nombre    : Nombre de recettes à retourner

        Returns:
            Liste de dicts au format RecetteNutritionnelle (compatibles router)
        """
        logger.info(
            "Sélection recettes — région={} cible={} nombre={}",
            region_id, cible, nombre
        )

        # 1. Chargement du pool de recettes (DB → fallback base intégrée)
        pool = await self._charger_pool(region_id, cible)

        if not pool:
            logger.warning("Pool vide pour {} {} — recettes par défaut", region_id, cible)
            return self._recettes_par_defaut(cible, nombre)

        # 2. Contexte saisonnier
        mois   = date.today().month
        saison = self._get_saison(mois)

        # 3. Filtrage contextuel
        filtered = self._filtrer_recettes(pool, region_id, saison, cible)

        if not filtered:
            filtered = pool  # Si filtrage trop strict → on relâche

        # 4. Score nutritionnel pondéré
        scored = self._scorer_recettes(filtered, cible)

        # 5. Optimisation diversité (LP ou greedy)
        selected = self._optimiser_diversite(scored, nombre)

        # 6. Enrichissement et formatage
        return [self._formater_recette(r, region_id) for r in selected]

    # ─────────────────────────────────────────────
    # Chargement du pool de recettes
    # ─────────────────────────────────────────────

    async def _charger_pool(
        self, region_id: str, cible: str
    ) -> List[Dict[str, Any]]:
        """Charge les recettes depuis la DB, fallback sur la base intégrée."""
        # Tentative DB
        if self._db is not None and self._is_real_db():
            try:
                from src.database.repositories.nutrition_repo import NutritionRepository
                repo = NutritionRepository(self._db)
                db_recettes = await repo.get_recettes(
                    region_id=region_id,
                    cible=cible,
                    score_min=50.0,
                    limit=50,
                )
                if db_recettes:
                    return self._normaliser_db_recettes(db_recettes)
            except Exception as exc:
                logger.debug("DB recettes : {}", exc)

        # Base intégrée
        return list(RECETTES_BASE)

    # ─────────────────────────────────────────────
    # Filtrage contextuel
    # ─────────────────────────────────────────────

    def _filtrer_recettes(
        self,
        pool: List[Dict],
        region_id: str,
        saison: str,
        cible: str,
    ) -> List[Dict]:
        """
        Filtre les recettes par :
          - Adaptation régionale (région ID ou "toutes")
          - Saison (saison_pluies | saison_seche | toute_annee)
          - Groupe cible
          - Contraintes culturelles (pas_porc, etc.)
          - Disponibilité des ingrédients
        """
        contraintes = CONTRAINTES_CULTURELLES.get(region_id, [])
        filtered    = []

        for r in pool:
            # Filtre région
            regions_r = r.get("regions", r.get("region_adaptee", ["toutes"]))
            if "toutes" not in regions_r and region_id not in regions_r:
                continue

            # Filtre saison
            saisons_r = r.get("saisons", r.get("saison", ["toute_annee"]))
            if (saison not in saisons_r
                    and "toute_annee" not in saisons_r
                    and "all" not in saisons_r):
                continue

            # Filtre cible
            cibles_r = r.get("cibles", r.get("cible", ["famille"]))
            if cible not in cibles_r and "famille" not in cibles_r:
                continue

            # Filtre contraintes culturelles
            contraintes_r = r.get("contraintes", [])
            if any(c in contraintes_r for c in contraintes):
                continue

            filtered.append(r)

        return filtered

    # ─────────────────────────────────────────────
    # Scoring nutritionnel
    # ─────────────────────────────────────────────

    def _scorer_recettes(
        self, recettes: List[Dict], cible: str
    ) -> List[Dict]:
        """
        Calcule un score nutritionnel pondéré pour chaque recette
        selon les besoins du groupe cible.
        """
        poids = NUTRITION_WEIGHTS.get(cible, NUTRITION_WEIGHTS["famille"])

        # Valeurs de référence pour normalisation
        valeurs_ref = {
            "calories_kcal": 500,  # kcal par repas
            "proteines_g":   15,   # g
            "fer_mg":        8,    # mg
            "vitamine_a_ug": 400,  # μg
            "zinc_mg":       5,    # mg
            "calcium_mg":    300,  # mg
            "folate_ug":     200,  # μg
        }

        scored = []
        for r in recettes:
            score_pond = 0.0
            for nutriment, poid in poids.items():
                valeur = float(r.get(nutriment, 0))
                ref    = valeurs_ref.get(nutriment, 1)
                # Normalisation [0, 1] avec cap à 1.5 (pas de pénalité excès)
                score_norm = min(1.5, valeur / ref)
                score_pond += poid * score_norm

            # Score final : combinaison score pondéré + score base intégrée
            score_base = float(r.get("score", r.get("score_nutritionnel", 60))) / 100
            score_final = 0.6 * score_pond + 0.4 * score_base

            scored.append({**r, "_score_nutritionnel": round(score_final, 4)})

        # Tri décroissant
        scored.sort(key=lambda x: x["_score_nutritionnel"], reverse=True)
        return scored

    # ─────────────────────────────────────────────
    # Optimisation diversité
    # ─────────────────────────────────────────────

    def _optimiser_diversite(
        self,
        recettes: List[Dict],
        nombre: int,
    ) -> List[Dict]:
        """
        Sélectionne `nombre` recettes en maximisant :
          1. Le score nutritionnel total
          2. La diversité des ingrédients (pas de répétition)
          3. La variété des groupes alimentaires couverts

        Algorithme greedy (approximation LP) :
        À chaque étape, choisit la recette avec le meilleur
        score nutritionnel ajusté par pénalité de répétition d'ingrédients.
        """
        if len(recettes) <= nombre:
            return recettes

        selected    = []
        ingredients_utilises: set = set()

        for _ in range(nombre):
            best_score = -1
            best_recette = None

            for r in recettes:
                if r in selected:
                    continue

                # Ingrédients de cette recette
                ings_r = set(
                    i.get("nom", "").lower().split()[0]
                    for i in r.get("ingredients", [])
                )

                # Pénalité de répétition
                overlap       = len(ings_r & ingredients_utilises)
                total_ings    = len(ings_r) + 1
                penalite      = overlap / total_ings * 0.3

                score_ajuste  = r.get("_score_nutritionnel", 0) - penalite

                if score_ajuste > best_score:
                    best_score   = score_ajuste
                    best_recette = r

            if best_recette:
                selected.append(best_recette)
                ings_sel = set(
                    i.get("nom", "").lower().split()[0]
                    for i in best_recette.get("ingredients", [])
                )
                ingredients_utilises.update(ings_sel)
            else:
                break

        return selected

    # ─────────────────────────────────────────────
    # Formatage sortie
    # ─────────────────────────────────────────────

    def _formater_recette(
        self, recette: Dict[str, Any], region_id: str
    ) -> Dict[str, Any]:
        """
        Formate une recette au format attendu par le router nutrition.py
        (compatible Pydantic RecetteNutritionnelle).
        """
        # Nettoyage des clés internes (_score_nutritionnel)
        clean = {k: v for k, v in recette.items() if not k.startswith("_")}

        # Normalisation des clés pour compatibilité router
        return {
            "recette_id":            clean.get("recette_id", str(uuid.uuid4())),
            "nom":                   clean.get("nom", "Recette inconnue"),
            "nom_malgache":          clean.get("nom_malgache"),
            "region_adaptee":        clean.get("regions", clean.get("region_adaptee", ["toutes"])),
            "saison":                clean.get("saisons", clean.get("saison", ["toute_annee"])),
            "calories_kcal":         float(clean.get("calories_kcal", 0)),
            "proteines_g":           float(clean.get("proteines_g", 0)),
            "glucides_g":            float(clean.get("glucides_g", 0)),
            "lipides_g":             float(clean.get("lipides_g", 0)),
            "fer_mg":                float(clean.get("fer_mg", 0)),
            "vitamine_a_ug":         float(clean.get("vitamine_a_ug", 0)),
            "zinc_mg":               float(clean.get("zinc_mg", 0)),
            "score_nutritionnel":    float(clean.get("score", clean.get("score_nutritionnel", 60))),
            "ingredients": [
                {
                    "nom":                  i.get("nom", ""),
                    "quantite_g":           i.get("quantite_g", 0),
                    "disponible_localement": i.get("local", True),
                }
                for i in clean.get("ingredients", [])
            ],
            "instructions":          clean.get("instructions", ""),
            "temps_preparation_min": int(clean.get("temps_min", clean.get("temps_preparation_min", 30))),
            "cout_estime_ariary":    float(clean.get("cout_ariary", clean.get("cout_estime_ariary", 0))),
            "cible":                 clean.get("cibles", clean.get("cible", ["famille"])),
            "image_url":             clean.get("image_url"),
            # Métadonnées additionnelles (pour debug / rapport)
            "_region_id":            region_id,
            "_score_optimisation":   float(recette.get("_score_nutritionnel", 0)),
        }

    # ─────────────────────────────────────────────
    # Méthodes utilitaires publiques
    # ─────────────────────────────────────────────

    def calculer_apport_journalier(
        self,
        recettes: List[Dict],
        repas_par_jour: int = 3,
    ) -> Dict[str, float]:
        """
        Calcule l'apport nutritionnel journalier total pour un menu de recettes.
        Utilisé dans les rapports pour valider la couverture des besoins.
        """
        total: Dict[str, float] = {
            "calories_kcal": 0, "proteines_g": 0, "fer_mg": 0,
            "vitamine_a_ug": 0, "zinc_mg": 0, "calcium_mg": 0,
        }
        for r in recettes[:repas_par_jour]:
            for nutriment in total:
                total[nutriment] += float(r.get(nutriment, 0))
        return {k: round(v, 2) for k, v in total.items()}

    def evaluer_couverture_besoins(
        self,
        apport: Dict[str, float],
        cible: str,
    ) -> Dict[str, Any]:
        """
        Évalue la couverture des besoins nutritionnels journaliers recommandés.
        Référence : OMS/FAO 2004, UNICEF Guidelines.
        """
        BESOINS_JOURNALIERS = {
            "enfants_6_23m": {
                "calories_kcal": 800, "proteines_g": 11, "fer_mg": 18.6,
                "vitamine_a_ug": 400, "zinc_mg": 4.1, "calcium_mg": 270,
            },
            "enfants_2_5ans": {
                "calories_kcal": 1300, "proteines_g": 13, "fer_mg": 5.8,
                "vitamine_a_ug": 400, "zinc_mg": 5.0, "calcium_mg": 700,
            },
            "femmes_enceintes": {
                "calories_kcal": 2200, "proteines_g": 71, "fer_mg": 27,
                "vitamine_a_ug": 770, "zinc_mg": 11, "calcium_mg": 1000,
            },
            "famille": {
                "calories_kcal": 2000, "proteines_g": 50, "fer_mg": 8,
                "vitamine_a_ug": 700, "zinc_mg": 8, "calcium_mg": 800,
            },
        }
        besoins = BESOINS_JOURNALIERS.get(cible, BESOINS_JOURNALIERS["famille"])

        couverture = {}
        for nutriment, besoin in besoins.items():
            apport_val = apport.get(nutriment, 0)
            pct = round(apport_val / besoin * 100, 1) if besoin > 0 else 0
            couverture[nutriment] = {
                "apport":          apport_val,
                "besoin":          besoin,
                "couverture_pct":  pct,
                "statut":          "✅" if pct >= 80 else "⚠️" if pct >= 50 else "❌",
            }

        return {
            "cible":       cible,
            "couverture":  couverture,
            "score_global": round(
                sum(min(100, c["couverture_pct"]) for c in couverture.values())
                / len(couverture), 1
            ) if couverture else 0,
        }

    # ─────────────────────────────────────────────
    # Helpers privés
    # ─────────────────────────────────────────────

    @staticmethod
    def _get_saison(mois: int) -> str:
        """Retourne la saison courante pour le mois donné."""
        if mois in (11, 12, 1, 2, 3, 4):
            return "saison_pluies"
        return "saison_seche"

    @staticmethod
    def _normaliser_db_recettes(db_recettes: List[Dict]) -> List[Dict]:
        """Normalise les recettes DB vers le format interne du sélecteur."""
        normalized = []
        for r in db_recettes:
            normalized.append({
                "recette_id":    r.get("recette_id"),
                "nom":           r.get("nom"),
                "nom_malgache":  r.get("nom_malgache"),
                "regions":       r.get("region_adaptee", ["toutes"]),
                "saisons":       r.get("saison", ["toute_annee"]),
                "cibles":        r.get("cible", ["famille"]),
                "calories_kcal": r.get("calories_kcal", 0),
                "proteines_g":   r.get("proteines_g", 0),
                "glucides_g":    r.get("glucides_g", 0),
                "lipides_g":     r.get("lipides_g", 0),
                "fer_mg":        r.get("fer_mg", 0),
                "vitamine_a_ug": r.get("vitamine_a_ug", 0),
                "zinc_mg":       r.get("zinc_mg", 0),
                "calcium_mg":    r.get("calcium_mg", 0),
                "folate_ug":     r.get("folate_ug", 0),
                "score":         r.get("score_nutritionnel", 60),
                "ingredients":   r.get("ingredients", []),
                "instructions":  r.get("instructions", ""),
                "temps_min":     r.get("temps_preparation_min", 30),
                "cout_ariary":   r.get("cout_estime_ariary", 0),
                "contraintes":   [],
                "culture":       ["toutes"],
                "image_url":     r.get("image_url"),
            })
        return normalized

    @staticmethod
    def _recettes_par_defaut(cible: str, nombre: int) -> List[Dict[str, Any]]:
        """Retourne des recettes de base si toutes les sources échouent."""
        return [
            {
                "recette_id":    f"default-{cible}-{i}",
                "nom":           f"Repas équilibré UNICEF #{i+1}",
                "nom_malgache":  None,
                "region_adaptee": ["toutes"],
                "saison":        ["toute_annee"],
                "calories_kcal": 200.0,
                "proteines_g":   10.0,
                "glucides_g":    30.0,
                "lipides_g":     5.0,
                "fer_mg":        3.0,
                "vitamine_a_ug": 200.0,
                "zinc_mg":       2.0,
                "score_nutritionnel": 65.0,
                "ingredients":   [],
                "instructions":  "Recette à définir selon disponibilité locale.",
                "temps_preparation_min": 20,
                "cout_estime_ariary": 800.0,
                "cible":         [cible, "famille"],
                "image_url":     None,
            }
            for i in range(min(nombre, 3))
        ]

    def _is_real_db(self) -> bool:
        """Vérifie si la session DB est une vraie session SQLAlchemy."""
        if self._db is None:
            return False
        return "Session" in type(self._db).__name__