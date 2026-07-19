"""
security/password.py
=====================
Hash et vérification des mots de passe avec bcrypt.

NOTE: On n'utilise PAS passlib ici, car passlib==1.7.4 tente de lire
bcrypt.__about__.__version__ qui a été supprimé dans bcrypt>=4.
Cela provoque une fausse erreur "password cannot be longer than 72 bytes"
même pour un mot de passe court, ce qui empêche la création des comptes
par défaut au démarrage.

On appelle directement l'API bcrypt (hashpw / checkpw / gensalt) qui est
identique entre bcrypt 3.x et bcrypt 5.x — aucun changement de dépendance requis.
"""

import bcrypt


def hash_password(password: str) -> str:
    """Retourne le hash bcrypt d'un mot de passe en clair."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie qu'un mot de passe en clair correspond à son hash bcrypt."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False
