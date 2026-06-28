"""
security/password.py
=====================
Hash et vérification des mots de passe avec bcrypt.
"""

from passlib.context import CryptContext

# Contexte bcrypt — standard sécurité industrie
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Retourne le hash bcrypt d'un mot de passe."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie qu'un mot de passe correspond à son hash."""
    return pwd_context.verify(plain_password, hashed_password)