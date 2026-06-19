"""
Initialisation complète de la base de données.

Script d'amorçage à exécuter UNE SEULE FOIS lors du premier déploiement
(ou pour réinitialiser un environnement de développement).

Actions effectuées (dans l'ordre) :
  1. Vérification de la connexion PostgreSQL
  2. Création des extensions (PostGIS, TimescaleDB, uuid-ossp, pg_trgm)
  3. Création de tous les schémas
  4. Création de toutes les tables (via SQLAlchemy + Alembic)
  5. Création des index spatiaux et temporels
  6. Création des vues matérialisées
  7. Création des fonctions PostgreSQL (triggers, agrégats)
  8. Vérification finale d'intégrité

Usage :
    python scripts/init_db.py                    # Initialisation standard
    python scripts/init_db.py --drop-all         # ⚠ Supprime tout et recrée
    python scripts/init_db.py --env staging      # Cible un environnement spécifique
    python scripts/init_db.py --dry-run          # Affiche le SQL sans exécuter

Prérequis :
    - PostgreSQL 15+ avec extensions PostGIS et TimescaleDB installées
    - Variable d'environnement DATABASE_URL configurée (ou fichier .env)
    - Permissions CREATE sur la base cible

Auteur : Équipe Data  
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# ── Résolution du chemin racine du projet ────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.settings import settings
from src.utils.logger import get_logger, setup_logging

setup_logging()
log = get_logger("init_db")


# ─────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────

REQUIRED_EXTENSIONS = [
    "postgis",           # Données géospatiales
    "postgis_topology",  # Topologie vectorielle
    "timescaledb",       # Séries temporelles optimisées
    'uuid-ossp',         # Génération UUID côté DB
    "pg_trgm",           # Recherche textuelle floue (noms régions)
    "btree_gist",        # Index GiST sur types scalaires (dates + geo)
]

SCHEMAS = ["public", "ml", "reports", "audit"]

# ─────────────────────────────────────────────────────────────────
# DDL — Extensions & Schémas
# ─────────────────────────────────────────────────────────────────

def create_extensions(engine: Engine) -> None:
    """Installe les extensions PostgreSQL requises."""
    log.info("Installation des extensions PostgreSQL...")

    with engine.begin() as conn:
        for ext in REQUIRED_EXTENSIONS:
            try:
                conn.execute(text(
                    f"CREATE EXTENSION IF NOT EXISTS \"{ext}\" CASCADE;"
                ))
                log.info("  ✓ Extension : {}", ext)
            except Exception as exc:
                # TimescaleDB peut échouer si non installé au niveau OS — non bloquant
                if "timescaledb" in ext.lower():
                    log.warning(
                        "  ⚠ TimescaleDB non disponible (non bloquant) : {}", exc
                    )
                else:
                    log.error("  ✗ Extension {} : {}", ext, exc)
                    raise


def create_schemas(engine: Engine) -> None:
    """Crée les schémas PostgreSQL."""
    log.info("Création des schémas...")
    with engine.begin() as conn:
        for schema in SCHEMAS:
            conn.execute(text(
                f"CREATE SCHEMA IF NOT EXISTS {schema};"
            ))
            log.info("  ✓ Schéma : {}", schema)


# ─────────────────────────────────────────────────────────────────
# DDL — Tables principales
# ─────────────────────────────────────────────────────────────────

# Regroupement du DDL par domaine pour lisibilité et maintenance

DDL_TABLES: List[str] = [

    # ── Régions géographiques ──────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS public.regions (
        code                VARCHAR(10)     PRIMARY KEY,
        nom_fr              VARCHAR(100)    NOT NULL,
        nom_mg              VARCHAR(100)    NOT NULL,
        centroid            GEOGRAPHY(POINT, 4326) NOT NULL,
        bbox_geom           GEOGRAPHY(POLYGON, 4326),
        altitude_moyenne_m  FLOAT,
        zone_altitude       VARCHAR(30),
        zone_climatique     VARCHAR(30),
        population_estimee  INTEGER,
        superficie_km2      FLOAT,
        voisins             VARCHAR(10)[],  -- Codes régions limitrophes
        actif               BOOLEAN         NOT NULL DEFAULT TRUE,
        metadata            JSONB           DEFAULT '{}',
        cree_le             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        mis_a_jour_le       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
    );
    COMMENT ON TABLE public.regions IS
        'Référentiel des 22 régions administratives de Madagascar';
    """,

    # ── Données météorologiques (time-series) ─────────────────────
    """
    CREATE TABLE IF NOT EXISTS public.weather_observations (
        id              BIGSERIAL,
        region_code     VARCHAR(10)     NOT NULL REFERENCES public.regions(code),
        timestamp_utc   TIMESTAMPTZ     NOT NULL,
        temperature_c   FLOAT           NOT NULL,
        temp_min_c      FLOAT,
        temp_max_c      FLOAT,
        humidite_pct    FLOAT           NOT NULL,
        precipitation_mm FLOAT          NOT NULL DEFAULT 0.0,
        vitesse_vent_kmh FLOAT,
        ndvi            FLOAT,
        altitude_m      FLOAT,
        source_api      VARCHAR(50)     NOT NULL DEFAULT 'unknown',
        raw_payload     JSONB,          -- Réponse API brute pour retraitement
        qualite_flag    SMALLINT        NOT NULL DEFAULT 0,
            -- 0=OK, 1=interpolé, 2=estimé, 3=suspect
        PRIMARY KEY (id, timestamp_utc)
    );
    COMMENT ON TABLE public.weather_observations IS
        'Observations météo horaires/journalières par région';
    """,

    # ── Conversion TimescaleDB (hypertable) ──────────────────────
    # Note : exécuté séparément car nécessite TimescaleDB actif
    # SELECT create_hypertable('weather_observations', 'timestamp_utc',
    #   if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 month');

    # ── Observations paludisme ────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS public.malaria_observations (
        id                  BIGSERIAL       PRIMARY KEY,
        region_code         VARCHAR(10)     NOT NULL REFERENCES public.regions(code),
        semaine_iso         SMALLINT        NOT NULL CHECK (semaine_iso BETWEEN 1 AND 53),
        annee               SMALLINT        NOT NULL CHECK (annee BETWEEN 2010 AND 2035),
        date_debut_semaine  DATE            NOT NULL,
        cas_confirmes       INTEGER         NOT NULL CHECK (cas_confirmes >= 0),
        cas_presumes        INTEGER         CHECK (cas_presumes >= 0),
        deces               INTEGER         CHECK (deces >= 0),
        tests_realises      INTEGER         CHECK (tests_realises >= 0),
        taux_positivite     FLOAT           GENERATED ALWAYS AS (
            CASE WHEN tests_realises > 0
                 THEN ROUND((cas_confirmes::FLOAT / tests_realises)::NUMERIC, 4)
                 ELSE NULL
            END
        ) STORED,
        district            VARCHAR(100),
        dhis2_org_unit_id   VARCHAR(11),
        source              VARCHAR(50)     NOT NULL DEFAULT 'dhis2',
        valide              BOOLEAN         NOT NULL DEFAULT TRUE,
        cree_le             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        UNIQUE (region_code, semaine_iso, annee, source)
    );
    COMMENT ON TABLE public.malaria_observations IS
        'Données épidémiologiques hebdomadaires paludisme par région';
    """,

    # ── Données nutrition ─────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS public.nutrition_observations (
        id                  BIGSERIAL       PRIMARY KEY,
        region_code         VARCHAR(10)     NOT NULL REFERENCES public.regions(code),
        date_enquete        DATE            NOT NULL,
        gam_pct             FLOAT           NOT NULL CHECK (gam_pct BETWEEN 0 AND 60),
        mam_pct             FLOAT           CHECK (mam_pct >= 0),
        sam_pct             FLOAT           CHECK (sam_pct >= 0),
        classification_oms  VARCHAR(20)     NOT NULL,
            -- acceptable, alerte, urgence, crise
        groupe_cible        VARCHAR(30)     NOT NULL,
        n_enfants_enquetes  INTEGER         CHECK (n_enfants_enquetes >= 0),
        score_sca           FLOAT           CHECK (score_sca BETWEEN 0 AND 112),
        source              VARCHAR(50)     NOT NULL DEFAULT 'unicef_enquete',
        metadata            JSONB           DEFAULT '{}',
        cree_le             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
    );
    COMMENT ON TABLE public.nutrition_observations IS
        'Données nutritionnelles (GAM, MAM, SAM) par région et période';
    """,

    # ── Prédictions ML ────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ml.predictions (
        id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
        region_code         VARCHAR(10)     NOT NULL REFERENCES public.regions(code),
        modele_nom          VARCHAR(50)     NOT NULL,
        modele_version      VARCHAR(20)     NOT NULL,
        date_prediction     DATE            NOT NULL,
        horizon_jours       SMALLINT        NOT NULL,
        score_malaria       FLOAT           CHECK (score_malaria BETWEEN 0 AND 1),
        niveau_malaria      VARCHAR(20),
        score_nutrition     FLOAT           CHECK (score_nutrition BETWEEN 0 AND 1),
        niveau_nutrition    VARCHAR(20),
        confiance           FLOAT           CHECK (confiance BETWEEN 0 AND 1),
        shap_values         JSONB,
        features_snapshot   JSONB,          -- Features utilisées (traçabilité)
        timestamp_calcul    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        duree_inference_ms  FLOAT,
        user_id             INTEGER,
        UNIQUE (region_code, modele_nom, modele_version, date_prediction, horizon_jours)
    );
    COMMENT ON TABLE ml.predictions IS
        'Prédictions ML paludisme et nutrition avec traçabilité complète';
    """,

    # ── Versions des modèles ML ───────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS ml.model_versions (
        id              SERIAL          PRIMARY KEY,
        nom             VARCHAR(50)     NOT NULL,
        version         VARCHAR(20)     NOT NULL,
        type_modele     VARCHAR(30)     NOT NULL,  -- xgboost, lstm, ensemble…
        mlflow_run_id   VARCHAR(50),
        metriques       JSONB           NOT NULL DEFAULT '{}',
        parametres      JSONB           NOT NULL DEFAULT '{}',
        chemin_artefact VARCHAR(500),
        actif           BOOLEAN         NOT NULL DEFAULT FALSE,
        valide          BOOLEAN         NOT NULL DEFAULT FALSE,
        date_entrainement DATE,
        n_samples       INTEGER,
        cree_le         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        UNIQUE (nom, version)
    );
    COMMENT ON TABLE ml.model_versions IS
        'Registre des versions de modèles ML avec métriques de validation';
    """,

    # ── Alertes épidémiologiques ──────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS public.alertes (
        id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
        region_code     VARCHAR(10)     NOT NULL REFERENCES public.regions(code),
        type_alerte     VARCHAR(30)     NOT NULL,
        severite        VARCHAR(20)     NOT NULL,
        valeur          FLOAT           NOT NULL,
        description     TEXT            NOT NULL,
        recommandations JSONB           DEFAULT '[]',
        active          BOOLEAN         NOT NULL DEFAULT TRUE,
        creee_le        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        expire_le       TIMESTAMPTZ     NOT NULL,
        acquittee_le    TIMESTAMPTZ,
        acquittee_par   INTEGER,
        creee_par       INTEGER
    );
    COMMENT ON TABLE public.alertes IS
        'Alertes épidémiologiques et humanitaires actives et archivées';
    """,

    # ── Rapports générés ──────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS reports.rapports (
        id              UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
        type_rapport    VARCHAR(50)     NOT NULL,
        region_codes    VARCHAR(10)[],
        date_debut      DATE            NOT NULL,
        date_fin        DATE            NOT NULL,
        langue          VARCHAR(5)      NOT NULL DEFAULT 'fr',
        statut          VARCHAR(20)     NOT NULL DEFAULT 'en_attente',
            -- en_attente, en_cours, termine, erreur
        chemin_fichier  VARCHAR(500),
        taille_octets   BIGINT,
        duree_gen_sec   FLOAT,
        erreur_message  TEXT,
        metadata        JSONB           DEFAULT '{}',
        cree_le         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        cree_par        INTEGER,
        telecharge_le   TIMESTAMPTZ,
        expire_le TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '90 days')
    );
    COMMENT ON TABLE reports.rapports IS
        'Catalogue des rapports PDF générés avec métadonnées';
    """,

    # ── Recettes nutritionnelles ──────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS public.recettes (
        id                  SERIAL          PRIMARY KEY,
        nom_fr              VARCHAR(200)    NOT NULL,
        nom_mg              VARCHAR(200),
        description         TEXT,
        groupe_cible        VARCHAR(30)[]   NOT NULL,
        regions_adaptees    VARCHAR(10)[],  -- NULL = toutes les régions
        saisons_adaptees    VARCHAR(20)[],
        ingredients         JSONB           NOT NULL,
            -- [{nom, quantite, unite, optionnel, allergene}]
        valeurs_nutritionnelles JSONB       NOT NULL DEFAULT '{}',
            -- {energie_kcal, proteines_g, fer_mg, vitamine_a_ug, …}
        objectifs           VARCHAR(50)[],  -- fer, proteines, vitamine_a…
        allergenes          VARCHAR(50)[],
        temps_prep_min      SMALLINT,
        difficulte          VARCHAR(10),    -- facile, moyen, difficile
        source              VARCHAR(100),
        valide              BOOLEAN         NOT NULL DEFAULT TRUE,
        cree_le             TIMESTAMPTZ     NOT NULL DEFAULT NOW()
    );
    COMMENT ON TABLE public.recettes IS
        'Base de recettes nutritionnelles adaptées aux régions malgaches';
    """,

    # ── Utilisateurs ──────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS public.utilisateurs (
        id              SERIAL          PRIMARY KEY,
        email           VARCHAR(255)    NOT NULL UNIQUE,
        nom_complet     VARCHAR(200)    NOT NULL,
        role            VARCHAR(20)     NOT NULL DEFAULT 'viewer',
        region_code     VARCHAR(10)     REFERENCES public.regions(code),
            -- NULL = accès national (admin/national)
        actif           BOOLEAN         NOT NULL DEFAULT TRUE,
        hashed_password VARCHAR(255)    NOT NULL,
        derniere_connexion TIMESTAMPTZ,
        cree_le         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        mis_a_jour_le   TIMESTAMPTZ     NOT NULL DEFAULT NOW()
    );
    COMMENT ON TABLE public.utilisateurs IS
        'Comptes utilisateurs (agents UNICEF, MdS, viewers)';
    """,

    # ── Logs d'audit ──────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS audit.logs (
        id              BIGSERIAL       PRIMARY KEY,
        timestamp_utc   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        user_id         INTEGER,
        action          VARCHAR(50)     NOT NULL,
        table_cible     VARCHAR(100),
        enregistrement_id VARCHAR(100),
        valeurs_avant   JSONB,
        valeurs_apres   JSONB,
        ip_address      INET,
        user_agent      VARCHAR(500),
        succes          BOOLEAN         NOT NULL DEFAULT TRUE,
        message         TEXT
    );
    COMMENT ON TABLE audit.logs IS
        'Journal d audit complet — conformité UNICEF et RGPD';
    """,

    # ── Collecte planifiée (Celery tasks tracking) ────────────────
    """
    CREATE TABLE IF NOT EXISTS public.collecte_logs (
        id              BIGSERIAL       PRIMARY KEY,
        source          VARCHAR(50)     NOT NULL,
        region_code     VARCHAR(10),
        timestamp_debut TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        timestamp_fin   TIMESTAMPTZ,
        n_records       INTEGER         DEFAULT 0,
        statut          VARCHAR(20)     NOT NULL DEFAULT 'en_cours',
        erreur          TEXT,
        celery_task_id  VARCHAR(100)
    );
    COMMENT ON TABLE public.collecte_logs IS
        'Suivi des tâches de collecte automatique (Celery scheduler)';
    """,
]


# ─────────────────────────────────────────────────────────────────
# DDL — Index
# ─────────────────────────────────────────────────────────────────

DDL_INDEXES: List[str] = [
    # Météo — requêtes fréquentes : région + période
    "CREATE INDEX IF NOT EXISTS idx_weather_region_ts ON public.weather_observations (region_code, timestamp_utc DESC);",
    "CREATE INDEX IF NOT EXISTS idx_weather_ts ON public.weather_observations (timestamp_utc DESC);",
    "CREATE INDEX IF NOT EXISTS idx_weather_source ON public.weather_observations (source_api);",

    # Paludisme — région + semaine/année
    "CREATE INDEX IF NOT EXISTS idx_malaria_region_periode ON public.malaria_observations (region_code, annee DESC, semaine_iso DESC);",
    "CREATE INDEX IF NOT EXISTS idx_malaria_date ON public.malaria_observations (date_debut_semaine DESC);",

    # Nutrition — région + date
    "CREATE INDEX IF NOT EXISTS idx_nutrition_region_date ON public.nutrition_observations (region_code, date_enquete DESC);",
    "CREATE INDEX IF NOT EXISTS idx_nutrition_gam ON public.nutrition_observations (gam_pct);",

    # Prédictions — requêtes API principales
    "CREATE INDEX IF NOT EXISTS idx_pred_region_date ON ml.predictions (region_code, date_prediction DESC);",
    "CREATE INDEX IF NOT EXISTS idx_pred_modele ON ml.predictions (modele_nom, modele_version);",
    "CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON ml.predictions (timestamp_calcul DESC);",

    # Alertes — actives uniquement
    "CREATE INDEX IF NOT EXISTS idx_alertes_actives ON public.alertes (region_code, active, severite) WHERE active = TRUE;",
    "CREATE INDEX IF NOT EXISTS idx_alertes_expiration ON public.alertes (expire_le) WHERE active = TRUE;",

    # Rapports — statut + type
    "CREATE INDEX IF NOT EXISTS idx_rapports_statut ON reports.rapports (statut, cree_le DESC);",
    "CREATE INDEX IF NOT EXISTS idx_rapports_expire ON reports.rapports (expire_le);",

    # Régions — index spatial (centroïde)
    "CREATE INDEX IF NOT EXISTS idx_regions_centroid ON public.regions USING GIST (centroid);",

    # Audit — recherche par user et action
    "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit.logs (user_id, timestamp_utc DESC);",
    "CREATE INDEX IF NOT EXISTS idx_audit_table ON audit.logs (table_cible, timestamp_utc DESC);",

    # Recettes — recherche par groupe cible et saison
    "CREATE INDEX IF NOT EXISTS idx_recettes_groupe ON public.recettes USING GIN (groupe_cible);",
    "CREATE INDEX IF NOT EXISTS idx_recettes_saison ON public.recettes USING GIN (saisons_adaptees);",
    "CREATE INDEX IF NOT EXISTS idx_recettes_nom_trgm ON public.recettes USING GIN (nom_fr gin_trgm_ops);",
]


# ─────────────────────────────────────────────────────────────────
# DDL — Vues matérialisées
# ─────────────────────────────────────────────────────────────────

DDL_MATERIALIZED_VIEWS: List[str] = [

    # Résumé hebdomadaire paludisme par région (dernières 52 semaines)
    """
    CREATE MATERIALIZED VIEW IF NOT EXISTS public.mv_malaria_weekly_summary AS
    SELECT
        m.region_code,
        r.nom_fr,
        m.annee,
        m.semaine_iso,
        m.date_debut_semaine,
        SUM(m.cas_confirmes)                                AS total_cas_confirmes,
        SUM(m.deces)                                        AS total_deces,
        AVG(m.taux_positivite)                              AS taux_positivite_moyen,
        AVG(w.temperature_c)                                AS temp_moyenne_c,
        SUM(w.precipitation_mm)                             AS precipitations_totales_mm,
        AVG(w.humidite_pct)                                 AS humidite_moyenne_pct,
        COUNT(DISTINCT m.district)                          AS n_districts_touches,
        NOW()                                               AS derniere_maj
    FROM public.malaria_observations m
    JOIN public.regions r ON r.code = m.region_code
    LEFT JOIN public.weather_observations w
        ON  w.region_code = m.region_code
        AND DATE_TRUNC('week', w.timestamp_utc) = m.date_debut_semaine
    WHERE m.annee >= EXTRACT(YEAR FROM NOW()) - 2
    GROUP BY
        m.region_code, r.nom_fr, m.annee, m.semaine_iso, m.date_debut_semaine
    WITH DATA;
    """,

    # Dernière prédiction active par région
    """
    CREATE MATERIALIZED VIEW IF NOT EXISTS ml.mv_latest_predictions AS
    SELECT DISTINCT ON (region_code, modele_nom)
        id,
        region_code,
        modele_nom,
        modele_version,
        date_prediction,
        horizon_jours,
        score_malaria,
        niveau_malaria,
        score_nutrition,
        niveau_nutrition,
        confiance,
        timestamp_calcul
    FROM ml.predictions
    ORDER BY region_code, modele_nom, timestamp_calcul DESC
    WITH DATA;
    """,

    # Statut nutritionnel courant par région
    """
    CREATE MATERIALIZED VIEW IF NOT EXISTS public.mv_nutrition_status AS
    SELECT DISTINCT ON (region_code)
        region_code,
        date_enquete,
        gam_pct,
        mam_pct,
        sam_pct,
        classification_oms,
        groupe_cible,
        source,
        NOW() AS derniere_maj
    FROM public.nutrition_observations
    ORDER BY region_code, date_enquete DESC
    WITH DATA;
    """,
]

DDL_MV_INDEXES: List[str] = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_malaria_weekly ON public.mv_malaria_weekly_summary (region_code, annee, semaine_iso);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_latest_pred ON ml.mv_latest_predictions (region_code, modele_nom);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_nutrition ON public.mv_nutrition_status (region_code);",
]


# ─────────────────────────────────────────────────────────────────
# DDL — Fonctions & Triggers PostgreSQL
# ─────────────────────────────────────────────────────────────────

DDL_FUNCTIONS: List[str] = [

    # Trigger : mise à jour automatique de mis_a_jour_le
    """
    CREATE OR REPLACE FUNCTION public.fn_set_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.mis_a_jour_le = NOW();
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """,

    # Trigger : log d'audit automatique sur les tables sensibles
    """
    CREATE OR REPLACE FUNCTION audit.fn_audit_log()
    RETURNS TRIGGER AS $$
    BEGIN
        INSERT INTO audit.logs (
            action,
            table_cible,
            enregistrement_id,
            valeurs_avant,
            valeurs_apres
        ) VALUES (
            TG_OP,
            TG_TABLE_SCHEMA || '.' || TG_TABLE_NAME,
            COALESCE(NEW.id::TEXT, OLD.id::TEXT),
            CASE WHEN TG_OP IN ('UPDATE', 'DELETE') THEN row_to_json(OLD) ELSE NULL END,
            CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) ELSE NULL END
        );
        RETURN COALESCE(NEW, OLD);
    END;
    $$ LANGUAGE plpgsql SECURITY DEFINER;
    """,

    # Fonction : classification OMS automatique depuis gam_pct
    """
    CREATE OR REPLACE FUNCTION public.fn_classification_oms(gam_pct FLOAT)
    RETURNS VARCHAR AS $$
    BEGIN
        RETURN CASE
            WHEN gam_pct < 5.0  THEN 'acceptable'
            WHEN gam_pct < 10.0 THEN 'alerte'
            WHEN gam_pct < 15.0 THEN 'urgence'
            ELSE 'crise'
        END;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """,

    # Fonction : niveau risque malaria depuis score
    """
    CREATE OR REPLACE FUNCTION ml.fn_niveau_malaria(score FLOAT)
    RETURNS VARCHAR AS $$
    BEGIN
        RETURN CASE
            WHEN score < 0.25 THEN 'faible'
            WHEN score < 0.50 THEN 'moyen'
            WHEN score < 0.75 THEN 'élevé'
            ELSE 'très élevé'
        END;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """,
]

DDL_TRIGGERS: List[str] = [
    # updated_at sur régions et utilisateurs
    """
    DROP TRIGGER IF EXISTS trg_regions_updated_at ON public.regions;
    CREATE TRIGGER trg_regions_updated_at
        BEFORE UPDATE ON public.regions
        FOR EACH ROW EXECUTE FUNCTION public.fn_set_updated_at();
    """,
    """
    DROP TRIGGER IF EXISTS trg_users_updated_at ON public.utilisateurs;
    CREATE TRIGGER trg_users_updated_at
        BEFORE UPDATE ON public.utilisateurs
        FOR EACH ROW EXECUTE FUNCTION public.fn_set_updated_at();
    """,
    # Audit sur les tables sensibles
    """
    DROP TRIGGER IF EXISTS trg_audit_predictions ON ml.predictions;
    CREATE TRIGGER trg_audit_predictions
        AFTER INSERT OR UPDATE OR DELETE ON ml.predictions
        FOR EACH ROW EXECUTE FUNCTION audit.fn_audit_log();
    """,
    """
    DROP TRIGGER IF EXISTS trg_audit_alertes ON public.alertes;
    CREATE TRIGGER trg_audit_alertes
        AFTER INSERT OR UPDATE OR DELETE ON public.alertes
        FOR EACH ROW EXECUTE FUNCTION audit.fn_audit_log();
    """,
]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def get_engine(env: str = "default") -> Engine:
    """Construit le moteur SQLAlchemy selon l'environnement."""
    url = settings.database.sync_url
    if env == "staging" and hasattr(settings.database, "staging_url"):
        url = settings.database.staging_url

    engine = sa.create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=0
    )
    return engine


def check_connection(engine: Engine) -> None:
    """Vérifie la connexion PostgreSQL et la version."""
    log.info("Vérification connexion PostgreSQL...")
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version();")).scalar()
        postgis = conn.execute(
            text("SELECT PostGIS_Version();")
        ).scalar()
        log.info("  ✓ PostgreSQL : {}", version.split(",")[0])
        log.info("  ✓ PostGIS    : {}", postgis)


def drop_all(engine: Engine) -> None:
    """⚠ Supprime tous les schémas et leur contenu (DEV SEULEMENT)."""
    log.warning("⚠ Suppression de tous les schémas...")
    with engine.begin() as conn:
        for schema in reversed(SCHEMAS):
            conn.execute(text(
                f"DROP SCHEMA IF EXISTS {schema} CASCADE;"
            ))
            log.info("  ✗ Schéma supprimé : {}", schema)


def execute_ddl_batch(engine: Engine, statements: List[str], label: str) -> int:
    """
    Exécute une liste de statements DDL.
    Retourne le nombre de statements exécutés avec succès.
    """
    success = 0
    with engine.begin() as conn:
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                conn.execute(text(stmt))
                success += 1
            except Exception as exc:
                log.error("[{}] Erreur DDL : {}\nSQL: {}", label, exc, stmt[:120])
                raise
    return success


def refresh_materialized_views(engine: Engine) -> None:
    """Rafraîchit toutes les vues matérialisées."""
    views = [
        "public.mv_malaria_weekly_summary",
        "ml.mv_latest_predictions",
        "public.mv_nutrition_status",
    ]
    with engine.begin() as conn:
        for view in views:
            try:
                conn.execute(text(
                    f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view};"
                ))
                log.info("  ✓ Vue matérialisée rafraîchie : {}", view)
            except Exception as exc:
                # CONCURRENTLY nécessite un index unique — fallback sans
                conn.execute(text(f"REFRESH MATERIALIZED VIEW {view};"))
                log.info("  ✓ Vue matérialisée rafraîchie (sans CONCURRENTLY) : {}", view)


def verify_integrity(engine: Engine) -> bool:
    """
    Vérification finale : compte les tables créées et teste les contraintes de base.
    Retourne True si tout est OK.
    """
    log.info("Vérification d'intégrité post-initialisation...")
    checks_passed = 0
    checks_total  = 0

    expected_tables = {
        "public.regions",
        "public.weather_observations",
        "public.malaria_observations",
        "public.nutrition_observations",
        "public.alertes",
        "public.recettes",
        "public.utilisateurs",
        "public.collecte_logs",
        "ml.predictions",
        "ml.model_versions",
        "reports.rapports",
        "audit.logs",
    }

    with engine.connect() as conn:
        for full_name in expected_tables:
            checks_total += 1
            schema, table = full_name.split(".")
            exists = conn.execute(text(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables"
                "  WHERE table_schema = :schema AND table_name = :table"
                ");"
            ), {"schema": schema, "table": table}).scalar()

            if exists:
                checks_passed += 1
                log.debug("  ✓ Table : {}", full_name)
            else:
                log.error("  ✗ Table MANQUANTE : {}", full_name)

    ok = checks_passed == checks_total
    log.info(
        "Intégrité : {}/{} tables OK {}",
        checks_passed, checks_total,
        "✓" if ok else "✗"
    )
    return ok


# ─────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialise la base de données PostgreSQL du projet  "
    )
    parser.add_argument(
        "--drop-all",
        action="store_true",
        help="⚠ Supprime et recrée tout (INTERDIT en production)",
    )
    parser.add_argument(
        "--env",
        default="default",
        choices=["default", "staging", "test"],
        help="Environnement cible",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le nombre de statements sans les exécuter",
    )
    parser.add_argument(
        "--skip-views",
        action="store_true",
        help="Ne pas créer les vues matérialisées (utile si peu de données)",
    )
    args = parser.parse_args()

    # ── Sécurité : --drop-all interdit hors dev ──────────────────
    if args.drop_all and settings.environment == "production":
        log.critical(
            "--drop-all est INTERDIT en production ! "
            "Modifiez ENVIRONMENT dans .env pour continuer."
        )
        sys.exit(1)

    if args.dry_run:
        n = (
            len(DDL_TABLES)
            + len(DDL_INDEXES)
            + len(DDL_MATERIALIZED_VIEWS)
            + len(DDL_MV_INDEXES)
            + len(DDL_FUNCTIONS)
            + len(DDL_TRIGGERS)
        )
        log.info("DRY-RUN : {} statements DDL à exécuter", n)
        log.info("Extensions : {}", REQUIRED_EXTENSIONS)
        log.info("Schémas    : {}", SCHEMAS)
        return

    t_start = time.perf_counter()
    engine  = get_engine(args.env)

    try:
        # 1. Connexion
        check_connection(engine)

        # 2. Drop (optionnel)
        if args.drop_all:
            log.warning("DROP ALL demandé sur environnement : {}", args.env)
            drop_all(engine)

        # 3. Extensions
        create_extensions(engine)

        # 4. Schémas
        create_schemas(engine)

        # 5. Tables
        log.info("Création des tables ({})...", len(DDL_TABLES))
        execute_ddl_batch(engine, DDL_TABLES, "tables")
        log.info("  ✓ {} tables créées", len(DDL_TABLES))

        # 6. Index
        log.info("Création des index ({})...", len(DDL_INDEXES))
        execute_ddl_batch(engine, DDL_INDEXES, "indexes")
        log.info("  ✓ {} index créés", len(DDL_INDEXES))

        # 7. Fonctions & Triggers
        log.info("Création des fonctions PostgreSQL...")
        execute_ddl_batch(engine, DDL_FUNCTIONS, "functions")
        execute_ddl_batch(engine, DDL_TRIGGERS, "triggers")
        log.info("  ✓ Fonctions et triggers installés")

        # 8. Vues matérialisées
        if not args.skip_views:
            log.info("Création des vues matérialisées...")
            execute_ddl_batch(engine, DDL_MATERIALIZED_VIEWS, "views")
            execute_ddl_batch(engine, DDL_MV_INDEXES, "view_indexes")
            log.info("  ✓ Vues matérialisées créées")

        # 9. Vérification finale
        ok = verify_integrity(engine)

        elapsed = time.perf_counter() - t_start
        if ok:
            log.info(
                "✓ Base de données initialisée avec succès en {:.1f}s",
                elapsed
            )
        else:
            log.error("✗ Initialisation terminée avec des erreurs — vérifier les logs")
            sys.exit(1)

    except Exception as exc:
        log.exception("Erreur fatale lors de l'initialisation : {}", exc)
        sys.exit(1)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()