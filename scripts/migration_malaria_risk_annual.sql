-- ─────────────────────────────────────────────────────────────────
-- Migration : table dédiée au risque paludisme ANNUEL par région
--
-- Contexte : malaria_observations est conçue pour du hebdomadaire
-- (semaine_iso, date_debut_semaine NOT NULL, partitionnée dessus) et sa
-- seule vraie source actuelle (who_gho_distribue) n'a pas de vraie
-- variation régionale — juste une incidence nationale redistribuée.
--
-- Le Malaria Atlas Project (MAP) donne une vraie variation régionale,
-- mais annuelle. Plutôt que de forcer une donnée annuelle dans un schéma
-- hebdomadaire (semaine_iso factice, etc.), on crée une table dédiée.
-- ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.malaria_risk_annual (
    id                   BIGSERIAL PRIMARY KEY,
    region_code          VARCHAR(20)  NOT NULL REFERENCES public.regions(code),
    annee                SMALLINT     NOT NULL CHECK (annee >= 1990 AND annee <= 2035),
    incidence_pour_mille NUMERIC(8,4),   -- Cases per Thousand (MAP: Incidence Rate)
    mortalite_pour_100k  NUMERIC(8,4),   -- MAP: Mortality Rate (déjà pour 100k)
    prevalence_pct       NUMERIC(6,3),   -- MAP: Infection Prevalence (%)
    source               VARCHAR(50)  NOT NULL DEFAULT 'malaria_atlas_project',
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT uq_malaria_risk_annual UNIQUE (region_code, annee, source)
);

CREATE INDEX IF NOT EXISTS idx_malaria_risk_annual_region_annee
    ON public.malaria_risk_annual (region_code, annee DESC);

-- Vérification après migration :
--   \d malaria_risk_annual