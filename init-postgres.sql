-- =============================================================================
-- init-postgres.sql
-- Runs automatically when the PostgreSQL container first starts
-- (via /docker-entrypoint-initdb.d/). Guarantees all extensions are
-- in place BEFORE init_db.py connects, so every
-- "CREATE EXTENSION IF NOT EXISTS ..." in Python becomes a true no-op
-- and never aborts its transaction.
-- =============================================================================

-- pg_trgm: fuzzy text search (region names) — available in standard postgres
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- uuid-ossp: server-side UUID generation — available in standard postgres
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- btree_gist: GiST index on scalar types — available in standard postgres
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- postgis: geospatial types — available in postgis/postgis image
-- Wrapped in a DO block so it fails gracefully if somehow not present.
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS postgis CASCADE;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'postgis skipped: %', SQLERRM;
END
$$;

-- postgis_topology: vector topology (optional companion to postgis)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS postgis_topology CASCADE;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'postgis_topology skipped: %', SQLERRM;
END
$$;
