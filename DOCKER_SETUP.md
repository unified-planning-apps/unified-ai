# HealthShield Backend — Docker Setup

## What you need installed
- **Docker Desktop** (Windows/macOS) or **Docker Engine + Docker Compose plugin** (Linux)
- Nothing else — no Python, no PostgreSQL, no Redis installed locally

---

## Files to place inside the backend folder (`unified-ai/`)

Copy these 3 files from this zip into your `unified-ai/` folder (the same folder that contains `src/`, `scripts/`, `requirements.txt`, etc.):

| File | Replaces / New |
|---|---|
| `Dockerfile` | **Replaces** the existing one (fixes the `uvicorn` entry point — `create_application` is a factory function, not an ASGI app instance; the original would crash on startup) |
| `docker-compose.yml` | **Replaces** the existing one (proper init ordering, seed script, MinIO bucket creation) |
| `.env` | **New** — your environment config (copy from `.env.example` equivalent) |
| `.dockerignore` | **New** — keeps build context lean (excludes `Archive.zip`, notebooks, etc.) |

Your folder structure should look like:

```
unified-ai/
├── .dockerignore       ← new
├── .env                ← new (fill in your API keys)
├── Dockerfile          ← replaced
├── docker-compose.yml  ← replaced
├── config/
├── scripts/
├── src/
├── requirements.txt
└── ...
```

---

## First-time startup

Open a terminal in the `unified-ai/` folder and run:

```bash
docker compose up --build
```

This does everything in order:
1. Builds the Python image from `requirements.txt` (takes ~3–5 min first time, cached after)
2. Starts PostgreSQL, Redis, MinIO
3. Runs `scripts/init_db.py` — creates all DB tables
4. Runs `scripts/seed_regions.py --with-recipes --with-users` — loads 22 Madagascar regions, base recipes, and default accounts (`admin/admin123`, `demo/demo123`)
5. Starts the FastAPI API on **http://localhost:8000**
6. Starts Celery worker + beat for background data ingestion

> **First startup takes a few minutes** because Docker pulls the base images and pip installs ~200 packages. Every subsequent `up` is instant unless `requirements.txt` changes.

---

## Day-to-day commands

```bash
# Start everything (background)
docker compose up -d

# Stop everything (keeps data)
docker compose down

# Wipe everything including database data (full reset)
docker compose down -v

# View live logs from the API
docker compose logs -f api

# View logs from all services
docker compose logs -f

# Restart just the API (after a code change that --reload didn't catch)
docker compose restart api

# Open a Python shell inside the running API container
docker compose exec api python

# Run a one-off script (e.g. re-seed regions)
docker compose exec api python scripts/seed_regions.py --with-recipes
```

---

## Useful URLs once running

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Interactive API docs (Swagger) | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| API health check | http://localhost:8000/health |
| MinIO web console | http://localhost:9001 (user: `minioadmin`, pass: `minioadmin`) |
| PostgreSQL | localhost:5432 (user: `unicef`, pass: `unicef123`, db: `unicef`) |
| Redis | localhost:6379 |

---

## Environment configuration (`.env`)

The `.env` file is already pre-filled with working defaults for everything Docker-internal. The only things you'll want to fill in are the external API keys — the app runs in **degraded mode** without them (no live weather data, no DHIS2 data, but all other endpoints work):

```env
# Fill these in to enable live data ingestion
OPENWEATHER_API_KEY=your_key_here
DHIS2_USERNAME=your_dhis2_user
DHIS2_PASSWORD=your_dhis2_password
```

**Do not change** the PostgreSQL/Redis/MinIO connection strings — they match the Docker service names and will break if you change them to `localhost`.

---

## Default accounts

Created automatically on first startup by `services/auth_service.py::create_default_admin`:

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | admin (full access) |
| `demo` | `demo123` | viewer (read-only) |

Change these immediately for anything beyond local testing.

---

## Connecting the frontend

Set `VITE_API_BASE_URL` in your frontend `.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

The backend CORS is already configured to allow `http://localhost:3000` in development mode.

---

## Troubleshooting

**`db-init` or `db-seed` fails on first run:**
PostgreSQL might not be fully ready yet even though the healthcheck passed. This is rare but can happen under load. Just re-run:
```bash
docker compose up --build
```
(It's safe to re-run — `init_db.py` uses `CREATE TABLE IF NOT EXISTS` and `seed_regions.py` uses upserts.)

**`api` crashes with `ModuleNotFoundError`:**
Make sure `PYTHONPATH=/app` is set. It's hardcoded in the Dockerfile's `ENV` and in `docker-compose.yml`'s `environment` block. If you edited either file, check those values.

**Port already in use:**
Something else is running on 8000, 5432, 6379, or 9000. Stop it, or change the host port mapping in `docker-compose.yml` (e.g. `"8001:8000"` to expose on 8001).

**MinIO bucket errors:**
The `minio-init` service creates the buckets automatically. If you wipe volumes and restart, it re-runs. If you see bucket errors in the API, run:
```bash
docker compose run --rm minio-init
```

**Full reset (start from scratch):**
```bash
docker compose down -v   # removes all volumes (database, redis, minio)
docker compose up --build
```
