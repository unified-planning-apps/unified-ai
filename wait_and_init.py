"""
wait_and_init.py
-----------------
Retry wrapper for scripts/init_db.py.

Why this exists: even after the postgres healthcheck passes (pg_isready +
SELECT 1), asyncpg can still raise CannotConnectNowError for a brief window
while PostGIS finishes loading its extensions. A shell retry loop in the
docker-compose YAML causes "Syntax error: end of file unexpected" because
Docker Compose interprets $$ before passing to sh, mangling the script.
This Python wrapper avoids that entirely.
"""
import subprocess
import sys
import time

MAX_ATTEMPTS = 6
DELAY_SECONDS = 5

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"[wait_and_init] attempt {attempt}/{MAX_ATTEMPTS}", flush=True)
    result = subprocess.run(
        [sys.executable, "scripts/init_db.py"],
        cwd="/app",
    )
    if result.returncode == 0:
        print("[wait_and_init] init_db.py succeeded.", flush=True)
        sys.exit(0)

    if attempt < MAX_ATTEMPTS:
        print(
            f"[wait_and_init] init_db.py exited {result.returncode}. "
            f"Retrying in {DELAY_SECONDS}s…",
            flush=True,
        )
        time.sleep(DELAY_SECONDS)

print(f"[wait_and_init] init_db.py failed after {MAX_ATTEMPTS} attempts.", flush=True)
sys.exit(1)
