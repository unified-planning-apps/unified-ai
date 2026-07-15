FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# System deps: gcc for native extensions, libpq-dev for psycopg2, curl for healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cached unless requirements change)
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . /app

# Pre-create dirs the app writes to at runtime
RUN mkdir -p /tmp/reports /var/log

EXPOSE 8000

# create_application() is a factory — call it as a factory function, not as an ASGI app directly.
# uvicorn's factory mode (--factory) is the correct way to run FastAPI app factories.
CMD ["uvicorn", "src.api.main:create_application", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
