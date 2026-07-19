#   Data Platform

## Overview

The **  Data Platform** is a data collection, processing, and analytics platform designed to support **malaria surveillance** and **climate monitoring** in Madagascar.

The platform aggregates meteorological and epidemiological data from multiple sources, stores them in PostgreSQL, processes them asynchronously with Celery, and exposes a REST API through FastAPI.

It is designed for:

- Malaria surveillance
- Climate monitoring
- Early warning systems
- Epidemiological analytics
- Machine Learning data preparation
- Dashboard integration

---

# Features

- FastAPI REST API
- JWT Authentication & Authorization
- PostgreSQL database
- Redis cache
- Celery Worker
- Celery Beat Scheduler
- MinIO Object Storage
- OpenWeatherMap data collection
- DHIS2 malaria data collection
- Historical weather aggregation
- Climate indices
- Weather anomaly detection
- National and regional malaria statistics
- Machine Learning ready datasets

---

# Technology Stack

- Python 3.12
- FastAPI
- SQLAlchemy Async
- PostgreSQL
- Redis
- Celery
- MinIO
- Docker
- Docker Compose
- Pydantic
- Alembic (optional)

---

# Project Structure

```
src/
│
├── api/
│   ├── main.py
│   ├── middleware/
│   ├── dependencies.py
│   └── routers/
│
├── config/
│
├── data_collection/
│   ├── weather_fetcher.py
│   ├── malaria_fetcher.py
│   ├── scheduler.py
│   └── celery_app.py
│
├── database/
│   ├── database.py
│   ├── models.py
│   ├── repositories/
│   └── seed/
│
├── preprocessing/
│
├── services/
│
├── security/
│   ├── jwt.py
│   └── password.py
│
└── utils/

scripts/
    init_db.py

Dockerfile
docker-compose.yml
requirements.txt
README.md
```

---

# Requirements

- Docker
- Docker Compose

Optional:

- Python 3.12+
- PostgreSQL
- Redis

---

# Quick Start

## Clone repository

```bash
git clone https://github.com/<your-org>/unicef-data-platform.git

cd unicef-data-platform
```

---

## Build containers

```bash
docker compose build
```

---

## Start all services

```bash
docker compose up
```

---

## View logs

```bash
docker compose logs -f api
```

---

## Initialize database

```bash
docker compose run --rm db-init
```

---

# Services

## API

Runs the FastAPI REST application.

Default URL:

```
http://localhost:8000
```

Swagger UI:

```
http://localhost:8000/docs
```

ReDoc:

```
http://localhost:8000/redoc
```

---

## Celery Worker

Processes asynchronous tasks such as:

- Weather collection
- Malaria data collection
- Notifications
- ML preprocessing

Run:

```bash
docker compose up celery-worker
```

---

## Celery Beat

Schedules periodic tasks.

Example jobs:

- Weather updates
- DHIS2 synchronization
- Alert generation

Run:

```bash
docker compose up celery-beat
```

---

## PostgreSQL

Database server.

Default configuration:

```
Host: postgres
Port: 5432

Database:
unicef

Username:
unicef

Password:
unicef123
```

---

## Redis

Used for:

- Cache
- Celery Broker
- Celery Result Backend

```
redis://redis:6379
```

---

## MinIO

Object storage.

API:

```
http://localhost:9000
```

Console:

```
http://localhost:9001
```

Credentials:

```
Access Key:
minioadmin

Secret Key:
minioadmin
```

---

## Database Initializer

Runs

```
scripts/init_db.py
```

Creates:

- Tables
- Seed data
- Default administrator
- Initial regions

---

# Environment Variables

Example `.env`

```env
DATABASE_URL=postgresql+asyncpg://unicef:unicef123@postgres:5432/unicef

REDIS_URL=redis://redis:6379/0

CELERY_BROKER_URL=redis://redis:6379/2

CELERY_RESULT_BACKEND=redis://redis:6379/3

MINIO_ENDPOINT=minio:9000

MINIO_ACCESS_KEY=minioadmin

MINIO_SECRET_KEY=minioadmin

MINIO_BUCKET_REPORTS=unicef-reports

MINIO_BUCKET_MODELS=ml-models

SECRET_KEY=change-me

APP_ENV=development

APP_DEBUG=true

HOST=0.0.0.0

PORT=8000
```

---

# API Modules

## Authentication

- Login
- JWT Token
- Current User
- Role-based authorization

---

## Weather

- Current weather
- Historical weather
- Climate indices
- Weather anomalies
- Regional statistics

---

## Malaria

- Weekly observations
- Regional statistics
- National statistics
- Trends
- Epidemiological indicators

---

## Alerts

- Climate alerts
- Epidemiological alerts
- Active alerts
- Alert history

---

# Background Jobs

Executed with Celery.

Examples:

- Fetch OpenWeatherMap data
- Import DHIS2 data
- Generate climate indices
- Detect weather anomalies
- Generate reports
- Machine learning preprocessing

---

# Local Development

Run API

```bash
uvicorn src.api.main:app --reload
```

Run Celery Worker

```bash
celery -A src.data_collection.celery_app worker --loglevel=info
```

Run Celery Beat

```bash
celery -A src.data_collection.celery_app beat --loglevel=info
```

---

# Docker Commands

Build

```bash
docker compose build
```

Start

```bash
docker compose up -d
```

Stop

```bash
docker compose down
```

Restart

```bash
docker compose restart
```

View logs

```bash
docker compose logs -f
```

---

# Security

- JWT Authentication
- Password hashing using bcrypt
- Role-based access control
- Environment-based configuration

---

# Future Improvements

- Grafana dashboards
- Prometheus monitoring
- ML prediction service
- GIS integration
- SMS/Email alerting
- Kubernetes deployment
- CI/CD pipeline
- Automated backups

---

# Production Notes

Before deploying:

- Change `SECRET_KEY`
- Use strong PostgreSQL credentials
- Secure Redis
- Secure MinIO credentials
- Enable HTTPS
- Configure reverse proxy (Nginx or Traefik)
- Configure backups
- Enable monitoring and logging

---

# License

Developed for the **  Malaria Surveillance Platform**.

For research, public health surveillance, and early warning systems.