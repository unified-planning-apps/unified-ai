FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* requirements.txt* /app/

RUN pip install --upgrade pip setuptools wheel \
    && if [ -f "poetry.lock" ]; then \
         pip install "poetry>=1.8" && poetry install --no-root --no-interaction --no-ansi; \
       elif [ -f "requirements.txt" ]; then \
         pip install -r requirements.txt; \
       else \
         echo "No requirements.txt or poetry.lock found" >&2; exit 1; \
       fi

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:create_application", "--host", "0.0.0.0", "--port", "8000"]