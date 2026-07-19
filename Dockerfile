FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "bcrypt==3.2.2"

COPY . /app

RUN mkdir -p /tmp/reports /var/log

EXPOSE 8000

CMD ["uvicorn", "src.api.main:create_application", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
