FROM python:3.13-slim

ENV POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root



EXPOSE 8000

CMD ["uvicorn", "--app-dir", "/app/src/backend", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
