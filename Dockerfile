FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

COPY tests/ tests/

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

CMD ["pytest", "-v"]
