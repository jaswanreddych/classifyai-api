FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install dependencies first so this layer is cached until requirements change.
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY app ./app
# The architecture file is small and lives in git. The weights are mounted at
# run time (see README) so the image stays free of the 87 MB model file.
COPY models/model_architecture.json ./models/model_architecture.json

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
