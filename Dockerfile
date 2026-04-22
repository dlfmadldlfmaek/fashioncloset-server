FROM python:3.10-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.2.2+cpu torchvision==0.17.2+cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps git+https://github.com/openai/CLIP.git

COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app

USER appuser

ENV WEB_CONCURRENCY=1
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*' --workers ${WEB_CONCURRENCY:-1}"]
