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
    pip install --no-cache-dir -r requirements.txt


COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
RUN python -c "import hashlib, pathlib; p=pathlib.Path('main.py'); print('BUILD_MAIN_EXISTS=', p.exists()); print('BUILD_MAIN_MD5=', hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None); q=pathlib.Path('services/style_encoder.py'); print('BUILD_STYLE_EXISTS=', q.exists()); print('BUILD_STYLE_MD5=', hashlib.md5(q.read_bytes()).hexdigest() if q.exists() else None)"

USER appuser

ENV WEB_CONCURRENCY=1
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*' --workers ${WEB_CONCURRENCY:-1}"]
