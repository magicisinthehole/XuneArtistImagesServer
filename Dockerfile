FROM python:3.12-slim

# System dependencies for opencv-python-headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY gunicorn.conf.py .

# Create artists directory mount point
RUN mkdir -p /artists

# Non-root user (override UID/GID with --build-arg for your system)
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} zuneapi && useradd -u ${APP_UID} -g zuneapi zuneapi
RUN chown -R zuneapi:zuneapi /app /artists
USER zuneapi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["gunicorn", "main:app"]
