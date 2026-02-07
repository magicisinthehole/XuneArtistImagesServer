import os

# Bind to port 8000 inside the container; reverse proxy handles 80/443 + TLS
bind = "0.0.0.0:8000"

# Workers and threads - conservative for opencv/numpy memory usage (~100MB/worker)
workers = int(os.environ.get("GUNICORN_WORKERS", 4))
threads = int(os.environ.get("GUNICORN_THREADS", 8))

# Timeout: 120s generous for /images endpoint (many external API calls + downloads)
timeout = 120
graceful_timeout = 30

# Keep-alive: standard value behind a reverse proxy
keepalive = 5

# Logging: stdout/stderr for Docker log collection
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Worker recycling: restart after N requests to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Preload app to share memory across forked workers
preload_app = True

# Trust X-Forwarded-* headers from reverse proxy (set to specific IPs if not behind trusted proxy)
forwarded_allow_ips = "*"
