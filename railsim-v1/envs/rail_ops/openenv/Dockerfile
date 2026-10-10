# Build context: envs/rail_ops (core/, openenv/ and tasks/ side by side). The curated data is not needed at runtime -
# tasks are prebuilt in tasks/rail-v1-eval and tasks/rail-v1-train.
#   docker build -f openenv/Dockerfile -t rail-ops .
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user PORT=8000 ENABLE_WEB_INTERFACE=true \
    RAIL_TASKS_DIR=/app/tasks/rail-v1-eval:/app/tasks/rail-v1-train
WORKDIR /app
COPY core /app/core
COPY openenv /app/openenv
COPY tasks /app/tasks
RUN pip install /app/core /app/openenv \
    && useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["sh", "-c", "uvicorn rail_openenv.server:app --host 0.0.0.0 --port ${PORT:-8000} --ws-ping-timeout 600"]
