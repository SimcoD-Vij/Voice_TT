FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "pocket-tts" \
    "torch>=2.5.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
RUN mkdir -p /voices

COPY pocket_tts_entrypoint.py /app/pocket_tts_entrypoint.py

EXPOSE 8000

# ── KEY CHANGE: healthcheck reads a file, not network ────────────────────────
# File is written AFTER model loads AND is injected into the server.
# This means "healthy" = truly ready to serve TTS, not just "port is open".
HEALTHCHECK --interval=10s --timeout=5s --start-period=400s --retries=40 \
  CMD test -f /tmp/pocket_tts_ready || exit 1

CMD ["python", "/app/pocket_tts_entrypoint.py"]
