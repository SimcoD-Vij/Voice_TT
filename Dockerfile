FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
# Install CPU-only PyTorch to save space and time
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure the local Qwen3-TTS package is in Python path if not already
ENV PYTHONPATH=$PYTHONPATH:/app:/app/Qwen3-TTS

# Expose the port
EXPOSE 8769

# Default command to run the server
# We expect the model path to be provided via environment variable or volume
CMD ["python", "server.py", "--model", "/app/models/Qwen3-TTS-12Hz-0.6B-Base", "--port", "8769", "--host", "0.0.0.0"]
