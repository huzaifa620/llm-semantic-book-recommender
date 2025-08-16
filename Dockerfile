FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libsm6 libxext6 libgl1 cmake rsync \
    && rm -rf /var/lib/apt/lists/* && git lfs install

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# use platform port (Railway sets $PORT)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
