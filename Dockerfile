FROM python:3.11-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    wget unzip sox git && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Prepare transformer and torch caches
RUN mkdir -p /root/.cache/huggingface/transformers && mkdir -p /root/.cache/torch

# Copy models and app
COPY models /app/models
COPY app.py /app/app.py

# Mount audio directory at runtime
VOLUME /app/audio

CMD ["python3", "app.py"]

