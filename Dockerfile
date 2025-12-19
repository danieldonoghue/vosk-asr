FROM python:3.11-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    wget unzip sox git \
    libsndfile1 libatomic1 && \
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

# Unpack language models
RUN unzip /app/models/en/model.zip -d /app/models/en && rm /app/models/en/model.zip
RUN unzip /app/models/nl/model.zip -d /app/models/nl && rm /app/models/nl/model.zip
RUN unzip /app/models/sv/model.zip -d /app/models/sv && rm /app/models/sv/model.zip
RUN unzip /app/models/es/model.zip -d /app/models/es && rm /app/models/es/model.zip
RUN unzip /app/models/fi/model.zip -d /app/models/fi && rm /app/models/fi/model.zip

# Mount audio directory at runtime
VOLUME /app/audio

CMD ["python3", "app.py"]

