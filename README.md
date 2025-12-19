# Vosk-ASR Multi-Language Docker Transcription

## Overview

This project provides a **self-hosted, multi-language automatic speech recognition (ASR) system** designed for **offline, EU-compliant deployment**. It supports the following languages:

* English (Vosk model)
* Spanish (Vosk model)
* Dutch (Vosk model)
* Swedish (Vosk model)
* Finnish (wav2vec2 fine-tuned model)

It is **containerized for ARM64**, CPU-only, and supports **custom dictionaries**, making it suitable for medical transcription or any domain-specific vocabulary.

This system is designed to:

* Run **locally in Docker** for evaluation
* Support **custom medical dictionaries**
* Be deployed on **Azure VM** or **Azure Kubernetes Service (AKS)** without GPU dependencies
* Scale horizontally for batch or real-time transcription

---

## Project Structure

```
vosk-asr/
├─ Dockerfile            # Docker build instructions
├─ requirements.txt      # Python dependencies
├─ app.py                # Main transcription script
├─ models/               # Pretrained models per language
│   ├─ en/               # English (Vosk)
│   ├─ es/               # Spanish (Vosk)
│   ├─ nl/               # Dutch (Vosk)
│   ├─ sv/               # Swedish (Vosk)
│   └─ fi/               # Finnish (wav2vec2)
└─ audio/                # Test audio files
```

---

## How It Works

### 1. Language Models

* **Vosk/Kaldi**: Pretrained models for English, Spanish, Dutch, Swedish
* **wav2vec2**: Finnish language model from Hugging Face
* All models are loaded locally in the Docker container

### 2. Dictionary Support

* Custom vocabulary is injected at runtime via the **KaldiRecognizer**
* Example: list of medicines (`aspirin`, `ibuprofen`, etc.)
* Enhances recognition of domain-specific terms

### 3. Audio Processing

* Accepts WAV audio files per language (`en_test.wav`, `fi_test.wav`, etc.)
* Processes audio in chunks for streaming or batch transcription
* Outputs text transcription to console or can be extended for API usage

### 4. Deployment Options

* **Local Docker**: For development and testing
* **Azure VM (ARM64)**: CPU-only deployment, mount models via volumes
* **Azure Kubernetes Service (AKS)**: Scales horizontally without GPU, supports multi-pod deployment

---

## Hardware & Cost Recommendations

### Local Testing

* CPU: 4–8 ARM64 cores (Apple M1/M2, Raspberry Pi 4, or similar)
* RAM: 8–16 GB
* Disk: 20–50 GB for models

### Azure VM (ARM64)

* Example VM: **Dpsv5 / Epsv5 ARM64**
* CPU: 8–16 vCPU
* RAM: 16–32 GB
* Disk: 50–100 GB
* Cost: ~€0.15–0.20/hour for 8 vCPU (~€100–150/month for continuous usage)

### AKS

* CPU-only nodes, ARM64 scale sets
* Horizontal scaling for concurrent audio streams
* Persistent Volumes for model storage

---

## Setup & Local Testing

### 1. Clone Repository

```bash
git clone <repository-url>
cd vosk-asr
```

### 2. Download Pretrained Models

**Vosk Models:**

* English: `https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip`
* Spanish: `https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip`
* Dutch: `https://alphacephei.com/vosk/models/vosk-model-nl.zip`
* Swedish: `https://alphacephei.com/vosk/models/vosk-model-sv.zip`

**Steps:**

```bash
mkdir -p models/en models/es models/nl models/sv
wget -O models/en/model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip models/en/model.zip -d models/en
# Repeat for other languages
```

**Finnish (wav2vec2):**

* Model: `wav2vec2-base-fi-voxpopuli-v2` from Hugging Face
* Can be downloaded inside Docker at runtime or pre-downloaded into `models/fi`

```bash
mkdir -p models/fi
# Use transformers to load model at first run
```

### 3. Add Test Audio

* Place `.wav` files in `audio/` folder
* Filename convention: `<lang>_<name>.wav` (e.g., `en_test.wav`, `fi_test.wav`)

### 4. Update Custom Dictionary

* Edit `MEDICINES` list in `app.py`

```python
MEDICINES = ["aspirin", "ibuprofen", "metformin", "paracetamol"]
```

* Add any domain-specific vocabulary here

### 5. Build Docker Image

```bash
docker build -t vosk-multilang .
```

### 6. Run Container Locally

```bash
docker run -it --rm -v $(pwd)/audio:/app/audio vosk-multilang
```

* The script will process all audio files and print transcriptions
* Dictionary words will be recognized in Vosk languages

---

## Deployment to Azure VM

1. Push Docker image to **Azure Container Registry (ACR)**
2. Create ARM64 VM (Dpsv5 / Epsv5)
3. Pull Docker image on VM:

```bash
docker pull <acr-repo>/vosk-multilang:latest
```

4. Run container with mounted models/audio:

```bash
docker run -d -v /models:/app/models -v /audio:/app/audio vosk-multilang
```

5. Logs will show transcriptions; optionally expose via REST API

---

## Deployment to AKS

1. Push Docker image to **ACR**
2. Create **AKS ARM64 cluster**
3. Use **Deployment + Service YAML** to launch container pods
4. Mount models using **Persistent Volume Claims (PVC)**
5. Expose container as **REST API** or message queue consumer
6. Scale pods horizontally for multiple simultaneous audio streams

---

## Testing & Metrics

* Compare **Word Error Rate (WER)** against baseline (e.g., Azure 4o-transcribe)
* Test **dictionary recognition** by adding domain-specific words
* Optionally fine-tune wav2vec2 for Finnish or other low-performing domains

---

## Next Steps / Enhancements

1. Convert into a **FastAPI or Flask REST API** for real-time transcription
2. Add **automatic language detection**
3. Implement **logging and batch processing pipeline**
4. Fine-tune acoustic models for domain-specific vocabulary
5. Deploy **multi-pod AKS cluster** for production-scale throughput

