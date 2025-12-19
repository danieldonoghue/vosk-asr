import os
import wave
import json

# Vosk for EN, ES, NL, SV; Wav2Vec2 for FI
from vosk import Model as VoskModel, KaldiRecognizer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio

# Folder paths for all language models
LANGUAGE_MODELS = {
    "en": "models/en/vosk-model-small-en-us-0.15",
    "es": "models/es/vosk-model-small-es-0.42",
    "nl": "models/nl/vosk-model-small-nl-0.22",
    "sv": "models/sv/vosk-model-small-sv-rhasspy-0.15",
    "fi": "models/fi"  # Wav2Vec2 Finnish model
}

# Example custom dictionary / vocabulary
MEDICINES = ["aspirin", "ibuprofen", "metformin", "paracetamol"]

# -----------------------------
# Load all models at startup
# -----------------------------
print("Loading Vosk models...")
vosk_models = {lang: VoskModel(path) for lang, path in LANGUAGE_MODELS.items() if lang != "fi"}
print("Vosk models loaded.")

print("Loading Finnish Wav2Vec2 model...")
processor_fi = Wav2Vec2Processor.from_pretrained(LANGUAGE_MODELS["fi"])
model_fi = Wav2Vec2ForCTC.from_pretrained(LANGUAGE_MODELS["fi"])
model_fi.eval()  # inference mode
print("Finnish model loaded.")

# -----------------------------
# Vosk transcription function
# -----------------------------
def transcribe_vosk(lang, filepath):
    model = vosk_models[lang]
    wf = wave.open(filepath, "rb")
    rec = KaldiRecognizer(model, wf.getframerate(), json.dumps(MEDICINES))
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            pass
    result = rec.FinalResult()
    return json.loads(result)["text"]

# -----------------------------
# Finnish transcription function
# -----------------------------
def transcribe_finnish(filepath):
    waveform, sr = torchaudio.load(filepath)
    
    # Convert to 16kHz mono if needed
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    
    input_values = processor_fi(
        waveform.squeeze().numpy(),
        return_tensors="pt",
        sampling_rate=16000
    ).input_values
    
    with torch.no_grad():
        logits = model_fi(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor_fi.decode(predicted_ids[0])
    return transcription.lower()

# -----------------------------
# Main function
# -----------------------------
def main():
    audio_dir = "/app/audio"  # mounted folder in Docker
    if not os.path.exists(audio_dir):
        print(f"Audio directory {audio_dir} does not exist.")
        return

    for file in os.listdir(audio_dir):
        if not file.endswith(".wav"):
            continue

        filepath = os.path.join(audio_dir, file)
        lang = file.split("_")[0]  # e.g., en_test.wav, fi_test.wav

        print(f"Processing {file} ({lang})")
        try:
            if lang == "fi":
                text = transcribe_finnish(filepath)
            else:
                text = transcribe_vosk(lang, filepath)
            print(f"[{lang}] Transcription: {text}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
