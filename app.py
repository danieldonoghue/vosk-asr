import os
import wave
import json

# Vosk for EN, ES, NL, SV; Wav2Vec2 for FI
from vosk import Model as VoskModel, KaldiRecognizer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio

# Colour codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# -----------------------------
# Model paths
# -----------------------------
LANGUAGE_MODELS = {
    "en": "models/en/vosk-model-small-en-us-0.15",
    "es": "models/es/vosk-model-small-es-0.42",
    "nl": "models/nl/vosk-model-small-nl-0.22",
    "sv": "models/sv/vosk-model-small-sv-rhasspy-0.15",
    "fi": "models/fi"  # Wav2Vec2 Finnish model
}

# Custom dictionary / vocabulary (optional, boosts recognition)
MEDICINES = ["aspirin", "ibuprofen", "metformin", "paracetamol"]

# -----------------------------
# Load Vosk models
# -----------------------------
print("Loading Vosk models...")
vosk_models = {}
for lang, path in LANGUAGE_MODELS.items():
    if lang != "fi":
        vosk_models[lang] = VoskModel(path)
print("Vosk models loaded.")

# -----------------------------
# Load Finnish Wav2Vec2
# -----------------------------
print("Loading Finnish Wav2Vec2 model...")
try:
    processor_fi = Wav2Vec2Processor.from_pretrained(LANGUAGE_MODELS["fi"])
    model_fi = Wav2Vec2ForCTC.from_pretrained(LANGUAGE_MODELS["fi"])
    model_fi.eval()
    FI_LOADED = True
    print("Finnish model loaded.")
except Exception as e:
    FI_LOADED = False
    print(f"{RED}Warning: Finnish model failed to load: {e}{RESET}")

# -----------------------------
# Vosk transcription
# -----------------------------
def transcribe_vosk(lang, filepath):
    model = vosk_models[lang]
    wf = wave.open(filepath, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    # Boost custom words if present
    if MEDICINES:
        rec.SetGrammar(json.dumps(MEDICINES))

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    result_json = json.loads(rec.FinalResult())
    return result_json.get("text", "")

# -----------------------------
# Finnish transcription
# -----------------------------
def transcribe_finnish(filepath):
    if not FI_LOADED:
        return f"{RED}Finnish model not loaded{RESET}"

    waveform, sr = torchaudio.load(filepath)
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
# Main loop
# -----------------------------
def main():
    audio_dir = "/app/audio"
    if not os.path.exists(audio_dir):
        print(f"{RED}Audio directory {audio_dir} does not exist{RESET}")
        return

    for file in os.listdir(audio_dir):
        if not file.endswith(".wav"):
            continue

        filepath = os.path.join(audio_dir, file)
        lang = file.split("_")[0]

        print(f"Processing {file} ({lang})")
        try:
            if lang == "fi":
                text = transcribe_finnish(filepath)
            else:
                text = transcribe_vosk(lang, filepath)

            # Colour-coded output
            if text.strip():
                print(f"[{lang}] Transcription: {GREEN}{text}{RESET}")
            else:
                print(f"[{lang}] Transcription: {RED}<empty>{RESET}")

        except Exception as e:
            print(f"{RED}Error processing {file}: {e}{RESET}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
