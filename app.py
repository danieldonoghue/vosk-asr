import os
import wave
import json

# Vosk for EN, ES, NL, SV; Wav2Vec2 for FI
from vosk import Model as VoskModel, KaldiRecognizer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
import soundfile as sf

# -----------------------------
# Terminal color codes
# -----------------------------
class Colors:
    EN = "\033[1;34m"   # Blue
    ES = "\033[1;32m"   # Green
    NL = "\033[1;36m"   # Cyan
    SV = "\033[1;35m"   # Magenta
    FI = "\033[1;33m"   # Yellow
    RESET = "\033[0m"

COLOR_MAP = {
    "en": Colors.EN,
    "es": Colors.ES,
    "nl": Colors.NL,
    "sv": Colors.SV,
    "fi": Colors.FI
}

# -----------------------------
# Folder paths for all language models
# -----------------------------
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
# Load all Vosk models
# -----------------------------
print("Loading Vosk models...")
vosk_models = {lang: VoskModel(path) for lang, path in LANGUAGE_MODELS.items() if lang != "fi"}
print("Vosk models loaded.")

# -----------------------------
# Load Finnish Wav2Vec2 model
# -----------------------------
print("Loading Finnish Wav2Vec2 model...")
processor_fi = Wav2Vec2Processor.from_pretrained(LANGUAGE_MODELS["fi"])
model_fi = Wav2Vec2ForCTC.from_pretrained(LANGUAGE_MODELS["fi"])
model_fi.eval()  # inference mode
print("Finnish model loaded.")

# -----------------------------
# Vosk transcription function
# -----------------------------
def transcribe_vosk(lang, filepath, custom_words=None):
    model = vosk_models[lang]
    wf = wave.open(filepath, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    # Merge custom words with model (runtime grammar)
    if custom_words:
        try:
            rec.SetWords(custom_words)
        except Exception:
            # fallback for older Vosk models
            rec.SetGrammar(json.dumps(custom_words))

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    result_json = json.loads(rec.FinalResult())
    return result_json.get("text", "")

# -----------------------------
# Finnish transcription function
# -----------------------------
def transcribe_finnish(filepath):
    # Use soundfile to avoid torchcodec dependency issues
    waveform, sr = sf.read(filepath, dtype='float32')
    
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)
    
    # Convert to 16kHz if needed
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    
    input_values = processor_fi(
        waveform,
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

        color = COLOR_MAP.get(lang, Colors.RESET)
        print(f"{color}Processing {file} ({lang})...{Colors.RESET}")

        try:
            if lang == "fi":
                text = transcribe_finnish(filepath)
            else:
                text = transcribe_vosk(filepath=filepath, lang=lang, custom_words=MEDICINES)
            
            # Print color-coded transcription
            print(f"{color}[{lang}] Transcription: {text}{Colors.RESET}\n")
        except Exception as e:
            print(f"{color}Error processing {file}: {e}{Colors.RESET}\n")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
