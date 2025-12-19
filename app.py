import os
import wave
import json

# Vosk for EN, ES, NL, SV; wav2vec2 for FI
from vosk import Model as VoskModel, KaldiRecognizer

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

LANGUAGE_MODELS = {
    "en": "models/en",
    "es": "models/es",
    "nl": "models/nl",
    "sv": "models/sv",
    "fi": "models/fi"
}

# Example dictionary / custom vocabulary
MEDICINES = ["aspirin", "ibuprofen", "metformin", "paracetamol"]

def transcribe_vosk(lang, filepath):
    model = VoskModel(LANGUAGE_MODELS[lang])
    wf = wave.open(filepath, "rb")
    rec = KaldiRecognizer(model, wf.getframerate(), json.dumps(MEDICINES))
    result = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            pass
    result = rec.FinalResult()
    return json.loads(result)

def transcribe_finnish(filepath):
    processor = Wav2Vec2Processor.from_pretrained("Finnish-NLP/wav2vec2-base-fi-voxpopuli-v2-finetuned")
    model = Wav2Vec2ForCTC.from_pretrained("Finnish-NLP/wav2vec2-base-fi-voxpopuli-v2-finetuned")
    waveform, sr = torchaudio.load(filepath)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

def main():
    audio_dir = "/app/audio"
    for file in os.listdir(audio_dir):
        filepath = os.path.join(audio_dir, file)
        if file.endswith(".wav"):
            lang = file.split("_")[0]  # filename convention: en_test.wav, fi_test.wav, etc.
            print(f"Processing {file} for language {lang}")
            if lang == "fi":
                text = transcribe_finnish(filepath)
            else:
                text = transcribe_vosk(lang, filepath)
            print(f"Transcription: {text}")

if __name__ == "__main__":
    main()

