import sys
import time
import tempfile
import os

import torch
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from gtts import gTTS
import pygame


# --- List all available input devices ---
def list_input_devices():
    print("Available input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"{i}: {dev['name']}")


# --- Automatically find a USB microphone (fallback: default input) ---
def find_usb_mic():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0 and "usb" in dev['name'].lower():
            print(f"USB mic found: {dev['name']} (index {i})")
            return i
    print("No USB mic found, using default input device.")
    return sd.default.device[0]


# --- Tekstist kõneks pygame'i abil ---
def speak_text(text, lang='et'):
    try:
        tts = gTTS(text=text, lang=lang)
        filename = "temp.mp3"
        tts.save(filename)

        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.unload()
        os.remove(filename)

    except Exception as e:
        print(f"Kõnesünteesi viga: {e}")


# --- Seadista mudel ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype=torch.float16 if torch.cuda.is_available() else torch.float32
device_index = 0 if torch.cuda.is_available() else -1

model_id = "openai/whisper-small"
print("Laen mudeli...")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=dtype,
    device=device_index,
)

print("Mudel on valmis.")
speak_text("Mudel on laetud ja valmis.")


# --- Heli salvestamine ja transkribeerimine ---
def record_and_transcribe(duration=3, device=None):
    print("Kuulan nüüd...")
    speak_text("Kuulan")

    device_info = sd.query_devices(device, 'input')
    samplerate = int(device_info['default_samplerate'])
    channels = 1 if device_info['max_input_channels'] >= 1 else device_info['max_input_channels']

    print(f"Using device: {device_info['name']}, samplerate: {samplerate}, channels: {channels}")

    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=channels,
                   dtype='float32',
                   device=device)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write(tmp.name, samplerate, audio)

        try:
            raw_audio, _ = sf.read(tmp.name)
            result = pipe(tmp.name, generate_kwargs={"task": "transcribe", "language": "estonian"})
            transcription = result["text"]

            print(f"Sa ütlesid: {transcription}")
            speak_text(f"Sa ütlesid: {transcription}")
            return transcription

        except Exception as e:
            print("Viga transkriptsioonis:", e)


# --- Peamine tsükkel ---
def main():
    usb_mic_device_index = find_usb_mic()

    try:
        while True:
            time.sleep(1)
            record_and_transcribe(duration=5, device=usb_mic_device_index)
    except KeyboardInterrupt:
        print("Peatamine kasutaja poolt.")
        speak_text("Lõpetan. Nägemist!")
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()

