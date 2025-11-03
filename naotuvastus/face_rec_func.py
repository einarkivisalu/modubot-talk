import os
import threading
import time
import pickle
import numpy as np
import cv2
import face_recognition
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from gtts import gTTS
import pygame
import tempfile

# --- Kõnesüntees ---
pygame.mixer.init()

def speak_text(text, lang='et'):
    try:
        print(f"Sa ütlesid: {text}")
        tts = gTTS(text=text, lang=lang)
        filename = "temp.mp3"
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        os.remove(filename)
    except Exception as e:
        print(f"[TTS VIGA]: {e}")

# --- Leia USB mikrofon ---
def find_usb_mic():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0 and "usb" in dev['name'].lower():
            print(f"Leidsin USB mikrofoni: {dev['name']} (index {i})")
            return i
    print("USB mikrofoni ei leitud, kasutan vaikimisi sisendit.")
    return sd.default.device[0]

# --- Lae Whisper ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device_index = 0 if torch.cuda.is_available() else -1

print("Laen Whisper mudelit...")
model_id = "openai/whisper-small"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=dtype, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device_index
)
print("Whisper mudel laetud")

# --- Globaalne muutujad ---
is_listening = threading.Event()
MIC_DEVICE = find_usb_mic()

# --- Salvestamine ja transkribeerimine ---
def record_and_transcribe(duration=3):
    if is_listening.is_set():
        print("Kuulamine juba käib.")
        return
    is_listening.set()

    try:
        print("Kuulan...")
        speak_text("Kuulan")

        device_info = sd.query_devices(MIC_DEVICE, 'input')
        samplerate = int(device_info['default_samplerate'])
        channels = 1 if device_info['max_input_channels'] >= 1 else device_info['max_input_channels']
        print(f"Salvestan {duration}s heli ({samplerate} Hz, {channels} kanalit) seadmega {device_info['name']}")

        audio = sd.rec(int(duration * samplerate),
                       samplerate=samplerate,
                       channels=channels,
                       dtype='float32',
                       device=MIC_DEVICE)
        sd.wait()

        # Kontrolli kas heli on olemas
        if np.abs(audio).mean() < 0.0005:
            print("Väga vaikne heli – midagi ei salvestatud.")
            speak_text("Ma ei kuulnud midagi.")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            write(tmp.name, samplerate, audio)
            print("Transkribeerin heli...")
            result = pipe(tmp.name, generate_kwargs={"task": "transcribe", "language": "estonian"})
            transcription = result["text"]
            print(f"Sa ütlesid: {transcription}")
            speak_text(f"Sa ütlesid: {transcription}")
            os.remove(tmp.name)

    except Exception as e:
        print("[VIGA transkriptsioonis]:", e)
    finally:
        is_listening.clear()


# --- Näotuvastus ---
def load_encodings(encodings_path="encodings.pickle"):
    if not os.path.exists(encodings_path):
        print(f"Encodings file not found: {encodings_path}")
        exit(1)
    with open(encodings_path, "rb") as f:
        data = pickle.loads(f.read())
    return data.get("encodings", []), data.get("names", [])

def init_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    if not cap.isOpened():
        print("Kaamerat ei saanud avada")
        cap.release()
        exit(1)
    return cap


# --- Peatsükkel ---
def face_and_audio_main():
    cv_scaler = 4
    known_face_encodings, known_face_names = load_encodings()
    cap = init_camera()
    already_greeted = set()
    last_listen_time = 0
    listen_cooldown = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kaadri lugemine ebaõnnestus.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if len(distances) > 0:
                    best_match = np.argmin(distances)
                    if matches[best_match]:
                        name = known_face_names[best_match]
                        if name.lower() not in already_greeted:
                            greeting = f"Tere, {name}"
                            print(greeting)
                            speak_text(greeting)
                            already_greeted.add(name.lower())

                            # Käivita kuulamine
                            if time.time() - last_listen_time > listen_cooldown:
                                threading.Thread(target=record_and_transcribe, daemon=True).start()
                                last_listen_time = time.time()

                face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= cv_scaler
                right *= cv_scaler
                bottom *= cv_scaler
                left *= cv_scaler
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("Katkestatud kasutaja poolt.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    face_and_audio_main()
