#!/usr/bin/env python3
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')  # Qt plugin (võid eemaldada, kui probleeme)

import face_recognition
import cv2
import numpy as np
import time
import pickle
from gtts import gTTS
import pygame

pygame.mixer.init()

def speak_text(text, lang='et'):
    try:
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
        print(f"Kõnesünteesi viga: {e}")

def load_encodings(encodings_path="encodings.pickle"):
    if not os.path.exists(encodings_path):
        print(f"❌ Encodings file not found: {encodings_path}")
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
        print("Camera could not be opened")
        cap.release()
        exit(1)
    return cap

def main():
    cv_scaler = 4
    known_face_encodings, known_face_names = load_encodings()
    cap = init_camera()
    already_greeted = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            resized_frame = cv2.resize(frame, (0,0), fx=1/cv_scaler, fy=1/cv_scaler)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(distances) > 0:
                    best_index = np.argmin(distances)
                    if matches[best_index]:
                        name = known_face_names[best_index]
                        if name.lower() not in already_greeted:
                            # Reaalajas tervitus
                            greeting = f"Tere {name}"
                            print(greeting)
                            speak_text(greeting)
                            already_greeted.add(name.lower())
                face_names.append(name)

            # Joonista tulemused
            for (top,right,bottom,left), name in zip(face_locations, face_names):
                top *= cv_scaler
                right *= cv_scaler
                bottom *= cv_scaler
                left *= cv_scaler
                cv2.rectangle(frame, (left,top),(right,bottom),(244,42,3),3)
                cv2.rectangle(frame, (left-3,top-35),(right+3,top),(244,42,3),cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left+6,top-6), font, 1.0, (255,255,255),1)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
