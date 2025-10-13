#!/usr/bin/env python3
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')  # proovib xcb pluginat (võid eemaldada, kui probleeme)

import face_recognition
import cv2
import numpy as np
import time
import pickle

show_video = True

# Load pre-trained face encodings
print("[INFO] loading encodings...")
encodings_path = "encodings.pickle"
if not os.path.exists(encodings_path):
    print(f"❌ Encodings file not found: {encodings_path}")
    exit(1)

try:
    with open(encodings_path, "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data.get("encodings", [])
    known_face_names = data.get("names", [])
except Exception as e:
    print(f"❌ Error loading encodings from {encodings_path}: {e}")
    exit(1)

# Initialize USB camera (instead of Picamera2)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Set correct resolution and format (silmi)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))  # YUYV422

if not cap.isOpened():
    print("❌ Camera could not be opened")
    cap.release()
    exit(1)

# Initialize variables
cv_scaler = 4  # whole number - scale factor for faster processing
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# Kogutud unikaalsed nimed kogu sessiooni jooksul
detected_names = []

def process_frame(frame):
    global face_locations, face_encodings, face_names, detected_names

    # Resize for faster processing
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))

    # BGR -> RGB
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        detected_name = "Unknown"

        # Best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                detected_name = known_face_names[best_match_index]
                # Lisa vaid kui pole juba listis (unikaalne)
                if detected_name not in detected_names:
                    detected_names.append(detected_name)
        face_names.append(detected_name)

    return frame

def draw_results(display):
    for (top, right, bottom, left), face_name in zip(face_locations, face_names):
        # Scale back up
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Face box
        cv2.rectangle(display, (left, top), (right, bottom), (244, 42, 3), 3)

        # Name label
        cv2.rectangle(display, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(display, face_name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

    return display

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

print("[INFO] Starting video stream. Press 'q' to quit (if show_video=True).")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Process + draw
        processed_frame = process_frame(frame)
        display_frame = draw_results(processed_frame)

        # FPS counter
        current_fps = calculate_fps()
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                    (display_frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show video only if enabled (headless-friendly)
        if show_video:
            try:
                cv2.imshow('Video', display_frame)
            except Exception as e:
                # Kui imshow teeb probleeme, lülitame video välja ja jätkame headless režiimis
                print(f"[WARN] cv2.imshow failed: {e}. Switching to headless mode.")
                show_video = False

            if cv2.waitKey(1) == ord("q"):
                break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user (KeyboardInterrupt)")

finally:
    # Cleanup
    try:
        cap.release()
    except Exception:
        pass

    if show_video:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Salvesta kogu sessiooni jooksul kogutud unikaalsed tuvastatud nimed faili
    output_file = "detected_names.txt"
    try:
        if detected_names:
            with open(output_file, "w", encoding="utf-8") as f:
                for name in detected_names:
                    f.write(name + "\n")
            print(f"[INFO] Tuvastatud nimed salvestatud {output_file}: {detected_names}")
        else:
            print("[INFO] Ühtegi nägu ei tuvastatud")
    except Exception as e:
        print(f"[ERROR] Ei õnnestunud salvestada faili {output_file}: {e}")



