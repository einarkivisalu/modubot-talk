from record_and_transcribe import record_and_transcribe
import time
import serial
import threading

# calendar imports
from record_and_transcribe import speak_text
import datetime


# Serial connection to Arduino
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

# Distance thresholds
DIST_STOP = 50
DIST_SLOW = 100

# === VOICE COMMAND FUNCTIONS ===
def activation_word(text: str):
    if any(word in text.lower() for word in ["tere", "tera", "teere"]):
        return response(text)

def response(text: str):
    if "paremale" in text.lower():
        print("paremale")
        arduino.write(b'paremale\n')
    elif "vasakule" in text.lower():
        print("vasakule")
        arduino.write(b'vasakule\n')
    elif "otse" in text.lower():
        print("otse")
        arduino.write(b"otse\n")
    elif "stop" in text.lower():
        print("stop")
        arduino.write(b"stop\n")
    else:
        print("i love cats")

# === VOICE LISTENER THREAD ===
def voice_listener():
    while True:
        try:
            text = record_and_transcribe(samplerate=16000, device=3)
            if text:
                activation_word(text)
        except Exception as e:
            print(f"Voice error: {e}")
        time.sleep(1)

# === CALENDAR CHECKER THREAD ===
def calendar():
    while True:
        now = datetime.datetime.now()
        if now.hour == 17 and now.minute == 10:
            speak_text("Praegu on kassiaeg")
            time.sleep(60)  # Väldib kordust sama minuti jooksul
        else:
            time.sleep(60)

# === START THREADS ===
threading.Thread(target=calendar, daemon=True).start()
threading.Thread(target=voice_listener, daemon=True).start()

# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Program exited.")
