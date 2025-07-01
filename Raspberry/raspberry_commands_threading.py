from record_and_transcribe import record_and_transcribe
import time
import serial
import RPi.GPIO as GPIO
import threading

# Set up serial connection
arduino = serial.Serial('/dev/ttyACM0', 9600)

# use "modubot" to activate robot
def activation_word(text: str):
    if any(word in text.lower() for word in ["tere", "tera", "teere"]):
        return response(text)
    else:
        return None

# Response function
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

# Background thread to listen to voice commands
def voice_listener():
    while True:
        try:
            text = record_and_transcribe(samplerate=16000, device=1)
            if text:
                activation_word(text)
        except Exception as e:
            print(f"Voice error: {e}")
        time.sleep(1)

# THREAD FOR AVOIDING OBJECTS AT ALL TIMES
def avoid_objects():
    while True:
        try:
            if arduino.in_waiting:
                line = arduino.readline().decode().strip()
                print(f"[Arduino] {line}")
            time.sleep(0.1)
        except Exception as e:
            print(f"[Error] {e}")
            break

# Start both background threads
threading.Thread(target=avoid_objects, daemon=True).start()
threading.Thread(target=voice_listener, daemon=True).start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Program exited.")
