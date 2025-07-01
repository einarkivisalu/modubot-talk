from record_and_transcribe import record_and_transcribe
import time
import serial
import threading

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

# === OBJECT AVOIDANCE THREAD ===
def avoid_objects():
    while True:
        try:
            if arduino.in_waiting:
                line = arduino.readline().decode().strip()
                print(f"[Arduino] {line}")

                # Expecting lines like: LEFT:45 CENTER:60 RIGHT:50
                if all(keyword in line for keyword in ["LEFT", "CENTER", "RIGHT"]):
                    try:
                        parts = line.replace("LEFT:", "").replace("CENTER:", "").replace("RIGHT:", "").split()
                        left, center, right = map(int, parts)

                        # Avoidance Logic
                        if center <= DIST_STOP or left <= DIST_STOP or right <= DIST_STOP:
                            print("Obstacle too close! Stopping and avoiding...")
                            arduino.write(b'stop\n')
                            time.sleep(0.3)
                            arduino.write(b'tagasi\n')
                            time.sleep(0.8)

                            if left > right:
                                arduino.write(b'vasakule\n')
                            else:
                                arduino.write(b'paremale\n')
                            time.sleep(0.5)
                            arduino.write(b'stop\n')
                            time.sleep(0.2)

                        elif center < DIST_SLOW or left < DIST_SLOW or right < DIST_SLOW:
                            print("Slowing down due to obstacle.")
                            arduino.write(b'otse\n')  # Assuming Arduino reduces speed in "otse" mode
                        else:
                            arduino.write(b'otse\n')

                    except Exception as e:
                        print(f"Error parsing distance data: {e}")

            time.sleep(0.1)
        except Exception as e:
            print(f"[Error] {e}")
            break

# === START THREADS ===
threading.Thread(target=avoid_objects, daemon=True).start()
threading.Thread(target=voice_listener, daemon=True).start()

# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Program exited.")
