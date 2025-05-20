from record_and_transcribe import record_and_transcribe
from time import time
import time
import serial
import threading

# Set up serial connection
arduino = serial.Serial('/dev/ttyACM0', 9600)

# use "modubot" to activate robot
def activation_word(text: str):
    if "modubot" in text.lower():
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

# Background thread function to continuously listen
def voice_listener():
    while True:
        try:
            text = record_and_transcribe()
            if text:
                response(text)
        except Exception as e:
            print(f"Voice error: {e}")
        time.sleep(1)  # Small delay to avoid tight loop if error happens

# Start the voice listener in a separate thread
listener_thread = threading.Thread(target=voice_listener, daemon=True)
listener_thread.start()

