from record_and_transcribe import record_and_transcribe, speak_text
import datetime
import time
import serial
import threading

# Set up serial connection
arduino = serial.Serial('/dev/ttyACM0', 9600)

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

# Main loop to check time
while True:
    now = datetime.datetime.now()

    if now.hour == 15 and now.minute == 0:
        speak_text("Praegu on kell 14:00")
        time.sleep(60)  # Wait a minute to avoid repeating during the same minute
    else:
        time.sleep(5)  # Light sleep to reduce CPU usage
