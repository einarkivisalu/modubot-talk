from record_and_transcribe import record_and_transcribe
import time
import serial

text = record_and_transcribe()

arduino = serial.Serial('/dev/ttyUSB0', 9600)
time.sleep(2)

def response(text: str):
    if "paremale" in text.lower():
        print("paremale")
        arduino.write(b'Right')

    if "vasakule" in text.lower():
        print("vasakule")
        arduino.write(b'Left')

    else:
        print("i love cats")

response(text)
