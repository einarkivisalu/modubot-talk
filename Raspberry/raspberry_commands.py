from record_and_transcribe import record_and_transcribe, speak_text
import time
import serial
import datetime

now = datetime.datetime.now()
if now.hour == 14 and now.minute == 35:
   speak_text("Praegu on kassiaeg!")

text = record_and_transcribe()

arduino = serial.Serial("/dev/ttyACM0", 9600)
time.sleep(2)

def response(text: str):
   if "paremale" in text.lower():
       print("paremale")
       arduino.write(b"paremale\n")

   if "vasakule" in text.lower():
       print("vasakule")
       arduino.write(b"vasakule\n")

   if "otse" in text.lower():
       print("otse")
       arduino.write(b"otse\n")

   if "stop" in text.lower():
       print("stop")
       arduino.write(b"stop\n")

   else:
       response(text)

response(text)
