from record_and_transcribe import record_and_transcribe
import sounddevice as sd
#from calender import *
import time
import serial


# Set up serial connection
arduino = serial.Serial('/dev/ttyACM0', 9600)


# use "modubot" to activate robot
def activation_word(text: str):
   if "tere" in text.lower():
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
    # Detect usable input device
    preferred_device_name = "AB13X USB Audio"  # Change this to match your mic name
    device_index = None
    for idx, dev in enumerate(sd.query_devices()):
        if preferred_device_name.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            device_index = idx
            break

    if device_index is None:
        print("No suitable input device found.")
        return

    # Get that device's native sample rate
    samplerate = int(sd.query_devices(device_index)["default_samplerate"])
    print(f"Using device {device_index} at {samplerate} Hz")

    while True:
        try:
            text = record_and_transcribe(duration=5, samplerate=samplerate, device=device_index)
            if text:
                activation_word(text)
        except Exception as e:
            print(f"Voice error: {e}")
        time.sleep(1)



# Start the voice listener in a separate thread
voice_listener()
