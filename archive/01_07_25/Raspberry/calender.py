from record_and_transcribe import speak_text
import datetime
import time

# Main loop to check time
while True:
    now = datetime.datetime.now()

    if now.hour == 15 and now.minute == 0:
        speak_text("Praegu on kassiaeg")
        time.sleep(60)  # Wait a minute to avoid repeating during the same minute
    else:
        time.sleep(60)  # Light sleep to reduce CPU usage

