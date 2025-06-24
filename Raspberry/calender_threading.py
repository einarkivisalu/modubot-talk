from record_and_transcribe import record_and_transcribe
import datetime
import time
import threading

def calendar_checker():
    while True:
        now = datetime.datetime.now()
        if now.hour == 15 and now.minute == 0:
            speak_text("Praegu on kassiaeg")
            time.sleep(60)  # Väldib kordust sama minuti jooksul
        else:
            time.sleep(60)

# Käivitab kalenderi taustal
threading.Thread(target=calendar_checker, daemon=True).start()

# Põhiprogramm
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Bye.")
