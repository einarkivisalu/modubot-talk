from Face_Recognition.facial_recognition import run_recognition
from Test import speak_text

# Käivitame tuvastamise
run_recognition()

# Loeme nimed faili ja tervitame
detected_names_file = "detected_names.txt"
with open(detected_names_file, "r", encoding="utf-8") as f:
    detected_names = [line.strip() for line in f if line.strip()]

for name in detected_names:
    if name.lower() == "anett":
        print("Tere Anett")
        speak_text("Tere Anett")
    elif name.lower() == "vera":
        print("Tere Vera")
        speak_text("Tere Vera")
    else:
        print(f"Tuvastatud: {name}")
