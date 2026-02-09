import time
import os
from gtts import gTTS
import pygame


def speak_text(text, lang='et'):
    try:
        tts = gTTS(text=text, lang=lang)
        filename = "temp.mp3"
        tts.save(filename)

        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.unload()
        pygame.mixer.quit()
        os.remove(filename)

    except Exception as e:
        print(f"Kõnesünteesi viga: {e}")


def main():
    text = "1991"
    speak_text(text)


if __name__ == "__main__":
    main()
