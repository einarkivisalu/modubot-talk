import time
import tempfile
import os


import torch
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


from gtts import gTTS
import pygame


# --- Tekstist kõneks pygame'i abil ---
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
       os.remove(filename)


   except Exception as e:
       print(f"Kõnesünteesi viga: {e}")


# --- Seadista mudel ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device_index = 0 if torch.cuda.is_available() else -1

# Whisper AI kasutamine - vaja alla laadida moodul
model_id = "openai/whisper-small"  # saaks ka olla whisper-large-v3
print("Laen mudeli...")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
   model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
   "automatic-speech-recognition",
   model=model,
   tokenizer=processor.tokenizer,
   feature_extractor=processor.feature_extractor,
   torch_dtype=torch_dtype,
   device=device_index,
)

print("Mudel on valmis.")
speak_text("Mudel on laetud ja valmis.")


# --- Heli salvestamine ja transkribeerimine ---
def record_and_transcribe(duration=5, samplerate=16000):
   print("Kuulan nüüd...")
   speak_text("Kuulan")


   audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
   sd.wait()


   with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
       write(tmp.name, samplerate, audio)


       try:
           raw_audio, _ = sf.read(tmp.name)
           inputs = processor(raw_audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
           inputs = {k: v.to(model.device) for k, v in inputs.items()}

           forced_decoder_ids = processor.get_decoder_prompt_ids(language="estonian", task="transcribe")

           generated_ids = model.generate(**inputs, forced_decoder_ids=forced_decoder_ids)
           transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


           print(f"Sa ütlesid: {transcription}")
           speak_text(f"Sa ütlesid: {transcription}")


       except Exception as e:
           print("Viga transkriptsioonis:", e)



# --- Peamine tsükkel ---
def main():
   try:
       while True:
           time.sleep(1)
           record_and_transcribe(duration=5)
   except KeyboardInterrupt:
       print("Lõpetan. Nägemist!")
       speak_text("Lõpetan. Nägemist!")




if __name__ == "__main__":
   main()
