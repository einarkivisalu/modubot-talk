# TALTECH ESTONIAN MODELS TalTechNLP/whisper-large-et, TalTechNLP/xls-r-300m-et:

import os
import warnings
import time
import soundfile as sf
import numpy as np
from scipy.signal import resample
import torch
from transformers import pipeline

# CHANGE (if needed)
# set threads for hpc
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["MKL_NUM_THREADS"] = "24"
os.environ["NUMEXPR_NUM_THREADS"] = "24"

# GPU settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Ignore future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Audio settings

#CHANGE
audio_path = r"C:\Users\anett\Downloads\Jaaegparoodia.mp3"

target_samplerate = 16000
audio_chunk = 30      # max seconds per chunk
stride_secs = 3       # overlap to avoid cutting words

# Load Whisper pipeline
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model="TalTechNLP/xls-r-300m-et",

    # CHANGE
    device=-1,  # CPU=-1, GPU=0
    torch_dtype=torch.float16
)

# Load audio
audio, sr = sf.read(audio_path)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Resample if needed
if sr != target_samplerate:
    print(f"Resampling from {sr} Hz to {target_samplerate} Hz")
    num_samples = int(len(audio) * target_samplerate / sr)
    audio = resample(audio, num_samples)
audio = audio.astype("float32")

# Chunking
chunk_size = audio_chunk * target_samplerate
stride = stride_secs * target_samplerate
step = chunk_size - stride

total_start_time = time.perf_counter()
final_text = []
chunk_counter = 0

print("30 sec Transcriptions:\n")

for start in range(0, len(audio), step):
    end = start + chunk_size
    chunk = audio[start:end]

    if len(chunk) < target_samplerate:
        break

    start_time = time.perf_counter()

    # Transcribe chunk (language automatically inferred from model)
    result = asr_pipeline(chunk)
    text_chunk = result["text"]

    print(text_chunk, flush=True)
    final_text.append(text_chunk)

    chunk_counter += 1
    end_time = time.perf_counter()
    print(f"\n chunk_{chunk_counter} (30sec) transcription time: {end_time - start_time:.2f} seconds")

total_end_time = time.perf_counter()
print("\nFull transcription done.")
print(f"Full transcription time: {total_end_time - total_start_time:.2f} seconds")
