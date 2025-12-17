import os
from transformers import pipeline
import soundfile as sf
import numpy as np
from scipy.signal import resample
import time
import warnings

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# CHANGE (if needed)
# set threads for hpc
os.environ["OMP_NUM_THREADS"] = "24"
os.environ["MKL_NUM_THREADS"] = "24"
os.environ["NUMEXPR_NUM_THREADS"] = "24"


#TODO fix warnings - warning "task=transcribe, but also have set `forced_decoder_ids.." comes from transformers library
#ignore whisper warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#CHANGE
audio_path = r"C:\Users\anett\Downloads\Jaaegparoodia.mp3"

target_samplerate = 16000
audio_chunk = 30      # max seconds for whisper models
stride_secs = 3      # overlap to avoid cutting words

# Load pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",

    #CHANGE
    device=0, ## CPU=-1, GPU=0
    torch_dtype="float16" #memory usage
)

# Load audio
audio, sr = sf.read(audio_path)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Resample audio if needed (whisper needs 16kHz)
if sr != target_samplerate:
    print(f"Resampling from {sr} Hz to {target_samplerate} Hz")
    num_samples = int(len(audio) * target_samplerate / sr)
    audio = resample(audio, num_samples)
audio = audio.astype("float32")

# cut sound to chunks for whisper (30sec max)
chunk_size = audio_chunk * target_samplerate
stride = stride_secs * target_samplerate
step = chunk_size - stride

total_start_time = time.perf_counter()

print("30 sec Transcriptions:\n")

final_text = []
chunk_counter = 0

for start in range(0, len(audio), step):
    end = start + chunk_size
    chunk = audio[start:end]

    if len(chunk) < target_samplerate:
        break

    start_time = time.perf_counter()

    # Transcribe this chunk
    result = asr_pipeline(chunk, generate_kwargs={"task": "transcribe", "language": "estonian"})
    text_chunk = result["text"]

    # Print when chunk transcribed
    print(text_chunk, end="\n", flush=True)
    final_text.append(text_chunk)

    chunk_counter += 1

    end_time = time.perf_counter()
    print(f"\n chunk_{chunk_counter} (30sec) transcription time: {end_time - start_time:.2f} seconds")

total_end_time = time.perf_counter()
print("\n\nFull transcription done.")
print(f"\n Full transcription time: {total_end_time - total_start_time:.2f} seconds")
