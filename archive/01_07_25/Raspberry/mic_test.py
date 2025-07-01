import sounddevice as sd

device = 0  # USB mic
samplerate = 48000
channels = 1

try:
    print("Testing microphone input...")
    duration = 2  # seconds
    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=channels,
                   dtype='float32',
                   device=device)
    sd.wait()
    print("Success! Microphone is working.")
except Exception as e:
    print("Error:", e)


import sounddevice as sd

device = 1  # USB mic

# Get microphone device info
device_info = sd.query_devices(device)
print("Device info:", device_info)

# Extract valid sample rate and channels
samplerate = device_info.get('default_samplerate', 44100)
if not isinstance(samplerate, (int, float)):
    raise ValueError("Invalid samplerate detected")

samplerate = int(samplerate)
channels = device_info['max_input_channels']

try:
    print(f"Testing microphone input at {samplerate} Hz with {channels} channel(s)...")
    duration = 2  # seconds
    frames = int(duration * samplerate)
    print(f"Recording {frames} frames")

    audio = sd.rec(frames,
                   samplerate=samplerate,
                   channels=channels,
                   dtype='float32',
                   device=device)
    sd.wait()
    print("Success! Microphone is working.")
except Exception as e:
    print("Error:", e)
