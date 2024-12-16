import whisper
import numpy as np
import os
from tqdm import tqdm
import json

model = whisper.load_model("large-v3")

files = ["egypt.wav", "jordan.wav", "palestine.wav"]

# Path to the directory containing segmented WAV files
segmented_wavs_folder = "data"

for wav_file in files:
    result = model.transcribe(f"{segmented_wavs_folder}/{wav_file}", carry_initial_prompt=True, initial_prompt="")
    print(result['text'])

