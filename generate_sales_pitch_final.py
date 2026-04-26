from pocket_tts import TTSModel
import scipy.io.wavfile
import torch
import numpy as np
import os
import sys
from pocket_tts.data.audio import audio_read

# Paths
VOICE_PROMPT_PATH = r"D:\voice tts\dataset\processed\recording_1.wav"
OUTPUT_PATH = r"D:\voice tts\sales_pitch_final.wav"

# SALES_PITCH
SALES_PITCH = """
Hello there! I'm calling to introduce you to the XOptimus Smart Charger. 
Are you tired of slow charging and bulky adapters? Our new smart charger is 
designed for maximum efficiency and portability. It uses advanced GAN technology 
to deliver lightning-fast charging while staying cool and compact. 
It's the only charger you'll ever need for your phone, laptop, and tablet. 
We're currently offering a special discount for new customers. 
Would you like me to send you more details?
"""

def generate_pitch():
    try:
        print("Loading Pocket TTS model (no quantization)...")
        # Load without quantization to avoid any NaN/Zero issues on certain CPUs
        model = TTSModel.load_model(language="english", quantize=False)
        
        print(f"Reading audio prompt: {VOICE_PROMPT_PATH}")
        wav, sr = audio_read(VOICE_PROMPT_PATH)
        
        # Use first 15 seconds
        duration_to_use = 15
        limit = duration_to_use * sr
        wav_subset = wav[:, :min(wav.shape[1], limit)]
        
        print(f"Cloning voice...")
        voice_state = model.get_state_for_audio_prompt(wav_subset)
        
        print("Generating audio pitch...")
        audio = model.generate_audio(voice_state, SALES_PITCH)
        
        print(f"Saving audio to: {OUTPUT_PATH} (as 16-bit PCM)")
        # Convert to 16-bit PCM
        audio_np = audio.numpy()
        audio_int16 = (audio_np.clip(-1, 1) * 32767).astype(np.int16)
        
        scipy.io.wavfile.write(OUTPUT_PATH, model.sample_rate, audio_int16)
        print("Generation complete!")
        
        # Quick silence check
        max_val = np.abs(audio_int16).max()
        print(f"Max audio value (int16): {max_val}")
        if max_val < 100:
            print("WARNING: Generated audio seems very quiet or silent.")
            
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_pitch()
