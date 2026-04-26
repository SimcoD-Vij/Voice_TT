from pocket_tts import TTSModel
import scipy.io.wavfile
import torch
import numpy as np
import os
from pocket_tts.data.audio import audio_read

# Paths
VOICE_PROMPT_PATH = r"D:\voice tts\dataset\processed\recording_1.wav"
OUTPUT_PATH = r"D:\voice tts\sales_pitch_v2.wav"

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
        print("Loading Pocket TTS model (unquantized)...")
        model = TTSModel.load_model(language="english", quantize=False)
        
        print(f"Reading FULL audio prompt: {VOICE_PROMPT_PATH}")
        wav, sr = audio_read(VOICE_PROMPT_PATH)
        
        # USE FULL AUDIO PROMPT for best possible cloning
        print(f"Cloning voice from {wav.shape[1]/sr:.1f} seconds of audio...")
        voice_state = model.get_state_for_audio_prompt(wav)
        
        print("Generating audio pitch...")
        audio = model.generate_audio(voice_state, SALES_PITCH)
        
        print(f"Saving audio to: {OUTPUT_PATH} (16-bit PCM)")
        audio_np = audio.numpy()
        audio_int16 = (audio_np.clip(-1, 1) * 32767).astype(np.int16)
        
        scipy.io.wavfile.write(OUTPUT_PATH, model.sample_rate, audio_int16)
        print("Generation complete!")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_pitch()
