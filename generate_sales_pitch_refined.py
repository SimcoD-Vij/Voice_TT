from pocket_tts import TTSModel
import scipy.io.wavfile
import torch
import os
import sys
from pocket_tts.data.audio import audio_read

# Paths
VOICE_PROMPT_PATH = r"D:\voice tts\dataset\processed\recording_1.wav"
OUTPUT_PATH = r"D:\voice tts\sales_pitch_output_refined.wav"

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
        print("Loading Pocket TTS model...")
        # Using quantize=True for speed, but let's ensure it's loaded correctly
        model = TTSModel.load_model(language="english", quantize=True)
        
        print(f"Reading audio prompt: {VOICE_PROMPT_PATH}")
        wav, sr = audio_read(VOICE_PROMPT_PATH)
        
        # INCREASED DURATION: Use first 20 seconds for better voice capture
        duration_to_use = 20
        limit = duration_to_use * sr
        if wav.shape[1] > limit:
            wav_subset = wav[:, :limit]
            print(f"Using first {duration_to_use} seconds for cloning.")
        else:
            wav_subset = wav
            print(f"Using entire audio prompt for cloning ({wav.shape[1]/sr:.1f}s).")
        
        print(f"Cloning voice...")
        voice_state = model.get_state_for_audio_prompt(wav_subset)
        
        print("Generating audio pitch (refined)...")
        # In the previous run, we used default parameters.
        # Let's try slightly increasing the temperature for more natural variance 
        # if the library supports it via kwargs (it passes them to FlowLMModel.lsd_decode).
        # FlowLMModel.lsd_decode has temperature=1.0 by default.
        audio = model.generate_audio(voice_state, SALES_PITCH)
        
        print(f"Saving audio to: {OUTPUT_PATH}")
        scipy.io.wavfile.write(OUTPUT_PATH, model.sample_rate, audio.numpy())
        print("Generation complete!")
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_pitch()
