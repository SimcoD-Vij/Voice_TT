from pocket_tts import TTSModel, export_model_state
import scipy.io.wavfile
import numpy as np
import wave
import os

# Paths
CLEAN_AUDIO = r"D:\voice tts\recording_clean.wav"
EMBEDDING_PATH = r"D:\voice tts\recording_1.safetensors"
OUTPUT_PATH = r"D:\voice tts\sales_pitch_optimized.wav"

SALES_PITCH = """
Hello there! I'm calling to introduce you to the XOptimus Smart Charger. 
Are you tired of slow charging and bulky adapters? Our new smart charger is 
designed for maximum efficiency and portability. It uses advanced GAN technology 
to deliver lightning-fast charging while staying cool and compact. 
It's the only charger you'll ever need for your phone, laptop, and tablet. 
We're currently offering a special discount for new customers. 
Would you like me to send you more details?
"""

def run_optimized_cloning():
    try:
        # Step 5: Export Voice to .safetensors
        print("Loading Pocket TTS model and exporting embedding...")
        model = TTSModel.load_model(language="english", quantize=True)
        
        # Convert WAV -> voice embedding (Step 5)
        print(f"Processing clean audio: {CLEAN_AUDIO}")
        voice_state = model.get_state_for_audio_prompt(CLEAN_AUDIO)
        
        print(f"Exporting embedding to {EMBEDDING_PATH}...")
        export_model_state(voice_state, EMBEDDING_PATH)
        
        # Step 6: Use Shared Embedding for Generation
        print("Generating optimized audio...")
        audio = model.generate_audio(voice_state, SALES_PITCH)
        audio_np = audio.numpy().flatten()
        
        # NORMALIZE
        audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        print(f"Saving to {OUTPUT_PATH}...")
        with wave.open(OUTPUT_PATH, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(model.sample_rate)
            f.writeframes(audio_int16.tobytes())
            
        print("Optimized generation complete!")
        print(f"File: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_optimized_cloning()
