import os
import torch
import numpy as np
import wave
from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read

# Paths
VOICE_PROMPT_PATH = r"D:\voice tts\dataset\processed\recording_1.wav"
OUTPUT_PATH = r"D:\voice tts\sales_pitch_final_v3.wav"

def generate_pitch_v3():
    try:
        print("Loading Pocket TTS model...")
        model = TTSModel.load_model(language="english", quantize=True)
        
        print(f"Reading prompt (first 20s): {VOICE_PROMPT_PATH}")
        wav, sr_in = audio_read(VOICE_PROMPT_PATH)
        wav_prompt = wav[:, :20*sr_in]
        
        print(f"Cloning voice...")
        voice_state = model.get_state_for_audio_prompt(wav_prompt)
        
        print("Generating audio pitch...")
        SALES_PITCH = """
        Hello there! I'm calling to introduce you to the XOptimus Smart Charger. 
        Are you tired of slow charging and bulky adapters? Our new smart charger is 
        designed for maximum efficiency and portability. It uses advanced GAN technology 
        to deliver lightning-fast charging while staying cool and compact. 
        It's the only charger you'll ever need for your phone, laptop, and tablet. 
        We're currently offering a special discount for new customers. 
        Would you like me to send you more details?
        """
        
        audio = model.generate_audio(voice_state, SALES_PITCH)
        audio_np = audio.numpy().flatten()
        
        # NORMALIZE
        max_abs = np.abs(audio_np).max()
        print(f"Raw Max Amplitude: {max_abs}")
        if max_abs > 1e-7:
            audio_np = audio_np / max_abs
            print("Normalized to 1.0 peak.")
        else:
            print("WARNING: Audio is silent.")

        # SAVE USING wave MODULE (Standard PCM 16-bit)
        print(f"Saving to {OUTPUT_PATH} using 'wave' module...")
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        with wave.open(OUTPUT_PATH, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2) # 16-bit
            f.setframerate(model.sample_rate)
            f.writeframes(audio_int16.tobytes())
            
        print(f"Generation complete! File: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_pitch_v3()
