import os
import torch
import numpy as np
import wave
from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read

# Paths
VOICE_PROMPT_PATH = r"D:\voice tts\dataset\processed\recording_1.wav"
OUTPUT_PATH = r"D:\voice tts\diagnostic_audio.wav"

def generate_diagnostic():
    try:
        print("Loading Pocket TTS model...")
        model = TTSModel.load_model(language="english", quantize=True)
        
        print(f"Reading prompt (first 10s): {VOICE_PROMPT_PATH}")
        wav, sr_in = audio_read(VOICE_PROMPT_PATH)
        wav_prompt = wav[:, :10*sr_in]
        
        print(f"Cloning voice...")
        voice_state = model.get_state_for_audio_prompt(wav_prompt)
        
        print("Generating voice sample 'Hello world'...")
        audio_voice = model.generate_audio(voice_state, "Hello World.")
        voice_np = audio_voice.numpy().flatten()
        
        # Create a BEEP (440Hz, 1s)
        sr = model.sample_rate
        t = np.linspace(0, 1.0, sr)
        beep_np = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Combine: BEEP (1s) + SILENCE (1s) + VOICE
        silence_np = np.zeros(sr)
        combined_np = np.concatenate([beep_np, silence_np, voice_np])
        
        # NORMALIZE
        combined_np = combined_np / (np.abs(combined_np).max() + 1e-7)
        
        print(f"Saving to {OUTPUT_PATH}...")
        audio_int16 = (combined_np * 32767).astype(np.int16)
        
        with wave.open(OUTPUT_PATH, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(audio_int16.tobytes())
            
        print("Diagnostic complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_diagnostic()
