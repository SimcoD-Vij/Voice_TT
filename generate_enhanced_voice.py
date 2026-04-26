from pocket_tts import TTSModel
import wave
import numpy as np
import os
import torch

# Paths
ENHANCED_PROMPT = r"D:\voice tts\recording_enhanced.wav"
RAW_OUTPUT = r"D:\voice tts\natural_output.wav"

# Human-like text with prosody markers (commas, ellipses)
SENTENCES = [
    "Hello there! I'm calling to introduce you to... the XOptimus Smart Charger.",
    "Are you tired of slow charging, and bulky adapters?",
    "Well... our new smart charger is designed for maximum efficiency, and portability.",
    "It uses advanced G A N technology to deliver lightning-fast charging, while staying cool and compact.",
    "It is... the only charger you will ever need, for your phone, laptop, and tablet.",
    "We are currently offering a special discount for new customers.",
    "Would you like me to send you... more details?"
]

def generate_enhanced():
    try:
        print("Loading Pocket TTS model...")
        model = TTSModel.load_model(language="english", quantize=True)
        
        print(f"Loading enhanced voice prompt: {ENHANCED_PROMPT}")
        voice_state = model.get_state_for_audio_prompt(ENHANCED_PROMPT)
        
        chunks = []
        sample_rate = model.sample_rate
        
        for i, s in enumerate(SENTENCES):
            print(f"Generating sentence {i+1}/{len(SENTENCES)}: {s}")
            audio = model.generate_audio(voice_state, s)
            chunks.append(audio.numpy().flatten())
            
            # Add 300ms silence gap between sentences
            silence = np.zeros(int(sample_rate * 0.3))
            chunks.append(silence)
            
        print("Stitching audio chunks...")
        final_audio = np.concatenate(chunks)
        
        # Normalize
        final_audio = final_audio / (np.abs(final_audio).max() + 1e-7)
        audio_int16 = (final_audio * 32767).astype(np.int16)
        
        print(f"Saving raw natural output to {RAW_OUTPUT}...")
        with wave.open(RAW_OUTPUT, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(audio_int16.tobytes())
            
        print("Layer 2 complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_enhanced()
