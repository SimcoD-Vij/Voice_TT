import os
import sys
import torch
import numpy as np
import wave

# Add Qwen3-TTS to path
sys.path.append(r"D:\voice tts\Qwen3-TTS")

# Import classes DIRECTLY to avoid AutoProcessor (which triggers torchvision/nms error)
from qwen_tts.core.models import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor, Qwen3TTSConfig

# Paths
MODEL_PATH = r"D:\voice tts\models\Qwen3-TTS-12Hz-0.6B-Base"
VOICE_PROMPT_PATH = r"D:\voice tts\dataset\processed\recording_1.wav"
OUTPUT_PATH = r"D:\voice tts\sales_pitch_qwen_final.wav"

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

def generate_qwen_pitch():
    try:
        print(f"Loading Qwen3-TTS (Manual loading to bypass torchvision)...")
        # Load processor directly
        processor = Qwen3TTSProcessor.from_pretrained(MODEL_PATH)
        
        # Load model directly
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32, # CPU mode
            device_map="cpu"
        )
        
        # We need the wrapper class for convenience, but we'll fulfill its constructor manually
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        model_wrapper = Qwen3TTSModel(model=model, processor=processor)
        
        print(f"Cloning voice from {VOICE_PROMPT_PATH}...")
        # Use first 30 seconds for quality
        wavs, sr = model_wrapper.generate_voice_clone(
            text=SALES_PITCH,
            language="English",
            ref_audio=VOICE_PROMPT_PATH,
            x_vector_only_mode=True
        )
        
        if wavs:
            print(f"Saving to {OUTPUT_PATH} (16-bit PCM)...")
            audio = wavs[0]
            # Normalize
            audio = audio / (np.abs(audio).max() + 1e-7)
            audio_int16 = (audio * 32767).astype(np.int16)
            
            with wave.open(OUTPUT_PATH, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sr)
                f.writeframes(audio_int16.tobytes())
            print("Success! Final output generated.")
            print(f"File: {OUTPUT_PATH}")
        else:
            print("No audio produced.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_qwen_pitch()
