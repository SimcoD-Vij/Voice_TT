import os
import sys
import torch
import numpy as np
import scipy.io.wavfile

# HACK: Bypass torchvision error by faking its module
import types
sys.modules['torchvision'] = types.ModuleType('torchvision')
sys.modules['torchvision.ops'] = types.ModuleType('torchvision.ops')
# Force 'transformers' to think torchvision is not available or broken
# so it doesn't try to import it for the processor

# Add Qwen3-TTS to path
sys.path.append(r"D:\voice tts\Qwen3-TTS")

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Paths
MODEL_PATH = r"D:\voice tts\models\Qwen3-TTS-12Hz-0.6B-Base"
VOICE_PROMPT_PATH = r"D:\voice tts\dataset\processed\recording_1.wav"
OUTPUT_PATH = r"D:\voice tts\sales_pitch_qwen_v2.wav"

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
        print(f"Loading Qwen3-TTS model (CPU mode for stability)...")
        # Ensure we use CPU if CUDA is broken
        model_wrapper = Qwen3TTSModel.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        print(f"Generating voice clone...")
        # Use first 30 seconds for Qwen as well
        wavs, sr = model_wrapper.generate_voice_clone(
            text=SALES_PITCH,
            language="English",
            ref_audio=VOICE_PROMPT_PATH,
            x_vector_only_mode=True
        )
        
        if wavs:
            print(f"Saving to {OUTPUT_PATH}...")
            audio = wavs[0]
            # Normalize and convert
            audio_norm = audio / (np.abs(audio).max() + 1e-7)
            audio_int16 = (audio_norm * 32767).astype(np.int16)
            scipy.io.wavfile.write(OUTPUT_PATH, sr, audio_int16)
            print("Generation complete!")
        else:
            print("No audio generated.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_qwen_pitch()
