import os
import torch
import numpy as np
import wave
import safetensors.torch
from pocket_tts import TTSModel
from pocket_tts.utils.utils import get_predefined_voice, download_if_necessary

def generate_english_proof():
    try:
        print("Loading Pocket TTS (no quantization)...")
        model = TTSModel.load_model(language="english", quantize=False)
        
        # 1. Predefined English voice 'alba'
        print("Loading Alba embedding...")
        vpath = get_predefined_voice("english", "alba")
        lpath = download_if_necessary(vpath)
        alba_data = safetensors.torch.load_file(lpath)
        
        # In pocket-tts, predefined voice safetensors have keys like 'transformer.layers...'
        # BUT generate_audio expects a dict where 'voice_latents' (or similar) is present
        # Actually, let's look at the model's forward pass.
        
        # If I can't find the key, I'll just pass the whole dict
        voice_state_alba = alba_data
        
        print("Generating English proof with Alba...")
        text = "Hello. This is the predefined voice Alba. I am speaking English correctly."
        audio = model.generate_audio(voice_state_alba, text)
        
        audio_np = audio.numpy().flatten()
        audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        p = r"D:\voice tts\proof_english_alba.wav"
        with wave.open(p, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(model.sample_rate)
            f.writeframes(audio_int16.tobytes())
        print(f"Success! Saved to {p}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_english_proof()
