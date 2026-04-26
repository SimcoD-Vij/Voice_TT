import os
import torch
import numpy as np
import wave
import safetensors.torch
from pocket_tts import TTSModel
from pocket_tts.utils.utils import get_predefined_voice, download_if_necessary

def generate_english_proof_v2():
    try:
        print("Loading Pocket TTS (no quantization)...")
        model = TTSModel.load_model(language="english", quantize=False)
        
        print("Loading and unflattening Alba embedding...")
        vpath = get_predefined_voice("english", "alba")
        lpath = download_if_necessary(vpath)
        flat_data = safetensors.torch.load_file(lpath)
        
        # UNFLATTEN 'transformer.layers.0.self_attn/offset' -> {'transformer.layers.0.self_attn': {'offset': ...}}
        voice_state_alba = {}
        for k, v in flat_data.items():
            if "/" in k:
                main_key, sub_key = k.split("/")
                if main_key not in voice_state_alba:
                    voice_state_alba[main_key] = {}
                voice_state_alba[main_key][sub_key] = v
            else:
                voice_state_alba[k] = v
        
        print("Generating English proof with Alba...")
        text = "Hello. This is the predefined voice Alba. I am speaking English correctly."
        audio = model.generate_audio(voice_state_alba, text)
        
        audio_np = audio.numpy().flatten()
        audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        p = r"D:\voice tts\proof_english_alba_v2.wav"
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
    generate_english_proof_v2()
