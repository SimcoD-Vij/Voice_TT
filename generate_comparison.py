import os
import torch
import numpy as np
import wave
import safetensors.torch
from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.utils.utils import get_predefined_voice, download_if_necessary

def generate_comparison():
    try:
        print("Loading Pocket TTS (no quantization)...")
        model = TTSModel.load_model(language="english", quantize=False)
        
        # 1. Predefined English voice 'alba'
        print("Scren shot 1: Generating Alba...")
        vpath = get_predefined_voice("english", "alba")
        lpath = download_if_necessary(vpath)
        alba_state = safetensors.torch.load_file(lpath)
        # Fix: predefined voices need to be wrapped in a dict with VOICE_LATENTS key
        voice_state_alba = {"voice_latents": alba_state["voice_latents"] if "voice_latents" in alba_state else list(alba_state.values())[0]}
        
        audio_alba = model.generate_audio(voice_state_alba, "This is a predefined voice named Alba speaking clear English.")
        
        # 2. Cloned voice from recording_1.wav
        print("Scren shot 2: Generating Clone...")
        VOICE_PATH = r"D:\voice tts\dataset\processed\recording_1.wav"
        wav, sr_in = audio_read(VOICE_PATH)
        voice_state_clone = model.get_state_for_audio_prompt(wav[:, :20*sr_in])
        audio_clone = model.generate_audio(voice_state_clone, "This is a cloned voice from your recording one file.")
        
        # Save both separately
        for name, audio in [("alba", audio_alba), ("clone", audio_clone)]:
            audio_np = audio.numpy().flatten()
            # Normalize
            audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            p = f"D:\\voice tts\\test_{name}.wav"
            with wave.open(p, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(model.sample_rate)
                f.writeframes(audio_int16.tobytes())
            print(f"Saved {name} to {p}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_comparison()
