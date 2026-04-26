import os
import torch
import numpy as np
import scipy.io.wavfile
import librosa
import safetensors.torch
from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
from pocket_tts.utils.utils import get_predefined_voice, download_if_necessary

# Paths
VOICE_PROMPT_PATH = r"D:\voice tts\dataset\processed\recording_1.wav"
OUTPUT_PATH = r"D:\voice tts\final_test_resampled.wav"

def generate_final_test():
    try:
        print("Loading Pocket TTS model...")
        model = TTSModel.load_model(language="english", quantize=True)
        
        # 1. Generate using Predefined Voice (ALBA)
        print("Generating with predefined voice 'alba'...")
        voice_path = get_predefined_voice("english", "alba")
        local_voice_path = download_if_necessary(voice_path)
        alba_state = safetensors.torch.load_file(local_voice_path)
        
        # Note: Predefined voices in pocket-tts are already states
        audio_alba = model.generate_audio(alba_state, "This is the predefined voice Alba. If you can hear this, the model is working.")
        
        # 2. Generate using Cloned Voice
        print("Generating with cloned voice from recording_1.wav...")
        wav, sr_in = audio_read(VOICE_PROMPT_PATH)
        wav_prompt = wav[:, :30*sr_in]
        voice_state_clone = model.get_state_for_audio_prompt(wav_prompt)
        audio_clone = model.generate_audio(voice_state_clone, "And this is the cloned voice from your audio file. Can you hear this too?")
        
        # Combine
        combined = torch.cat([audio_alba, torch.zeros(int(model.sample_rate * 1)), audio_clone])
        audio_np = combined.numpy()
        
        # NORMALIZE
        audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
        
        # RESAMPLE to 44.1kHz for compatibility
        target_sr = 44100
        print(f"Resampling from {model.sample_rate} to {target_sr}...")
        audio_resampled = librosa.resample(y=audio_np, orig_sr=model.sample_rate, target_sr=target_sr)
        
        # Save as 16-bit PCM
        audio_int16 = (audio_resampled * 32767).astype(np.int16)
        scipy.io.wavfile.write(OUTPUT_PATH, target_sr, audio_int16)
        print(f"Final test saved to {OUTPUT_PATH}")
        print(f"Max Amplitude: {np.abs(audio_int16).max()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_final_test()
