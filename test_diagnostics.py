import torch
from pocket_tts.data.audio import audio_read
import numpy as np

PATH = r"D:\voice tts\dataset\processed\recording_1.wav"

try:
    wav, sr = audio_read(PATH)
    print(f"Source: {PATH}")
    print(f"Max Amplitude: {wav.abs().max().item()}")
    print(f"Mean Amplitude: {wav.abs().mean().item()}")
except Exception as e:
    print(f"Error: {e}")

# Corrected test for predefined voice
from pocket_tts import TTSModel
import scipy.io.wavfile

def test_predefined_v2():
    try:
        model = TTSModel.load_model(language="english", quantize=True)
        # Load embedding directly as a state dict
        from pocket_tts.utils.utils import get_predefined_voice, download_if_necessary
        import safetensors.torch
        
        print("Downloading predefined voice embedding...")
        voice_path = get_predefined_voice("english", "alba")
        local_path = download_if_necessary(voice_path)
        
        print(f"Loading embedding from {local_path}...")
        voice_state = safetensors.torch.load_file(local_path)
        
        print("Generating audio...")
        audio = model.generate_audio(voice_state, "Can you hear this voice?")
        
        out_path = r"D:\voice tts\test_predefined_v2.wav"
        scipy.io.wavfile.write(out_path, model.sample_rate, (audio.numpy().clip(-1, 1) * 32767).astype(np.int16))
        print(f"Saved to {out_path}. Max: {audio.abs().max().item()}")
    except Exception as e:
        print(f"Error in test: {e}")

if __name__ == "__main__":
    test_predefined_v2()
