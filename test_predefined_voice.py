from pocket_tts import TTSModel
import scipy.io.wavfile
import numpy as np
import os

OUTPUT_PATH = r"D:\voice tts\test_predefined.wav"

def test_predefined():
    try:
        print("Loading Pocket TTS model...")
        model = TTSModel.load_model(language="english", quantize=True)
        
        print("Using predefined voice 'alba'...")
        # Get predefined voice state
        # In pocket_tts, predefined voices are usually strings or loaded via get_state_for_voice
        # Let's check how main.py handles it
        from pocket_tts.utils.utils import get_predefined_voice
        voice_path = get_predefined_voice("english", "alba")
        voice_state = model.get_state_for_audio_prompt(voice_path)
        
        print("Generating test audio with predefined voice...")
        audio = model.generate_audio(voice_state, "Hello! This is a test of the predefined voice. Can you hear me clearly?")
        
        print(f"Saving to {OUTPUT_PATH}...")
        audio_np = audio.numpy()
        audio_int16 = (audio_np.clip(-1, 1) * 32767).astype(np.int16)
        scipy.io.wavfile.write(OUTPUT_PATH, model.sample_rate, audio_int16)
        
        print(f"Test complete. Max amplitude: {np.abs(audio_int16).max()}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_predefined()
