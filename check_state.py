from pocket_tts import TTSModel
from pocket_tts.data.audio import audio_read
import torch
import os

def check_voice_state():
    try:
        model = TTSModel.load_model(language="english")
        VOICE_PATH = r"D:\voice tts\dataset\processed\recording_1.wav"
        wav, sr_in = audio_read(VOICE_PATH)
        voice_state = model.get_state_for_audio_prompt(wav[:, :1*sr_in])
        print(f"Voice state type: {type(voice_state)}")
        print(f"Voice state keys: {voice_state.keys()}")
        for k, v in voice_state.items():
            if isinstance(v, dict):
                print(f"Sub-key {k} keys: {v.keys()}")
            else:
                print(f"Key {k} type: {type(v)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_voice_state()
