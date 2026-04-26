import scipy.io.wavfile
import numpy as np

# Verify if a standard beep can be heard
def test_beep():
    sr = 24000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # 440Hz Sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    audio_int16 = (audio * 32767).astype(np.int16)
    out_path = r"D:\voice tts\test_beep.wav"
    scipy.io.wavfile.write(out_path, sr, audio_int16)
    print(f"Beep saved to {out_path}")

if __name__ == "__main__":
    test_beep()
