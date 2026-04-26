import torch
from pocket_tts.data.audio import audio_read

PATH = r"D:\voice tts\sales_pitch_output_refined.wav"

try:
    wav, sr = audio_read(PATH)
    print(f"File: {PATH}")
    print(f"Shape: {wav.shape}")
    print(f"Sample Rate: {sr}")
    print(f"Max Amplitude: {wav.abs().max().item()}")
    print(f"Mean Amplitude: {wav.abs().mean().item()}")
    if wav.abs().max().item() < 1e-6:
        print("ALERT: The audio is SILENT (zeros or very quiet).")
    else:
        print("The audio contains non-zero data.")
except Exception as e:
    print(f"Failed to read audio: {e}")
