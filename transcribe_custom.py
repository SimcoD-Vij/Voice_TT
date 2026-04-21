import os
from f5_tts.api import F5TTS
import pydub
from pydub import AudioSegment

def transcribe_ref():
    ref_audio_path = r"D:\voice tts\dataset\processed\recording_1.wav"
    
    print(f"Loading first 15 seconds of {ref_audio_path}...")
    audio = AudioSegment.from_file(ref_audio_path)
    audio_15s = audio[:15000]
    
    temp_ref = "temp_ref_15s.wav"
    audio_15s.export(temp_ref, format="wav")
    
    print("Transcribing...")
    hf_cache_dir = r"D:\hf_cache"
    if not os.path.exists(hf_cache_dir):
        os.makedirs(hf_cache_dir)
    f5tts = F5TTS(hf_cache_dir=hf_cache_dir)
    # transcribe uses whisper model which will be downloaded
    ref_text = f5tts.transcribe(temp_ref)
    
    print(f"Transcript: {ref_text}")
    
    # Clean up
    if os.path.exists(temp_ref):
        os.remove(temp_ref)

if __name__ == "__main__":
    transcribe_ref()
