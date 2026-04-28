import asyncio
import os
import sys
import psutil

def print_memory(label):
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    print(f"[{label}] System Free: {mem.available / 1024 / 1024:.2f} MB | Process RSS: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Add dograh and pipecat to path (PRIORITIZE LOCAL)
sys.path.insert(0, os.path.join(os.getcwd(), "dograh", "pipecat", "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "dograh"))

from pipecat.services.pocket_tts import PocketTTSService
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

async def test_pocket_tts():
    print_memory("Start")
    model_path = r"D:\voice tts\pocket-tts\pocket_tts\config\english.yaml"
    voice_id = r"D:\voice tts\dataset\processed\recording_1.wav"
    
    try:
        service = PocketTTSService(
            model_path=model_path,
            voice_id=voice_id,
            use_enhanced_pipeline=True,
            quantize=True
        )
        
        print(f"--- Testing Pocket TTS Integration ---")
        print_memory("Service Created")
        
        text = "Hello."
        
        async for frame in service.run_tts(text, "test-ctx"):
            if isinstance(frame, TTSStartedFrame):
                print("TTS Started")
                print_memory("During Generation")
            elif isinstance(frame, TTSAudioRawFrame):
                pass
            elif isinstance(frame, TTSStoppedFrame):
                print("TTS Stopped")
                print_memory("After Generation")

        print("SUCCESS: Test completed.")
            
    except Exception as e:
        print(f"Test crashed with error: {e}")
        print_memory("After Crash")

if __name__ == "__main__":
    asyncio.run(test_pocket_tts())
