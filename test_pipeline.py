import requests
import sys
import time

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
POCKETTTS_URL = "http://localhost:8100/tts"
MODEL = "llama3.2:1b"
VOICE_PATH = "/voices/recording_1_short.wav"
PROMPT = "Explain in one sentence why streaming reduces latency."
OUTPUT_FILE = "test_pipeline_latency.wav"
TIMEOUT = 300

def test_pipeline_with_latency():
    total_start = time.time()
    
    # 1. Ollama Latency
    print(f"1. Requesting text from Ollama ({MODEL})...")
    ollama_start = time.time()
    try:
        ollama_resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": PROMPT,
            "stream": False
        })
        ollama_resp.raise_for_status()
        ollama_end = time.time()
        text = ollama_resp.json().get("response", "")
        print(f"   - LLM text generated in: {ollama_end - ollama_start:.2f}s")
        print(f"   - Generated text: \"{text}\"")
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        sys.exit(1)

    # 2. PocketTTS Latency (TTFB and Total)
    print(f"\n2. Sending text to PocketTTS for synthesis...")
    tts_start = time.time()
    ttfb = None
    total_audio_bytes = 0
    
    try:
        tts_resp = requests.post(POCKETTTS_URL, data={
            "text": text,
            "voice_url": VOICE_PATH
        }, stream=True, timeout=TIMEOUT)
        tts_resp.raise_for_status()
        
        with open(OUTPUT_FILE, "wb") as f:
            for chunk in tts_resp.iter_content(chunk_size=8192):
                if chunk:
                    if ttfb is None:
                        ttfb = time.time() - tts_start
                        print(f"   - Time to First Byte (TTFB): {ttfb:.2f}s")
                    f.write(chunk)
                    total_audio_bytes += len(chunk)
        
        tts_end = time.time()
        print(f"   - Total synthesis time: {tts_end - tts_start:.2f}s")
        print(f"   - Total audio bytes saved: {total_audio_bytes}")
        
    except Exception as e:
        print(f"Error calling PocketTTS: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response body: {e.response.text}")
        sys.exit(1)

    total_end = time.time()
    print(f"\nTotal Pipeline Processing Time: {total_end - total_start:.2f}s")
    print(f"Final audio saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    test_pipeline_with_latency()
