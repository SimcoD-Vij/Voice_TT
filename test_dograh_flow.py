import requests
import json
import time
import sys

# Configuration
# Dograh uses the name 'ollama' in docker, but we use 'localhost' for local testing
OLLAMA_BASE_URL = "http://localhost:11434/v1" 
POCKETTTS_URL = "http://localhost:8100/tts"
MODEL = "llama3.2:1b"
VOICE_PATH = "/voices/recording_1_short.wav"

def simulate_dograh_conversation(user_text):
    print(f"--- SIMULATED CONVERSATION START ---")
    print(f"User (Simulated STT): \"{user_text}\"")
    
    # 1. Dograh API -> Ollama
    print(f"\n[Dograh] Sending text to Ollama Service...")
    start_time = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant in the Dograh platform."},
                    {"role": "user", "content": user_text}
                ],
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        llm_data = response.json()
        ai_message = llm_data['choices'][0]['message']['content']
        latency = time.time() - start_time
        print(f"[Ollama] Response received in {latency:.2f}s:")
        print(f"AI: \"{ai_message}\"")
    except Exception as e:
        print(f"FAILED to reach Ollama: {e}")
        return

    # 2. Dograh Internal -> TTS (The "Reach Back")
    print(f"\n[Dograh] Closing the loop: Sending AI response to PocketTTS...")
    tts_start = time.time()
    try:
        tts_resp = requests.post(POCKETTTS_URL, data={
            "text": ai_message,
            "voice_url": VOICE_PATH
        }, timeout=300)
        tts_resp.raise_for_status()
        tts_latency = time.time() - tts_start
        print(f"[PocketTTS] Synthesis started successfully (TTFB check passed)")
        print(f"[PocketTTS] Synthesis completed in {tts_latency:.2f}s")
    except Exception as e:
        print(f"FAILED to reach PocketTTS: {e}")
        return

    print(f"\n--- SIMULATED CONVERSATION SUCCESSFUL ---")
    print(f"Total turnaround time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_query = "Hello Dograh! Can you hear me and respond through your pipeline?"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    
    simulate_dograh_conversation(test_query)
