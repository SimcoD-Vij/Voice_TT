import os
import argparse
import sys
from pathlib import Path

# Add Qwen3-TTS to path if not installed - ENSURE IT COMES FIRST
PROJECT_ROOT = Path(__file__).parent
QWEN_TTS_PATH = PROJECT_ROOT / "Qwen3-TTS"
if QWEN_TTS_PATH.exists():
    sys.path.insert(0, str(QWEN_TTS_PATH))

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset"
VOICE_PROMPT_PATH = DATASET_DIR / "voice_prompt.pt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
    return model

def main():
    text = (
        "Hello! Are you tired of your smartphone battery dying just when you need it most? "
        "Meet XOptimus, the world's smartest AI-powered charger. "
        "Made in India for your high-end devices, XOptimus doesn't just charge—it protects. "
        "With intelligent monitoring and custom modes for gaming and daily use, it can double your battery's lifespan. "
        "Don't let your thousand-dollar phone be ruined by a ten-dollar charger. "
        "Switch to XOptimus today for just fourteen ninety-nine rupees. Your battery will thank you."
    )
    
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    output_path = OUTPUTS_DIR / "xoptimus_latest_pitch.wav"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(model_id)

    # Load voice prompt
    if not VOICE_PROMPT_PATH.exists():
        print(f"[ERROR] {VOICE_PROMPT_PATH} not found.")
        return

    print(f"Loading voice prompt from {VOICE_PROMPT_PATH.name}...")
    voice_clone_prompt = torch.load(str(VOICE_PROMPT_PATH), map_location="cuda:0" if torch.cuda.is_available() else "cpu", weights_only=False)

    # Generate
    print("Generating audio...")
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="auto",
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=2048,
    )

    # Save
    sf.write(str(output_path), wavs[0], sr)
    print(f"[SAVED] {output_path}")

if __name__ == "__main__":
    main()
