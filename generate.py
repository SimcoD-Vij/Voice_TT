"""
generate.py
===========
CLI tool to convert text → cloned voice audio using your saved voice prompt.

Usage:
    python generate.py --text "Hello world, this is my voice."
    python generate.py --text "..." --output my_clip.wav --play
    python generate.py --text "..." --language Chinese
    python generate.py --text "..." --no_cache   (rebuild voice prompt each call)

Quickstart:
    1. Run preprocess.py      (convert M4A → WAV + transcribe)
    2. Run build_voice_prompt.py  (create voice_prompt.safetensors)
    3. Run this script!
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
# Ensure local patched Qwen3-TTS is used
sys.path.insert(0, str(PROJECT_ROOT / "Qwen3-TTS"))
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = DATASET_DIR / "processed"
VOICE_PROMPT_PATH = DATASET_DIR / "voice_prompt.safetensors"
VOICE_META_PATH = DATASET_DIR / "voice_meta.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_model(model_id: str, use_cpu: bool = False):
    """Load model with FlashAttention 2 if available, else eager or cpu."""
    import torch
    from qwen_tts import Qwen3TTSModel

    local_path = PROJECT_ROOT / "models" / Path(model_id).name
    model_source = str(local_path) if local_path.exists() else model_id

    device = "cpu" if use_cpu else "cuda:0"
    dtype = torch.float32 if use_cpu else torch.bfloat16
    
    print(f"Loading model: {model_source} on {device}")
    
    if use_cpu:
        model = Qwen3TTSModel.from_pretrained(
            model_source,
            device_map="cpu",
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        print("[OK] Model loaded on CPU")
    else:
        try:
            model = Qwen3TTSModel.from_pretrained(
                model_source,
                device_map=device,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
            )
            print("[OK] FlashAttention 2 enabled")
        except Exception:
            model = Qwen3TTSModel.from_pretrained(
                model_source,
                device_map=device,
                torch_dtype=dtype,
                attn_implementation="eager",
            )
            print("[OK] Model loaded (eager attention)")
    return model


def load_voice_prompt(model, use_cpu: bool = False):
    """Load precomputed voice prompt from .pt file."""
    prompt_pt = DATASET_DIR / "voice_prompt.pt"
    if not prompt_pt.exists():
        print("[ERROR] voice_prompt.pt not found.")
        print("Run: python build_voice_prompt.py")
        sys.exit(1)

    print(f"Loading voice prompt from {prompt_pt.name}…")
    import torch
    device = "cpu" if use_cpu else ("cuda:0" if torch.cuda.is_available() else "cpu")
    voice_clone_prompt = torch.load(str(prompt_pt), map_location=device, weights_only=False)
    print(f"[OK] Voice prompt loaded for {device}")
    return voice_clone_prompt



def generate_audio(model, voice_clone_prompt, text: str, language: str = "auto", max_new_tokens: int = 2048) -> tuple:
    """Generate audio and return (waveform_numpy, sample_rate)."""
    import time
    t0 = time.time()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=max_new_tokens,   # CRITICAL: prevents infinite generation
    )
    elapsed = time.time() - t0
    audio_duration = len(wavs[0]) / sr
    rtf = elapsed / audio_duration if audio_duration > 0 else 0
    print(f"[OK] Generated {audio_duration:.1f}s audio in {elapsed:.2f}s (RTF={rtf:.2f}x)")
    return wavs[0], sr



def play_audio(wav_path: Path):
    """Attempt to play audio with system player (best-effort)."""
    import subprocess, platform
    if platform.system() == "Windows":
        os.startfile(str(wav_path))
    elif platform.system() == "Darwin":
        subprocess.Popen(["afplay", str(wav_path)])
    else:
        subprocess.Popen(["aplay", str(wav_path)])


def main():
    parser = argparse.ArgumentParser(
        description="Generate cloned-voice audio from text using Qwen3-TTS"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output WAV path (default: outputs/output_<timestamp>.wav)"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="auto",
        help="Language code: English|Chinese|Japanese|Korean|German|French|Russian|Portuguese|Spanish|Italian|auto"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Qwen3-TTS model ID or local path"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio after generation"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU instead of GPU"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" QWEN3-TTS VOICE GENERATOR")
    print("=" * 60)
    print(f"Text     : {args.text}")
    print(f"Language : {args.language}")
    print()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        output_path = OUTPUTS_DIR / f"output_{ts}.wav"

    # Load model
    model = load_model(args.model, use_cpu=args.cpu)

    # Load precomputed voice prompt
    voice_clone_prompt = load_voice_prompt(model, use_cpu=args.cpu)

    # Generate
    print(f"\nGenerating audio…")
    import soundfile as sf
    audio, sr = generate_audio(model, voice_clone_prompt, args.text, args.language)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr)
    print(f"[SAVED] {output_path}")

    if args.play:
        print("Playing audio…")
        play_audio(output_path)

    print()
    print("[DONE]")


if __name__ == "__main__":
    main()
