"""
build_voice_prompt.py — Voice Clone Prompt Builder
====================================================
Reads a processed WAV + transcript, TRIMS to 10-15 seconds of audio
(with matching transcript), then builds a reusable voice prompt.

WHY WE CLIP:
  Qwen3-TTS uses KV-cache for the reference audio. A 3-min recording
  creates a ~10,000 token KV cache → generation takes HOURS.
  A 10-second clip creates a ~120 token cache → generation is FAST.
  Voice identity is fully captured in 10-15 seconds. More ≠ better.

Usage:
    python build_voice_prompt.py
    python build_voice_prompt.py --ref recording_2
    python build_voice_prompt.py --ref_dur 12       # seconds of audio
    python build_voice_prompt.py --x_vector_only    # fastest mode
"""

import os, sys, json, time
from pathlib import Path

PROJECT_ROOT   = Path(__file__).parent
DATASET_DIR    = PROJECT_ROOT / "dataset"
PROCESSED_DIR  = DATASET_DIR / "processed"
TRIMMED_DIR    = DATASET_DIR / "trimmed"
MODELS_DIR     = PROJECT_ROOT / "models"
VOICE_PROMPT_PATH = DATASET_DIR / "voice_prompt.pt"
VOICE_META_PATH   = DATASET_DIR / "voice_meta.json"

# How many seconds of audio to use as reference (10-15 is ideal)
DEFAULT_REF_DURATION = 12.0


def _add_conda_to_path():
    conda_env = Path(sys.executable).parent
    for p in [conda_env / "Library" / "bin", conda_env / "Scripts", conda_env]:
        s = str(p)
        if s not in os.environ.get("PATH", ""):
            os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")


def trim_wav(src: Path, dst: Path, duration: float) -> Path:
    """Trim WAV to `duration` seconds using soundfile (no ffmpeg needed)."""
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(str(src))
    samples = int(duration * sr)
    if len(data) <= samples:
        print(f"  Audio shorter than {duration}s — using full clip ({len(data)/sr:.1f}s)")
        dst.write_bytes(src.read_bytes())
        return dst
    trimmed = data[:samples]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), trimmed, sr)
    print(f"  Trimmed to {duration}s → {dst.name}")
    return dst


def trim_transcript(text: str, max_chars: int = 150) -> str:
    """Keep only the first complete sentence(s) within max_chars."""
    if len(text) <= max_chars:
        return text
    # Try to end on a sentence boundary
    chunk = text[:max_chars]
    for sep in ['. ', '! ', '? ']:
        idx = chunk.rfind(sep)
        if idx > 20:
            return chunk[:idx + 1].strip()
    # Fall back to word boundary
    idx = chunk.rfind(' ')
    return (chunk[:idx] if idx > 0 else chunk).strip()


def find_best_reference():
    import soundfile as sf
    wav_files = sorted(PROCESSED_DIR.glob("*.wav"))
    if not wav_files:
        return None, None

    best_wav, best_txt, best_dur = None, None, 0
    print("Available clips:")
    for wav in wav_files:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"  {wav.name}: no transcript")
            continue
        transcript = txt.read_text(encoding="utf-8").strip()
        if not transcript:
            continue
        try:
            dur = sf.info(str(wav)).duration
        except Exception:
            dur = 0
        preview = transcript[:60] + "…" if len(transcript) > 60 else transcript
        print(f"  {wav.name}: {dur:.1f}s — {preview}")
        if dur > best_dur:
            best_dur, best_wav, best_txt = dur, wav, txt

    return best_wav, best_txt


def load_model(model_id: str):
    import torch
    from qwen_tts import Qwen3TTSModel
    local_path = MODELS_DIR / Path(model_id).name
    source = str(local_path) if local_path.exists() else model_id
    print(f"\nLoading model: {source}")
    for attn in ["flash_attention_2", "eager"]:
        try:
            m = Qwen3TTSModel.from_pretrained(
                source, device_map="cuda:0",
                torch_dtype=torch.bfloat16, attn_implementation=attn,
            )
            print(f"[OK] Model loaded (attn={attn})")
            return m
        except Exception as e:
            if attn == "eager":
                raise
            print(f"  {attn} unavailable, trying eager…")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--ref", default=None, help="Clip name e.g. recording_2")
    parser.add_argument("--ref_dur", type=float, default=DEFAULT_REF_DURATION,
                        help=f"Seconds of audio to use as reference (default: {DEFAULT_REF_DURATION})")
    parser.add_argument("--x_vector_only", action="store_true",
                        help="No transcript needed — fastest build, slightly less accurate clone")
    parser.add_argument("--output", default=str(VOICE_PROMPT_PATH))
    args = parser.parse_args()

    print("=" * 60)
    print("  VOICE PROMPT BUILDER")
    print(f"  Reference duration: {args.ref_dur}s (short = fast generation)")
    print("=" * 60)

    _add_conda_to_path()

    if not PROCESSED_DIR.exists():
        print("[ERROR] processed/ not found. Run: python preprocess.py")
        sys.exit(1)

    # Select reference clip
    if args.ref:
        wav_path = PROCESSED_DIR / f"{args.ref}.wav"
        txt_path = PROCESSED_DIR / f"{args.ref}.txt"
        if not wav_path.exists():
            print(f"[ERROR] {wav_path} not found")
            sys.exit(1)
    else:
        print("\nSelecting best reference clip…")
        wav_path, txt_path = find_best_reference()
        if wav_path is None:
            print("[ERROR] No valid WAV+transcript pairs. Run: python preprocess.py")
            sys.exit(1)

    full_transcript = ""
    if txt_path and txt_path.exists():
        full_transcript = txt_path.read_text(encoding="utf-8").strip()

    # ──────────────────────────────────────────────────────────────
    # CRITICAL: Trim the audio and transcript to keep KV cache tiny!
    # ──────────────────────────────────────────────────────────────
    TRIMMED_DIR.mkdir(parents=True, exist_ok=True)
    trimmed_wav = TRIMMED_DIR / f"ref_{args.ref_dur:.0f}s.wav"
    print(f"\nTrimming reference audio to {args.ref_dur}s…")
    trimmed_wav = trim_wav(wav_path, trimmed_wav, args.ref_dur)

    if args.x_vector_only:
        ref_text = ""
        print("Mode: x-vector only (no transcript required)")
    else:
        ref_text = trim_transcript(full_transcript, max_chars=150)
        print(f"Reference transcript ({len(ref_text)} chars): {ref_text}")

    # Load model
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        sys.exit(1)

    # Build voice prompt
    print(f"\nBuilding voice prompt from {args.ref_dur}s clip…")
    t0 = time.time()

    try:
        build_kwargs = dict(
            ref_audio=str(trimmed_wav),
            x_vector_only_mode=args.x_vector_only,
        )
        if not args.x_vector_only and ref_text:
            build_kwargs["ref_text"] = ref_text

        prompt_items = model.create_voice_clone_prompt(**build_kwargs)
    except Exception as e:
        print(f"[ERROR] create_voice_clone_prompt failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"[OK] Prompt built in {elapsed:.1f}s")

    # Save
    import torch
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_items, str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[OK] Saved → {output_path} ({size_mb:.1f} MB)")

    meta = {
        "model": args.model,
        "ref_audio": str(trimmed_wav),
        "ref_text": ref_text,
        "ref_duration": args.ref_dur,
        "x_vector_only": args.x_vector_only,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    VOICE_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Metadata → {VOICE_META_PATH}")
    print()
    print("[DONE] Voice prompt ready!")
    print('  python generate.py --text "Hello, this is my voice."')


if __name__ == "__main__":
    main()
