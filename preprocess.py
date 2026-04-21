"""
preprocess.py
=============
Converts M4A recordings in dataset/ to 16kHz mono WAV files and
auto-transcribes them using OpenAI Whisper.

Usage:
    python preprocess.py
    python preprocess.py --dataset_dir D:/voice tts/dataset
    python preprocess.py --whisper_model base  # tiny|base|small|medium
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def _add_conda_to_path():
    """Add conda env Library/bin and Scripts to PATH so ffmpeg is found."""
    import sys
    conda_env = Path(sys.executable).parent
    extra = [
        str(conda_env / "Library" / "bin"),
        str(conda_env / "Scripts"),
        str(conda_env),
    ]
    for p in extra:
        if p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


def check_ffmpeg():
    """Verify ffmpeg is available on PATH."""
    _add_conda_to_path()
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("[OK] ffmpeg found.")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print("[ERROR] ffmpeg not found. Install it:")
    print("  conda install -c conda-forge ffmpeg -y")
    print("  OR download from https://ffmpeg.org/download.html")
    return False


def convert_m4a_to_wav(m4a_path: Path, wav_path: Path) -> bool:
    """Convert a single M4A file to 16kHz mono WAV using ffmpeg."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(m4a_path),
            "-ar", "16000",       # 16kHz sample rate (ideal for Whisper)
            "-ac", "1",           # mono
            "-sample_fmt", "s16", # 16-bit PCM
            str(wav_path)
        ],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  [OK] Converted: {m4a_path.name} → {wav_path.name}")
        return True
    else:
        print(f"  [ERROR] Failed to convert {m4a_path.name}:")
        print(result.stderr[-500:])
        return False


def transcribe_wav(wav_path: Path, txt_path: Path, model_name: str = "base") -> str:
    """Transcribe a WAV file using OpenAI Whisper. Returns transcript text."""
    import whisper
    print(f"  Loading Whisper '{model_name}' model…")
    model = whisper.load_model(model_name)
    print(f"  Transcribing {wav_path.name}…")
    result = model.transcribe(str(wav_path), language=None, verbose=False)
    text = result["text"].strip()
    txt_path.write_text(text, encoding="utf-8")
    detected = result.get("language", "unknown")
    print(f"  [OK] Transcript saved ({len(text)} chars, language={detected})")
    print(f"       Preview: {text[:120]}…" if len(text) > 120 else f"       Text: {text}")
    return text


def get_audio_duration(wav_path: Path) -> float:
    """Return duration in seconds of a WAV file."""
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return info.duration
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Preprocess M4A dataset for Qwen3-TTS voice cloning")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(Path(__file__).parent / "dataset"),
        help="Path to folder containing M4A files"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (base recommended for speed+accuracy balance)"
    )
    parser.add_argument(
        "--skip_transcription",
        action="store_true",
        help="Skip Whisper transcription (only convert audio)"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" QWEN3-TTS DATASET PREPROCESSOR")
    print("=" * 60)
    print(f"Dataset dir : {dataset_dir}")
    print(f"Output dir  : {processed_dir}")
    print()

    # Check ffmpeg
    if not check_ffmpeg():
        sys.exit(1)

    # Find all M4A files
    m4a_files = sorted(dataset_dir.glob("*.m4a"))
    if not m4a_files:
        print("[WARN] No .m4a files found in", dataset_dir)
        print("       Also checking for .wav files already present…")
        wav_existing = sorted(dataset_dir.glob("*.wav"))
        if wav_existing:
            print(f"       Found {len(wav_existing)} WAV file(s), copying to processed/")
            import shutil
            for f in wav_existing:
                dest = processed_dir / f.name
                shutil.copy2(f, dest)
            m4a_files = []
        else:
            print("[ERROR] No audio files found. Add .m4a or .wav files to:", dataset_dir)
            sys.exit(1)

    print(f"Found {len(m4a_files)} M4A file(s):\n")

    converted_wavs = []
    for i, m4a_path in enumerate(m4a_files, 1):
        # Sanitize filename: remove spaces, special chars
        safe_name = f"recording_{i}.wav"
        wav_path = processed_dir / safe_name
        txt_path = processed_dir / f"recording_{i}.txt"

        print(f"[{i}/{len(m4a_files)}] Processing: {m4a_path.name}")

        # Convert
        if wav_path.exists():
            print(f"  [SKIP] WAV already exists: {wav_path.name}")
        else:
            if not convert_m4a_to_wav(m4a_path, wav_path):
                continue

        # Duration info
        duration = get_audio_duration(wav_path)
        print(f"  Duration: {duration:.1f} seconds")

        converted_wavs.append((wav_path, txt_path, duration))

        # Transcribe
        if not args.skip_transcription:
            if txt_path.exists():
                existing_text = txt_path.read_text(encoding="utf-8").strip()
                print(f"  [SKIP] Transcript already exists ({len(existing_text)} chars)")
            else:
                try:
                    transcribe_wav(wav_path, txt_path, args.whisper_model)
                except ImportError:
                    print("  [ERROR] whisper not installed. Run: pip install openai-whisper")
                except Exception as e:
                    print(f"  [ERROR] Transcription failed: {e}")
        print()

    # Summary
    print("=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    total_duration = sum(d for _, _, d in converted_wavs)
    print(f"  Processed files : {len(converted_wavs)}")
    print(f"  Total duration  : {total_duration:.1f} sec ({total_duration/60:.1f} min)")
    print()

    if total_duration < 30:
        print("[WARN] Total audio < 30 seconds.")
        print("       Zero-shot cloning will still work, but for fine-tuning,")
        print("       aim for 30+ minutes of clean recordings with transcripts.")
    elif total_duration < 1800:
        print("[INFO] Zero-shot cloning: Ready to use!")
        print(f"[INFO] Fine-tuning: Need {(1800 - total_duration)/60:.0f} more minutes of audio for best results.")
    else:
        print("[OK] Enough data for fine-tuning!")

    print()
    print("Next step: Run build_voice_prompt.py to create your voice embedding.")
    print(f"  python build_voice_prompt.py")


if __name__ == "__main__":
    main()
