"""
setup.py
========
One-click setup script. Run this FIRST before anything else.

Steps:
  1. Checks conda environment
  2. Installs ffmpeg (via conda)
  3. Gets qwen-tts from the cloned repo (installs in editable mode)
  4. Installs all Python dependencies
  5. Downloads Qwen3-TTS-12Hz-0.6B-Base model weights
  6. Installs FlashAttention 2 (optional, for VRAM reduction)

Usage:
    python setup.py
    python setup.py --model 1.7b   (for higher quality, needs ~6GB VRAM)
    python setup.py --skip_flash   (skip FlashAttention build)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
QWEN_REPO_DIR = PROJECT_ROOT / "Qwen3-TTS"
CONDA_ENV_PATH = r"E:\NvidiaGPU\conda310\envs\ml_gpu"


def run(cmd: list, check=True, shell=False, cwd=None) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd, check=check, shell=shell,
        cwd=cwd or str(PROJECT_ROOT),
        capture_output=False, text=True
    )
    return result


def pip(packages: list, extra_args: list = None):
    """Install packages into the ml_gpu conda env."""
    pip_path = Path(CONDA_ENV_PATH) / "Scripts" / "pip.exe"
    if not pip_path.exists():
        pip_path = Path(CONDA_ENV_PATH) / "bin" / "pip"
    cmd = [str(pip_path), "install", "--upgrade"] + packages
    if extra_args:
        cmd += extra_args
    return run(cmd)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="One-click setup for Voice Clone TTS system")
    parser.add_argument(
        "--model", choices=["0.6b", "1.7b"], default="0.6b",
        help="Model size: 0.6b (faster, 4GB VRAM) or 1.7b (better, 6GB VRAM)"
    )
    parser.add_argument(
        "--skip_flash", action="store_true",
        help="Skip FlashAttention 2 installation (add this if it times out)"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip model download (if you already have it)"
    )
    args = parser.parse_args()

    model_map = {
        "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    }
    model_id = model_map[args.model]
    model_name = Path(model_id).name

    print("=" * 60)
    print("  VOICE CLONE TTS — ONE-CLICK SETUP")
    print("=" * 60)
    print(f"  Model          : {model_id}")
    print(f"  Conda env      : {CONDA_ENV_PATH}")
    print(f"  Project root   : {PROJECT_ROOT}")
    print(f"  Models dir     : {MODELS_DIR}")

    # -----------------------------------------------------------------
    section("STEP 1 — Verify conda environment")
    # -----------------------------------------------------------------
    python_path = Path(CONDA_ENV_PATH) / "python.exe"
    if not python_path.exists():
        python_path = Path(CONDA_ENV_PATH) / "bin" / "python"
    if not python_path.exists():
        print(f"[ERROR] Python not found at {CONDA_ENV_PATH}")
        print("Are you sure the ml_gpu conda environment exists?")
        sys.exit(1)
    print(f"[OK] Python found: {python_path}")

    result = subprocess.run(
        [str(python_path), "-c",
         "import torch; print('PyTorch:', torch.__version__); "
         "import torch; print('CUDA:', torch.cuda.is_available()); "
         "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("[WARN] Could not import PyTorch:", result.stderr[:200])

    # -----------------------------------------------------------------
    section("STEP 2 — Install ffmpeg (for M4A conversion)")
    # -----------------------------------------------------------------
    ffmpeg_check = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if ffmpeg_check.returncode == 0:
        print("[OK] ffmpeg already available on PATH")
    else:
        print("Installing ffmpeg via conda…")
        conda_path = Path(CONDA_ENV_PATH).parent.parent / "condabin" / "conda.bat"
        if not conda_path.exists():
            conda_path = "conda"  # try PATH
        run([str(conda_path), "install", "-p", CONDA_ENV_PATH,
             "-c", "conda-forge", "ffmpeg", "-y"])

    # -----------------------------------------------------------------
    section("STEP 3 — Clone Qwen3-TTS repo")
    # -----------------------------------------------------------------
    if QWEN_REPO_DIR.exists():
        print(f"[OK] Repo already exists at {QWEN_REPO_DIR}")
        print("     Pulling latest changes…")
        run(["git", "pull"], cwd=str(QWEN_REPO_DIR))
    else:
        print("Cloning QwenLM/Qwen3-TTS…")
        run(["git", "clone", "https://github.com/QwenLM/Qwen3-TTS.git",
             str(QWEN_REPO_DIR)])

    # -----------------------------------------------------------------
    section("STEP 4 — Install Python dependencies")
    # -----------------------------------------------------------------
    # Install qwen-tts from cloned repo
    print("Installing qwen-tts from cloned repo…")
    pip(["-e", str(QWEN_REPO_DIR)])

    # Install project requirements
    req_path = PROJECT_ROOT / "requirements.txt"
    print("Installing project requirements…")
    pip(["-r", str(req_path)])

    # -----------------------------------------------------------------
    section("STEP 5 — Install FlashAttention 2 (optional, ~15 min build)")
    # -----------------------------------------------------------------
    if args.skip_flash:
        print("[SKIP] FlashAttention 2 installation skipped (--skip_flash flag)")
        print("       The system will use eager attention (slightly more VRAM)")
    else:
        print("Installing FlashAttention 2 (this can take 15-30 minutes to compile)…")
        print("Press Ctrl+C to skip if it takes too long — the system works without it.")
        try:
            pip(["flash-attn"], extra_args=["--no-build-isolation"])
            print("[OK] FlashAttention 2 installed")
        except KeyboardInterrupt:
            print("\n[SKIP] FlashAttention 2 build cancelled — will use eager attention")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] FlashAttention 2 build failed — will use eager attention")
            print(f"       Error: {e}")
            print("       This is usually fine. Retry with: pip install flash-attn --no-build-isolation")

    # -----------------------------------------------------------------
    section("STEP 6 — Download model weights")
    # -----------------------------------------------------------------
    if args.skip_download:
        print("[SKIP] Model download skipped (--skip_download flag)")
    else:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        local_model_path = MODELS_DIR / model_name

        if local_model_path.exists() and any(local_model_path.iterdir()):
            print(f"[OK] Model already downloaded at {local_model_path}")
        else:
            print(f"Downloading {model_id}…")
            print("(This is ~2-3GB for 0.6B, ~6GB for 1.7B — may take a while)")
            # Also download tokenizer
            pip(["huggingface_hub[cli]"])
            hf_cli = Path(CONDA_ENV_PATH) / "Scripts" / "huggingface-cli.exe"
            if not hf_cli.exists():
                hf_cli = "huggingface-cli"
            run([str(hf_cli), "download", model_id,
                 "--local-dir", str(local_model_path)])
            # Also download tokenizer
            tokenizer_path = MODELS_DIR / "Qwen3-TTS-Tokenizer-12Hz"
            if not tokenizer_path.exists():
                run([str(hf_cli), "download",
                     "Qwen/Qwen3-TTS-Tokenizer-12Hz",
                     "--local-dir", str(tokenizer_path)])

    # -----------------------------------------------------------------
    section("SETUP COMPLETE!")
    # -----------------------------------------------------------------
    print()
    print("All done! Now run these in order:")
    print()
    print("  Step 1: Convert your M4A recordings and transcribe them")
    print("    > python preprocess.py")
    print()
    print("  Step 2: Build your voice prompt (one-time, ~30 sec)")
    print("    > python build_voice_prompt.py")
    print()
    print("  Step 3: Generate your first cloned audio!")
    print('    > python generate.py --text "Hello, this is my voice." --play')
    print()
    print("  Step 4 (optional): Start the API server")
    print("    > python server.py")
    print("    Then POST to http://localhost:8765/generate")
    print()


if __name__ == "__main__":
    main()
