# Qwen3-TTS Voice Cloning System

This repository provides a comprehensive pipeline for high-fidelity voice cloning using the Qwen3-TTS 0.6B model. It supports both high-performance GPU inference and accessible CPU-only generation.

## 🚀 Prerequisites

- **Environment**: [Conda](https://docs.anaconda.com/free/anaconda/install/index.html) (highly recommended for dependency management).
- **Hardware**: NVIDIA GPU with at least 6GB VRAM (required for local inference).
- **Python**: 3.10+

## 🛠️ Installation

1.  **Create and Activate Conda Environment**:
    ```powershell
    conda create -n voice_cloning python=3.10
    conda activate voice_cloning
    ```

2.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```
    *Note: If `soundfile` or `sox` errors occur, ensure you have the necessary system libraries installed.*

3.  **Download the Model**:
    To avoid download hangs during the first run, download the model manually using the `huggingface-cli`:
    ```powershell
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/Qwen3-TTS-12Hz-0.6B-Base
    ```

## 🎙️ Usage Workflow

### 1. Data Preparation
Place your reference audio (e.g., `recording_1.wav`) in `dataset/processed/`. If you have a transcript, save it as `recording_1.txt` in the same directory.

### 2. Build Voice Prompt
Build a optimized KV-cache prompt from your reference audio.
```powershell
# Using recording_1 as reference
python build_voice_prompt.py --ref recording_1
```
- **Reference Dataset**: `recording_1.wav` (12-second trimmed segment).
- **Process**: Extracts X-vector and KV-cache tokens, saving them to `dataset/voice_prompt.pt` for instant loading.

### 3. Generate Audio (GPU vs CPU)

#### High Performance (NVIDIA GPU)
```powershell
python generate.py --text "Protect your smartphone..." --output outputs/xoptimus_rec1_en_qwen.wav
```

#### Universal Compatibility (CPU Only)
```powershell
python generate.py --text "Protect your smartphone..." --output outputs/xoptimus_rec1_en_qwen_cpu.wav --cpu
```

## 🐳 Docker Usage

For a consistent and portable environment, you can use Docker.

1.  **Build the Image**:
    ```powershell
    docker compose build
    ```

2.  **Building Voice Prompt inside Docker**:
    ```powershell
    # GPU Version
    docker compose run --rm tts python build_voice_prompt.py --ref recording_1
    ```

3.  **Generating Audio inside Docker**:
    ```powershell
    # GPU Version
    docker compose run --rm tts python generate.py --text "Hello world" --output outputs/docker_output.wav
    
    # CPU Version
    docker compose run --rm tts-cpu python generate.py --text "Hello world" --output outputs/docker_output_cpu.wav --cpu
    ```

> [!IMPORTANT]
> To use GPU in Docker, you must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host system.

## 📊 Performance Benchmarks (0.6B Model)

| Hardware | RTF (Real-Time Factor) | Generation Time (30s Audio) |
| :--- | :--- | :--- |
| **NVIDIA GPU (6GB)** | ~6.7x | ~232s |
| **Intel/AMD CPU** | ~11.3x | ~304s |

*Note: RTF is calculated as Elapsed Time / Audio Duration. Lower RTF in some contexts means faster, but here it is reported as the ratio of processing time.*

## 🧬 Replication Summary

To replicate the exact XOptimus voice clone:
1. Ensure `dataset/processed/recording_1.wav` is present.
2. Run the prompt builder: `python build_voice_prompt.py --ref recording_1`.
3. Use `generate.py` with the sales pitch text.
4. Results will be saved in the `outputs/` folder.

## 📁 Key Files

- `preprocess.py`: Converts M4A to WAV and prepares the dataset.
- `build_voice_prompt.py`: Extracts voice features into a reusable prompt.
- `generate.py`: Main CLI for audio generation.
- `Qwen3-TTS/`: The core model architecture (patched for v4.48 compatibility).

## 💡 Tips
- **Clip Duration**: 10-15 seconds is the "sweet spot" for voice quality vs. generation speed.
- **English-Only**: For best results in English, ensure the reference clip is also English.
- **GPU Memory**: If you encounter CUDA Out of Memory, try using `attn_implementation="eager"` in `generate.py`.

## 🎬 Quality & Reproducibility

To guarantee the same high-quality voice generated in previous tests:

1.  **Strict Reference Trimming**: Always use the first **12 seconds** of `recording_1.wav`. The `build_voice_prompt.py` script does this automatically.
2.  **Sampling Parameters**: The `generate.py` script uses the model's default sampling settings (Temperature ~1.0, Top-P ~1.0).
3.  **Model Precision**:
    - **GPU**: Uses `bfloat16` for fast, high-quality inference.
    - **CPU**: Uses `float32` for better numerical stability on processors.
    *Sound quality remains nearly identical across both.*
4.  **Audio Rate**: The output is generated at **24kHz**, providing a professional balance between file size and clarity.

By following the exact steps in [Replication Summary](#-replication-summary), you will achieve the identical voice characteristics seen in the XOptimus prototypes.
