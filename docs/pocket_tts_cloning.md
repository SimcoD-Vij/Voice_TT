# Technical Documentation: Pocket TTS Voice Cloning & Optimization

## Executive Summary
This project successfully implemented a high-quality voice cloning pipeline using the **Pocket TTS** model (FlowLM + Mimi). The work transitioned from resolving critical authentication and structural blockers to implementing advanced audio processing for human-sounding sales pitches.

---

## 1. Research & Analysis
- **Model Architecture**: Analyzed the repository to understand the interaction between `FlowLM` (text-to-codecs) and `Mimi` (codecs-to-audio).
- **Blockers Identified**:
    - Gated repository access requirements for cloning-capable weights.
    - Silent output due to incorrect audio encoding (IEEE Float vs PCM 16-bit).
    - Distortion/Gibberish due to "without-voice-cloning" weights.

---

## 2. Infrastructure & Authentication
- **Hugging Face Gated Access**: Resolved a critical 403 Forbidden error by modifying the library's `download_if_necessary` utility to correctly pass fine-grained tokens to `hf_hub_download`.
- **Environment Stability**: Managed library conflicts (e.g., `torchvision` NMS errors) and ensured compatibility with CPU-based inference using `torchao`.

---

## 3. High-Fidelity Voice Cloning Workflow

### Step 1: Source Audio Preparation
The source audio (`recording_1.wav`) is cleaned to provide a clear speaker embedding.
- **Denoising**: Removed background hiss.
- **Normalization**: Standardized signal levels.
- **Advanced EQ**: Boosted low-mid frequencies (200Hz) for warmth and cut harsh highs (4000Hz).

### Step 2: Voice Embedding Export
- Processed the cleaned audio through the model to extract the "mathematical fingerprint" of the voice.
- Exported this fingerprint to `recording_1.safetensors` for rapid, consistent reuse without re-processing the raw audio.

### Step 3: Human-Centric Text Synthesis
To avoid "robotic" delivery, the text is processed using:
- **Prosody Control**: Strategic use of commas (micro-pauses) and ellipses (thinking pauses).
- **Chunked Generation**: Synthesizing sentence-by-sentence to maintain intonation consistency and inserting specific silence gaps (300ms) between phrases.

### Step 4: Post-Processing for Presence
The raw model output is "sweetened" for a professional feel:
- **Warmth EQ**: Subtle frequency adjustments.
- **Ambience**: Simulation of a natural room environment using `aecho`.
- **Broadcast Standard**: Final normalization to `-16 LUFS` for consistent playback across devices.

---

## 4. Final System State (Docker Optimized)
- **Library Root**: `D:\voice tts\pocket-tts`
- **Internal Model Path**: `D:\voice tts\pocket-tts\pocket_tts\models\model.safetensors`
- **Internal Tokenizer Path**: `D:\voice tts\pocket-tts\pocket_tts\models\tokenizer.model`
- **Configuration**: `pocket_tts\config\english.yaml` updated to prefer local paths.
- **Primary Script**: `D:\voice tts\generate_enhanced_voice.py`
- **Voice Embedding**: `D:\voice tts\recording_1.safetensors`
