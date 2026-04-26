"""
server.py
=========
FastAPI HTTP API server for real-time voice generation.
The model and voice prompt are loaded ONCE at startup for near-instant generation.

Endpoints:
    GET  /health          → status check
    POST /generate        → JSON body {text, language?} → WAV file bytes
    POST /generate/stream → streaming WAV response
    GET  /voice/info      → info about current loaded voice

Usage:
    python server.py
    python server.py --port 8765 --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
    python server.py --host 0.0.0.0  # expose on network

Integration with Pipecat / Twilio:
    Send POST /generate with {"text": "..."} and pipe the returned WAV
    bytes directly into your audio frame.
"""

import os
import sys
import json
import time
import io
import asyncio
import argparse
from pathlib import Path
from typing import Optional, AsyncIterator, List, Dict, Any
import base64
PROJECT_ROOT = Path(__file__).parent.absolute()
# Ensure local Qwen3-TTS is in path (priority)
sys.path.insert(0, str(PROJECT_ROOT / "Qwen3-TTS"))
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Security, status
from fastapi.security import APIKeyHeader
from fastapi.responses import Response, StreamingResponse, JSONResponse
from pydantic import BaseModel
import torch
import numpy as np
import soundfile as sf
DATASET_DIR = PROJECT_ROOT / "dataset"
VOICE_PROMPT_PATH = DATASET_DIR / "voice_prompt.safetensors"
VOICE_META_PATH = DATASET_DIR / "voice_meta.json"

# Global state — loaded once
_model = None
_voice_prompt = None
_model_id = r"D:\voice tts\models\Qwen3-TTS-12Hz-0.6B-Base"
_sample_rate = 24000
_expected_api_key = None  # Set at startup

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if _expected_api_key:
        if api_key != _expected_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key",
                headers={"WWW-Authenticate": API_KEY_NAME},
            )
    return api_key

# CPU Optimizations
if not torch.cuda.is_available():
    # Optimize for CPU
    threads = os.cpu_count() or 4
    torch.set_num_threads(threads)
    print(f"[CPU] Optimization: Setting torch threads to {threads}")


def load_model(model_id: str):
    import torch
    from qwen_tts import Qwen3TTSModel
    local_path = PROJECT_ROOT / "models" / Path(model_id).name
    model_source = str(local_path) if local_path.exists() else model_id
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    try:
        m = Qwen3TTSModel.from_pretrained(
            model_source, device_map=device, torch_dtype=dtype,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        )
        print(f"[OK] Model loaded on {device} (FlashAttention 2: {torch.cuda.is_available()})")
    except Exception as e:
        print(f"[INFO] Falling back to eager attention: {e}")
        m = Qwen3TTSModel.from_pretrained(
            model_source, device_map=device, torch_dtype=dtype,
            attn_implementation="eager",
        )
        print(f"[OK] Model loaded on {device} (eager attention)")
    return m


def load_voice_prompt(model):
    prompt_pt = DATASET_DIR / "voice_prompt.pt"
    if not prompt_pt.exists():
        raise FileNotFoundError(
            f"voice_prompt.pt not found at {prompt_pt}\n"
            "Run: python build_voice_prompt.py"
        )
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompt = torch.load(str(prompt_pt), map_location=device, weights_only=False)
    print(f"[OK] Voice prompt loaded from {prompt_pt.name} to {device}")
    return prompt


def audio_to_wav_bytes(audio_numpy, sr: int) -> bytes:
    """Convert numpy audio array to WAV bytes in memory."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_numpy, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


class Qwen3TTSWebSocketStreamer:
    """Custom streamer for Qwen3-TTS that sends audio chunks over WebSocket."""

    def __init__(self, model, websocket: WebSocket, context_id: str, chunk_size: int = 12):
        self.model = model
        self.websocket = websocket
        self.context_id = context_id
        self.chunk_size = chunk_size
        self.token_buffer = []
        self.loop = asyncio.get_event_loop()
        self.is_finished = False

    def put(self, value):
        """Called by transformers.generate at each step."""
        # Value is the token id for the first codebook.
        # But we need the full codec_ids from the talker's hidden states!
        # Since modeling_qwen3_tts.py puts codec_ids into hidden_states,
        # we can't easily get it here from 'value'.
        # However, we can use a TRICK: the 'value' passed to streamers
        # is the result of the last generation step.
        pass

    def end(self):
        self.is_finished = True


def clean_text(text: str) -> str:
    """Strip Llama 3.2 / ChatML headers and other leaked tokens."""
    if not text:
        return ""
    
    # Common leaked tokens
    to_strip = [
        "<|start_header_id|>", "<|end_header_id|>", 
        "<|reserved_special_token", "<|begin_of_text|>",
        "assistant", "user", "system"
    ]
    
    cleaned = text
    for token in to_strip:
        cleaned = cleaned.replace(token, "")
    
    # Strip any leading/trailing whitespace or punctuation that might be left
    return cleaned.strip()


async def stream_audio_from_codes(model, codes_queue: asyncio.Queue, websocket: WebSocket, context_id: str, chunk_size: int = 12):
    """Worker that takes codec tokens from a queue, decodes them, and sends audio."""
    accumulated_codes = []
    try:
        while True:
            code = await codes_queue.get()
            if code is None:  # Sentinel for end of streaming
                break
            
            accumulated_codes.append(code)
            
            if len(accumulated_codes) >= chunk_size:
                # Decode accumulated codes
                codes_tensor = torch.stack(accumulated_codes, dim=0) # (chunk_size, num_code_groups)
                # Decode returns (wavs, sr)
                # We need to wrap it in a format the decoder expects: list of dicts or ModelOutput
                with torch.no_grad():
                    wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": codes_tensor}])
                
                audio_data = wavs[0].astype(np.float32)
                # Convert to base64
                import base64
                import soundfile as sf
                
                # We send raw PCM_16 bytes for lower overhead than full WAV for chunks
                # but many clients expect base64 of PCM samples.
                # Pipecat Qwen3TTSService expects base64.
                
                # Convert to int16 PCM
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_base64 = base64.b64encode(audio_int16.tobytes()).decode()
                
                await websocket.send_json({
                    "type": "audio",
                    "context_id": context_id,
                    "audio": audio_base64,
                    "sample_rate": sr
                })
                
                accumulated_codes = []
                codes_queue.task_done()

        # Handle remaining codes
        if accumulated_codes:
            codes_tensor = torch.stack(accumulated_codes, dim=0)
            with torch.no_grad():
                wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": codes_tensor}])
            audio_int16 = (wavs[0] * 32767).astype(np.int16)
            audio_base64 = base64.b64encode(audio_int16.tobytes()).decode()
            await websocket.send_json({
                "type": "audio",
                "context_id": context_id,
                "audio": audio_base64,
                "sample_rate": sr
            })

        await websocket.send_json({
            "type": "final",
            "context_id": context_id
        })

    except Exception as e:
        print(f"[STREAM ERROR] {e}")
        import traceback; traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except: pass


# Global state for token interception
_current_codes_queue = None

def patch_talker_for_streaming(talker):
    """Patch the talker's forward method to intercept codec tokens."""
    # Patch the class to ensure PyTorch __call__ picks it up
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
    original_forward = Qwen3TTSTalkerForConditionalGeneration.forward
    
    from functools import wraps
    
    @wraps(original_forward)
    def streaming_forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = original_forward(self, *args, **kwargs)

        try:
            if _current_codes_queue is not None and hasattr(outputs, "hidden_states"):
                h = outputs.hidden_states
                if isinstance(h, (list, tuple)) and len(h) > 1:
                    codec_ids = h[1]
                    if codec_ids is not None:
                        # codec_ids is (B, 16). We take all 16 codes for the first batch item.
                        token = codec_ids[0].detach().cpu()
                        _current_codes_queue.put_nowait(token)
                        # Remove print after verifying it works, but adding it now to debug
                        # print(f"[DEBUG] Sent token: {token.shape}")
                else:
                    print(f"[DEBUG] hidden_states present but not tuple/list > 1: {type(h)}")
            elif _current_codes_queue is not None:
                print(f"[DEBUG] outputs missing hidden_states! {type(outputs)}")
        except Exception as e:
            print(f"[STREAM DEBUG ERROR]: {e}")
            import traceback; traceback.print_exc()

        return outputs
    
    Qwen3TTSTalkerForConditionalGeneration.forward = streaming_forward
    return original_forward


def create_app(model_id: str, api_key: Optional[str] = None):
    from fastapi import FastAPI, HTTPException, Depends, Security
    from fastapi.responses import Response, StreamingResponse, JSONResponse
    from pydantic import BaseModel

    app = FastAPI(
        title="Voice Clone TTS API",
        description="Real-time voice generation using Qwen3-TTS with your cloned voice",
        version="1.0.0",
    )

    class GenerateRequest(BaseModel):
        text: str
        language: Optional[str] = "auto"
        output_format: Optional[str] = "wav"  # wav only for now

    @app.on_event("startup")
    async def startup():
        global _model, _voice_prompt, _model_id, _sample_rate, _expected_api_key
        _model_id = model_id
        _expected_api_key = api_key
        print(f"\n{'='*60}")
        print(" QWEN3-TTS SERVER STARTING")
        print(f"{'='*60}")
        print(f"Model: {model_id}")

        try:
            _model = load_model(model_id)
            _voice_prompt = load_voice_prompt(_model)
            # Get sample rate from a tiny generation
            print("Warming up model (first call takes longer)…")
            t0 = time.time()
            wavs, sr = _model.generate_voice_clone(
                text="Initializing.",
                language="English",
                voice_clone_prompt=_voice_prompt,
            )
            _sample_rate = sr
            print(f"[OK] Warmup done in {time.time()-t0:.1f}s | Sample rate: {sr}Hz")
            print(f"\n[READY] Server running — send requests to /generate")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"[ERROR] Failed to initialize: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    @app.get("/health", dependencies=[Depends(verify_api_key)])
    async def health():
        return {
            "status": "ok",
            "model": _model_id,
            "voice_prompt": str(VOICE_PROMPT_PATH.name) if VOICE_PROMPT_PATH.exists() else None,
            "sample_rate": _sample_rate,
        }

    @app.get("/voice/info", dependencies=[Depends(verify_api_key)])
    async def voice_info():
        if VOICE_META_PATH.exists():
            meta = json.loads(VOICE_META_PATH.read_text(encoding="utf-8"))
            return meta
        return {"info": "No voice metadata found. Run build_voice_prompt.py"}

    @app.post("/generate", dependencies=[Depends(verify_api_key)])
    async def generate(req: GenerateRequest):
        """
        Generate audio WAV from text using your cloned voice.
        Returns WAV file as bytes.
        """
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="text cannot be empty")
        if _model is None or _voice_prompt is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        t0 = time.time()
        try:
            # Run in thread executor so it doesn't block the event loop
            loop = asyncio.get_event_loop()
            wavs, sr = await loop.run_in_executor(
                None,
                lambda: _model.generate_voice_clone(
                    text=req.text,
                    language=req.language,
                    voice_clone_prompt=_voice_prompt,
                    max_new_tokens=2048,   # prevents infinite generation hangs
                )
            )
            audio_duration = len(wavs[0]) / sr
            elapsed = time.time() - t0
            rtf = elapsed / audio_duration
            print(f"[GEN] {audio_duration:.1f}s audio | {elapsed:.2f}s gen | RTF={rtf:.2f} | text: {req.text[:60]}")

            wav_bytes = audio_to_wav_bytes(wavs[0], sr)
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "X-Audio-Duration": f"{audio_duration:.3f}",
                    "X-Generation-Time": f"{elapsed:.3f}",
                    "X-RTF": f"{rtf:.3f}",
                    "Content-Disposition": 'attachment; filename="output.wav"',
                }
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate/batch", dependencies=[Depends(verify_api_key)])
    async def generate_batch(texts: list[str], language: str = "auto"):
        """Generate audio for multiple texts. Returns JSON with base64 WAV data."""
        import base64
        if not texts:
            raise HTTPException(status_code=400, detail="texts list cannot be empty")
        if len(texts) > 10:
            raise HTTPException(status_code=400, detail="max 10 texts per batch")

        loop = asyncio.get_event_loop()
        wavs, sr = await loop.run_in_executor(
            None,
            lambda: _model.generate_voice_clone(
                text=texts,
                language=[language] * len(texts),
                voice_clone_prompt=_voice_prompt,
            )
        )
        results = []
        for wav in wavs:
            wav_bytes = audio_to_wav_bytes(wav, sr)
            results.append({
                "audio_base64": base64.b64encode(wav_bytes).decode(),
                "sample_rate": sr,
                "duration": len(wav) / sr,
            })
        return {"results": results, "count": len(results)}

    @app.websocket("/voice/stream")
    async def websocket_endpoint(websocket: WebSocket, api_key: Optional[str] = None):
        if _expected_api_key:
            # For WebSockets, we often check a query parameter 'api_key'
            # fallback to header if available in handshakes (though less reliable)
            query_key = websocket.query_params.get("api_key")
            actual_key = query_key or api_key
            if actual_key != _expected_api_key:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

        await websocket.accept()
        print(f"[WS] Connection accepted")
        
        context_id = None
        current_task = None
        codes_queue = asyncio.Queue()
        
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                msg_type = msg.get("type")
                
                if msg_type == "config":
                    # Initial config, currently ignored as model is pre-loaded
                    await websocket.send_json({"type": "ready"})
                
                elif msg_type == "create_context":
                    context_id = msg.get("context_id")
                    await websocket.send_json({"type": "context_created", "context_id": context_id})

                elif msg_type == "synthesize":
                    text = clean_text(msg.get("text"))
                    context_id = msg.get("context_id", context_id)
                    
                    if not text:
                        continue
                        
                    print(f"[WS] Synthesize: {text[:40]}...")
                    
                    # Start synthesis as a background task so we don't block the receive loop
                    async def perform_synthesis(txt, ctx_id, lang):
                        codes_q = asyncio.Queue()
                        streamer_task = asyncio.create_task(
                            stream_audio_from_codes(_model, codes_q, websocket, ctx_id, chunk_size=12)
                        )
                        
                        main_loop = asyncio.get_event_loop()
                        def run_gen(loop):
                            global _current_codes_queue
                            _current_codes_queue = codes_q
                            try:
                                original_forward = patch_talker_for_streaming(_model.model.talker)
                                try:
                                    _model.generate_voice_clone(
                                        text=txt,
                                        language=lang,
                                        voice_clone_prompt=_voice_prompt,
                                        max_new_tokens=2048,
                                    )
                                finally:
                                    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
                                    Qwen3TTSTalkerForConditionalGeneration.forward = original_forward
                                    if "forward" in _model.model.talker.__dict__:
                                        del _model.model.talker.forward
                                    loop.call_soon_threadsafe(codes_q.put_nowait, None)
                                    _current_codes_queue = None
                            except Exception as e:
                                print(f"[GEN ERROR] {e}")
                                loop.call_soon_threadsafe(codes_q.put_nowait, None)

                        try:
                            await asyncio.get_event_loop().run_in_executor(None, run_gen, main_loop)
                            await streamer_task
                        except Exception as e:
                            print(f"[SYNTHESIS TASK ERROR] {e}")

                    asyncio.create_task(perform_synthesis(text, context_id, msg.get("language", "English")))

                elif msg_type == "close_context":
                    # Stop any current generation
                    pass

        except WebSocketDisconnect:
            print(f"[WS] Disconnected")
        except Exception as e:
            print(f"[WS ERROR] {e}")

    return app


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Clone API Server")
    parser.add_argument("--host", default="localhost", help="Host to bind (use 0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=8765, help="Port number")
    parser.add_argument("--model", default=r"D:\voice tts\models\Qwen3-TTS-12Hz-0.6B-Base", help="Model ID or local path")
    parser.add_argument("--api-key", default=None, help="Optional API Key for authentication")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload for development")
    args = parser.parse_args()

    import uvicorn
    app = create_app(args.model, args.api_key)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,  # Can't use reload with global model state
        log_level="info",
    )


if __name__ == "__main__":
    main()
