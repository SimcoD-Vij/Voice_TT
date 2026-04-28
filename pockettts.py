#!/usr/bin/env python3
import sys
import logging
import threading
import time
import multiprocessing

# Force fork for stability (same as Docker entrypoint)
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("pockettts-cli")

def start_server(host, port):
    import uvicorn
    # Import here to avoid early loading
    from pocket_tts.main import web_app
    uvicorn.run(
        web_app,
        host=host,
        port=port,
        workers=1,
        loop="asyncio",
        reload=False,
        log_level="info",
    )

def main():
    # Simple argument parsing: python pockettts.py 0.0.0.0 8100
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    # Start uvicorn in background
    server_thread = threading.Thread(target=start_server, args=(host, port), daemon=True)
    server_thread.start()
    log.info(f"Pocket TTS server starting on {host}:{port} in background...")

    # Wait for uvicorn to bind
    time.sleep(2)

    # Load and inject model (Decoupled logic)
    log.info("Loading model...")
    try:
        from pocket_tts import TTSModel
        import pocket_tts.main as ptts_main
        
        # Load model weights
        _model = TTSModel.load_model()
        
        # Pre-warm
        import copy
        _state = _model.get_state_for_audio_prompt("alba")
        _ = _model.generate_audio(copy.deepcopy(_state), "Ready.")
        
        # Inject
        ptts_main.tts_model = _model
        
        # Readiness
        with open("/tmp/pocket_tts_ready", "w") as f:
            f.write("ready")
            
        log.info("Model loaded and injected. Ready to serve TTS.")
    except Exception as e:
        log.error(f"Failed to load/inject model: {e}")
        # We keep the server alive even if model fails so the port remains open
        # but requests will fail until model is fixed.

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")

if __name__ == "__main__":
    main()
