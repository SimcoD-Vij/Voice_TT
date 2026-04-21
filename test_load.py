import torch
try:
    from qwen_tts import Qwen3TTSModel
    print("Import successful")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base", 
        device_map="cuda:0", 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error: {e}")
