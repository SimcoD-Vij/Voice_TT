import requests
import os
from pathlib import Path

def download_file(url, output_path, token):
    print(f"Downloading {url} to {output_path}...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get redirect URL (S3)
    response = requests.get(url, headers=headers, allow_redirects=False)
    if response.status_code in (301, 302, 307, 308):
        redirect_url = response.headers.get("Location")
        # Download from redirect URL without Authorization
        print(f"Redirected to LFS storage. Downloading...")
        response = requests.get(redirect_url, stream=True)
    else:
        print(f"Direct download (status {response.status_code}).")
        response = requests.get(url, headers=headers, stream=True)
    
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

# Token from cache
token_path = Path.home() / ".cache" / "huggingface" / "token"
if not token_path.exists():
    print("Token not found. Please log in first.")
    exit(1)
token = token_path.read_text().strip()

# Target files
WEIGHTS_DIR = Path(r"D:\voice tts\weights\english")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Main model weights
model_url = "https://huggingface.co/kyutai/pocket-tts/resolve/39592ff23c9ef80098bb74895d104c26275fe2c9/languages/english/model.safetensors"
download_file(model_url, WEIGHTS_DIR / "model.safetensors", token)

# 2. Tokenizer (public)
tokenizer_url = "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/d29db7978e464fb90cb3359ee0c69a273b9142cc/languages/english/tokenizer.model"
download_file(tokenizer_url, WEIGHTS_DIR / "tokenizer.model", token)

print(f"\nAll files downloaded to {WEIGHTS_DIR}")
