from huggingface_hub import hf_hub_download, list_repo_files
import os
from pathlib import Path

repo_id = "kyutai/pocket-tts"
filename = "languages/english/model.safetensors"
revision = "39592ff23c9ef80098bb74895d104c26275fe2c9"

# Token from cache
token_path = Path.home() / ".cache" / "huggingface" / "token"
token = None
if token_path.exists():
    token = token_path.read_text().strip()

print(f"Token found: {token[:5]}...{token[-5:]}" if token else "Token NOT found.")

try:
    print(f"Listing files in {repo_id} with token...")
    files = list_repo_files(repo_id=repo_id, token=token)
    print("Success! Access to gated repo confirmed.")
    
    print(f"Attempting download of {filename}...")
    # Explicitly pass token
    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token)
    print(f"Success! Model downloaded to: {path}")
    
except Exception as e:
    print(f"Error: {e}")
    if "403" in str(e):
        print("ALERT: 403 Forbidden. This means you might need to accept terms at https://huggingface.co/kyutai/pocket-tts")
    elif "Connection" in str(e):
        print("ALERT: Connection error. This machine has trouble reaching HF storage buckets.")
