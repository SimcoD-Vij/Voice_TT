import requests
from pathlib import Path
from huggingface_hub import hf_hub_url

repo_id = "kyutai/pocket-tts"
filename = "languages/english/model.safetensors"
revision = "39592ff23c9ef80098bb74895d104c26275fe2c9"

# Token from cache
token_path = Path.home() / ".cache" / "huggingface" / "token"
token = token_path.read_text().strip()

# Get the ACTUAL download URL (including redirects)
url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
print(f"Target URL: {url}")

headers = {"Authorization": f"Bearer {token}"}

try:
    print("Fetching redirect URL...")
    # Follow redirect manually to handle token stripping
    response = requests.get(url, headers=headers, allow_redirects=False)
    print(f"Status: {response.status_code}")
    if response.status_code in (301, 302, 307, 308):
        s3_url = response.headers.get("Location")
        print(f"Downloading from S3: {s3_url[:100]}...")
        # DOWNLOAD WITHOUT TOKEN
        response = requests.get(s3_url, stream=True)
        response.raise_for_status()
        
        target_path = Path(r"D:\voice tts\weights\model_gated.safetensors")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Success! Model saved to {target_path}")
    else:
        print(f"Error: Expected redirect, got {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Failed: {e}")
