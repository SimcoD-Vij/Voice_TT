import requests
from pathlib import Path

# Use token in URL
token_path = Path.home() / ".cache" / "huggingface" / "token"
token = token_path.read_text().strip()

url = f"https://onetime@pocket:{token}@huggingface.co/kyutai/pocket-tts/resolve/39592ff23c9ef80098bb74895d104c26275fe2c9/languages/english/model.safetensors"
output = r"D:\voice tts\weights\model_gated.safetensors"
Path(output).parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading from authenticated URL...")
try:
    # Use requests with manual redirect handling because LFS might strip the auth
    response = requests.get(url, allow_redirects=False)
    print(f"Initial status: {response.status_code}")
    if response.status_code in (301, 302, 307, 308):
        redirect_url = response.headers.get("Location")
        print(f"Redirecting to S3. Downloading payload...")
        # LFS download from S3 doesn't need token if we have the signed URL
        response = requests.get(redirect_url, stream=True)
        response.raise_for_status()
        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Success! Gated weights downloaded.")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Failed: {e}")
