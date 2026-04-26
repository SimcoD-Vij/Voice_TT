import urllib.request
from pathlib import Path

# Token from cache
token_path = Path.home() / ".cache" / "huggingface" / "token"
token = token_path.read_text().strip()

# Use 'main' branch
url = "https://huggingface.co/kyutai/pocket-tts/resolve/main/languages/english/model.safetensors"
output = r"D:\voice tts\weights\model_gated.safetensors"
Path(output).parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading {url} using urllib...")

req = urllib.request.Request(url)
req.add_header("Authorization", f"Bearer {token}")

try:
    with urllib.request.urlopen(req) as response:
        print(f"Status: {response.getcode()}")
        with open(output, "wb") as f:
            f.write(response.read())
    print("Success! Gated weights downloaded.")
except Exception as e:
    print(f"Failed: {e}")
