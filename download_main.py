import requests
from pathlib import Path

token_path = Path.home() / ".cache" / "huggingface" / "token"
token = token_path.read_text().strip()

# TRY MAIN BRANCH
url = "https://huggingface.co/kyutai/pocket-tts/resolve/main/languages/english/model.safetensors"
output = r"D:\voice tts\weights\model_gated_main.safetensors"
Path(output).parent.mkdir(parents=True, exist_ok=True)

headers = {"Authorization": f"Bearer {token}"}

print(f"Downloading from {url}...")
try:
    # Use standard requests with automatic redirect
    # If it fails with 403, we try WITHOUT the token on the redirected URL
    response = requests.get(url, headers=headers, allow_redirects=True)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        with open(output, "wb") as f:
            f.write(response.content)
        print("Success! Gated weights from main branch downloaded.")
    else:
        print(f"Failed with status {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
