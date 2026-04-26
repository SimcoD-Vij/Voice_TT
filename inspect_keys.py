import safetensors.torch
from huggingface_hub import hf_hub_download

repo_id = "kyutai/pocket-tts-without-voice-cloning"
filename = "languages/english/model.safetensors"

try:
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = safetensors.torch.load_file(path)
    print("Keys found in non-gated model:")
    found = False
    for key in state_dict.keys():
        if "speaker_proj_weight" in key:
            print(f"MATCH: {key}")
            found = True
    if not found:
        print("No speaker_proj_weight found in non-gated model.")
except Exception as e:
    print(f"Failed to inspect keys: {e}")
