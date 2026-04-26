import safetensors.torch
from pocket_tts.utils.utils import get_predefined_voice, download_if_necessary

voice_path = get_predefined_voice("english", "alba")
local_path = download_if_necessary(voice_path)
state_dict = safetensors.torch.load_file(local_path)
print(f"Keys in predefined voice: {state_dict.keys()}")
