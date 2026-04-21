import sys
import traceback

print("Checking auto_docstring import...")
try:
    from transformers.utils import auto_docstring
    print("Success: auto_docstring is available.")
except ImportError:
    print("Failed: auto_docstring is NOT available in transformers.utils")
    # traceback.print_exc()

print("\nChecking qwen_tts import...")
try:
    import sys
    import os
    sys.path.append(os.path.join(os.getcwd(), "Qwen3-TTS"))
    import qwen_tts
    print("Success: qwen_tts imported.")
except Exception as e:
    print("Failed: qwen_tts import error.")
    traceback.print_exc()
