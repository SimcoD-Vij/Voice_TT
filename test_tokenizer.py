from pocket_tts import TTSModel
from pocket_tts.conditioners.text import SentencePieceTokenizer
import os

def test_tokenizer():
    try:
        model = TTSModel.load_model(language="english")
        tokenizer = model.text_conditioner.tokenizer
        text = "Hello World."
        token_ids = tokenizer.encode(text)
        print(f"Text: {text}")
        print(f"Token IDs: {token_ids}")
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded: {decoded}")
        if decoded.lower().strip(".") == text.lower().strip("."):
            print("Tokenizer is working correctly for English.")
        else:
            print("Tokenizer failure! English is not being tokenized correctly.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_tokenizer()
