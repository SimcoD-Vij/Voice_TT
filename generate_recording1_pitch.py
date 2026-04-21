import os
import sys
import numpy as np
import soundfile as sf
from f5_tts.api import F5TTS

def generate_recording1_pitch():
    # Initialize F5-TTS
    print("Initializing IndicF5 model...")
    hf_cache_dir = r"D:\hf_cache"
    f5tts = F5TTS(hf_cache_dir=hf_cache_dir)

    # Define paths
    base_dir = r"D:\voice tts"
    output_dir = os.path.join(base_dir, "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Reference data from recording_1.wav
    ref_file = os.path.join(base_dir, "dataset", "processed", "recording_1.wav")
    # Transcript for the reference
    ref_text = "Bella replied that numbers also have unique pronunciations. She slowly counted 0, 1, 2, 3."

    # 1. English Pitch with Recording 1 Voice
    print("Generating English sales pitch with Recording 1 voice...")
    en_gen_text = (
        "Protect your smartphone and extend your battery life with XOptimus. "
        "Our AI-powered smart charger, the XOptimus wall adapter, intelligently monitors your battery to prevent overcharging and overheating. "
        "With dedicated modes for everyday use, gaming, and full charging, it's the smartest way to power your expensive devices. "
        "Made in India by Hivericks Technologies, XOptimus is the investment your phone deserves. "
        "Get yours today for just fourteen ninety-nine rupees."
    )
    
    en_output_file = os.path.join(output_dir, "xoptimus_pitch_rec1_en.wav")
    
    f5tts.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=en_gen_text,
        file_wave=en_output_file
    )
    print(f"Recording 1 voice English pitch saved to: {en_output_file}")

    # 2. Tamil Pitch with Recording 1 Voice
    print("Generating Tamil sales pitch with Recording 1 voice...")
    ta_gen_text = (
        "உங்கள் ஸ்மார்ட்போனைப் பாதுகாக்கவும், பேட்டரி ஆயுளை நீட்டிக்கவும் எக்ஸ்-ஆப்டிமஸ் ஐத் தேர்ந்தெடுங்கள். "
        "எங்களின் ஏஐ மூலம் இயங்கும் ஸ்மார்ட் சார்ஜர், உங்கள் பேட்டரியை புத்திசாலித்தனமாக கண்காணித்து, அதிகப்படியான சார்ஜிங் மற்றும் வெப்பமடைவதைத் தடுக்கிறது. "
        "தினசரி பயன்பாடு, கேமிங் மற்றும் முழு சார்ஜிங்கிற்கான பிரத்யேக மோட்களுடன், உங்கள் விலையுயர்ந்த சாதனங்களுக்கு இதுவே சிறந்த தேர்வாகும். "
        "ஹெய்வெரிக்ஸ் டெக்னாலஜிஸ் மூலம் இந்தியாவில் தயாரிக்கப்பட்ட எக்ஸ்-ஆப்டிமஸ், உங்கள் போனுக்குத் தேவையான ஒரு சிறந்த முதலீடாகும். "
        "இன்று வெறும் ஆயிரத்து நானூற்று தொண்ணூற்றி ஒன்பது ரூபாய்க்கு இதைப் பெறுங்கள்."
    )
    
    ta_output_file = os.path.join(output_dir, "xoptimus_pitch_rec1_ta.wav")
    
    f5tts.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=ta_gen_text,
        file_wave=ta_output_file
    )
    print(f"Recording 1 voice Tamil pitch saved to: {ta_output_file}")

if __name__ == "__main__":
    generate_recording1_pitch()
