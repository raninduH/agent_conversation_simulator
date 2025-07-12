"""
Script to generate a voice sample for each Kokoro voice and save the audio files.
"""

import os
import json
import base64
import requests

# Path to kokoro_voices.json
VOICES_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kokoro_voices.json")
# Output directory for samples
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "kokor_voice_samples")
# Kokoro API endpoint (adjust if needed)
KOKORO_API_URL = "http://localhost:8001/synthesize"

SAMPLE_TEXT = "Hi, how are you? it's a pleasure to meet you."

def load_voices():
    with open(VOICES_JSON, "r", encoding="utf-8") as f:
        voices = json.load(f)
    # Flatten to a list of (voice, gender)
    all_voices = []
    for gender, voice_list in voices.items():
        for v in voice_list:
            all_voices.append((v, gender))
    return all_voices

def synthesize_voice(voice_name, text):
    payload = {
        "text": text,
        "voice": voice_name,
        "return_format": "base64"
    }
    try:
        response = requests.post(KOKORO_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            audio_base64 = data.get("audio_base64")
            if audio_base64:
                return base64.b64decode(audio_base64)
            else:
                print(f"Warning: No audio_base64 in response for {voice_name}")
        else:
            print(f"Error: {voice_name} - HTTP {response.status_code}")
    except Exception as e:
        print(f"Exception for {voice_name}: {e}")
    return None

def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    voices = load_voices()
    for voice_name, gender in voices:
        print(f"Generating sample for {voice_name} ({gender})...")
        audio_data = synthesize_voice(voice_name, SAMPLE_TEXT)
        if audio_data:
            filename = f"{voice_name}_{gender}.wav"
            filepath = os.path.join(SAMPLES_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(audio_data)
            print(f"Saved: {filepath}")
        else:
            print(f"Failed to generate audio for {voice_name}")

if __name__ == "__main__":
    main()