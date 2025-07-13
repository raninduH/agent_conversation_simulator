from kokoro import KPipeline
import soundfile as sf
import torch
import os
import subprocess
import sys

pipeline = KPipeline(lang_code='a')
text = '''
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''

print("Generating audio with Kokoro...")
generator = pipeline(text, voice='am_adam')

audio_files = []
for i, (gs, ps, audio) in enumerate(generator):
    print(f"Chunk {i}: gs={gs}, ps={ps}, audio_shape={audio.shape}")
    filename = f'{i}.wav'
    sf.write(filename, audio, 24000)
    audio_files.append(filename)
    print(f"Saved: {filename}")

print(f"\nGenerated {len(audio_files)} audio files:")
for file in audio_files:
    print(f"  - {file}")

# Function to play audio on Windows
def play_audio(filename):
    """Play audio file using Windows default player"""
    try:
        if sys.platform == "win32":
            os.startfile(filename)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", filename])
        else:  # Linux
            subprocess.run(["xdg-open", filename])
        return True
    except Exception as e:
        print(f"Could not play {filename}: {e}")
        return False

# Play the first audio file
if audio_files:
    print(f"\nAttempting to play first audio file: {audio_files[0]}")
    if play_audio(audio_files[0]):
        print("Audio should be playing in your default audio player")
    else:
        print("Could not auto-play audio. Please manually open the .wav files to listen")

print(f"\nTest completed! Check the generated .wav files in: {os.getcwd()}")