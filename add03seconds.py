from pydub import AudioSegment
import os
from audio_duration import get_audio_duration


def add_silence_to_audio(input_path, silence_duration_ms=300):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    audio = AudioSegment.from_file(input_path)
    silence = AudioSegment.silent(duration=silence_duration_ms)
    new_audio = audio + silence

    # Tạo tên file mới
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_padded{ext}"

    new_audio.export(output_path, format=ext.replace('.', ''))
    return output_path
# print(get_audio_duration("/workspace/multitalk_verquant/audio/ella_de.wav"))
# output = add_silence_to_audio("/workspace/multitalk_verquant/audio/ella_de.wav")
# print("New audio saved to:", output)
# print("Duration of new audio:", get_audio_duration(output))