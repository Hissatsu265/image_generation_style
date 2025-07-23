from pydub import AudioSegment

def merge_audio_files(audio_path1, audio_path2, output_path="merged_audio.mp3"):
    audio1 = AudioSegment.from_file(audio_path1)
    audio2 = AudioSegment.from_file(audio_path2)
    merged = audio1 + audio2
    merged.export(output_path, format="wav")
    print(f"✅ Merged audio saved to: {output_path}")
    return output_path

# Ví dụ sử dụng
# merge_audio_files("audio1.wav", "audio2.wav", "output.wav")
