from pydub import AudioSegment

def create_silent_audio(duration_sec, output_path="silent_audio.mp3"):
    """
    Tạo file audio im lặng với thời lượng tùy chọn.

    Args:
        duration_sec (float): Thời lượng im lặng (tính bằng giây).
        output_path (str): Đường dẫn file đầu ra (hỗ trợ .mp3, .wav, ...).
    """
    duration_ms = int(duration_sec * 1000)
    silent_audio = AudioSegment.silent(duration=duration_ms)
    silent_audio.export(output_path, format=output_path.split(".")[-1])
    print(f"✅ Saved silent audio: {output_path} ({duration_sec:.2f}s)")

# Ví dụ:
create_silent_audio(0.5, "silent_0.5s.mp3")
