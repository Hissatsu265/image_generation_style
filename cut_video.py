from moviepy.editor import VideoFileClip
import os

def cut_video(input_path, end_time, output_path=None):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    clip = VideoFileClip(input_path)
    
    end_time = min(end_time, clip.duration)

    subclip = clip.subclip(0, round(end_time, 3))

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cut_{round(end_time, 3)}s{ext}"

    subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return output_path


def cut_video_1(input_path, start_time=0.0, end_time=None, output_path=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    clip = VideoFileClip(input_path)
    
    if end_time is None:
        end_time = clip.duration

    # Đảm bảo thời gian nằm trong giới hạn video
    start_time = max(0, min(start_time, clip.duration))
    end_time = max(start_time, min(end_time, clip.duration))

    subclip = clip.subclip(round(start_time, 3), round(end_time, 3))

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cut_{round(start_time, 3)}s_to_{round(end_time, 3)}s{ext}"

    subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return output_path
from pydub import AudioSegment
import os

def cut_audio_from_time(input_path, start_time_sec, output_path=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")
    
    audio = AudioSegment.from_file(input_path)
    
    start_ms = int(start_time_sec * 1000)
    
    if start_ms > len(audio):
        raise ValueError("Thời gian bắt đầu lớn hơn độ dài file audio.")

    trimmed_audio = audio[start_ms:]

    base, ext = os.path.splitext(input_path)  # ✅ luôn lấy ext từ input
    if not output_path:
        output_path = f"{base}_cut_from_{start_time_sec}s{ext}"
    
    trimmed_audio.export(output_path, format=ext[1:])  # safe vì ext luôn có
    print(f"✅ Đã cắt audio từ {start_time_sec}s và lưu tại: {output_path}")
    return output_path

def cut_audio(input_path: str, output_path: str, end_time_sec: float):
    from pydub import AudioSegment
    import os

    # Đọc audio
    audio = AudioSegment.from_file(input_path)
    
    # Cắt từ đầu đến thời điểm chỉ định
    end_time_ms = int(end_time_sec * 1000)
    cut_audio = audio[:end_time_ms]

    # Tạo thư mục nếu cần
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Lưu audio
    cut_audio.export(output_path, format=os.path.splitext(output_path)[-1][1:])
    print(f"✅ Đã cắt audio và lưu tại: {output_path}")
    return output_path
