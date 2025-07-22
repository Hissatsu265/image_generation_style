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
    
    # Load file audio
    audio = AudioSegment.from_file(input_path)
    
    # Chuyển giây sang milliseconds
    start_ms = int(start_time_sec * 1000)
    
    # Cắt từ start_ms đến hết file
    if start_ms > len(audio):
        raise ValueError("Thời gian bắt đầu lớn hơn độ dài file audio.")

    trimmed_audio = audio[start_ms:]

    # Tạo tên file output nếu chưa có
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cut_from_{start_time_sec}s{ext}"
    
    # Lưu file
    trimmed_audio.export(output_path, format=ext[1:])  # bỏ dấu chấ
