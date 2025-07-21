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