from pydub import AudioSegment
from pathlib import Path

def add_silence_to_start(audio_path, jobid, duration_ms=500):
    audio = AudioSegment.from_file(audio_path)
    silence = AudioSegment.silent(duration=duration_ms)
    output = silence + audio
    audio_path = Path(audio_path)
    output_path = audio_path.with_name(f"{audio_path.stem}_{jobid}{audio_path.suffix}")
    output.export(str(output_path), format=audio_path.suffix.replace(".", ""))
    return str(output_path)


# out = add_silence_to_start("sample.mp3", jobid="123456")
# print("File output:", out)
from moviepy.editor import VideoFileClip
import os

def trim_video_start(video_path, duration=0.5):
    video_path = Path(video_path)
    clip = VideoFileClip(str(video_path))
    trimmed = clip.subclip(duration, clip.duration)
    temp_path = video_path.with_name(f"{video_path.stem}_temp{video_path.suffix}")
    trimmed.write_videofile(str(temp_path), codec="libx264", audio_codec="aac")
    
    os.remove(video_path)
    
    os.rename(temp_path, video_path)
    
    return str(video_path)



# out = trim_video_start("sample.mp4", duration=0.5)
# print("File output:", out)
