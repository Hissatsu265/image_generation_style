from moviepy.editor import VideoFileClip, AudioFileClip

def replace_audio_trimmed(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    min_duration = min(video.duration, audio.duration)
    video = video.subclip(0, min_duration)
    audio = audio.subclip(0, min_duration)
    final = video.set_audio(audio)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path
if __name__ == "__main__":
    video_path = "/workspace/multitalk_verquant/merged_videoddd.mp4"
    audio_path = "/workspace/multitalk_verquant/audio/audio_rs_demo_side_view.mp3"
    output_path = "output_video_with_audioddd.mp4"
    result_path = replace_audio_trimmed(video_path, audio_path, output_path)
    print("Video with replaced audio created at:", result_path)