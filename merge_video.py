# !pip install moviepy
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

def concat_videos(video_paths, output_path="output_combined.mp4"):

    if not video_paths:
        raise ValueError("Danh sách video rỗng.")
    try:
        clips = [VideoFileClip(path) for path in video_paths]
    except Exception as e:
        raise RuntimeError(f"Lỗi khi đọc video: {e}")
    final_clip = concatenate_videoclips(clips, method="compose")
    try:
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    except Exception as e:
        raise RuntimeError(f"Lỗi khi ghi video: {e}")
    finally:
        for clip in clips:
            clip.close()

    return os.path.abspath(output_path)
# if __name__ == "__main__":
#     video_list = [
#         "/workspace/multitalk_verquant/d1.mp4",
#         "/workspace/multitalk_verquant/d2.mp4", 
#         # "/workspace/multitalk_verquant/c3.mp4",
#         "/workspace/multitalk_verquant/d3.mp4",
#         # "/workspace/multitalk_verquant/c5.mp4",
#         # "/workspace/multitalk_verquant/c6.mp4"
#         ]
#     result_path = concat_videos(video_list, "merged_videoddd.mp4")
#     print("Video đã được tạo tại:", result_path)
