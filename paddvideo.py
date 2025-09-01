from moviepy.editor import VideoFileClip, ColorClip, CompositeVideoClip

async def add_green_background(input_video: str, output_video: str, target_w: int = 1280, target_h: int = 720):

    clip = VideoFileClip(input_video)
    orig_w, orig_h = clip.size

    green_bg = ColorClip(size=(target_w, target_h), color=(0, 255, 0), duration=clip.duration)

    scale = target_h / orig_h
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_clip = clip.resize((new_w, new_h))

    x_center = (target_w - new_w) // 2
    y_center = (target_h - new_h) // 2

    final = CompositeVideoClip([green_bg, resized_clip.set_position((x_center, y_center))])

    final.write_videofile(output_video, codec="libx264", audio_codec="aac")

# Ví dụ sử dụng
# add_green_background("/content/input.mp4", "output1.mp4")
# =============================================================================================
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os
import tempfile
import shutil

def replace_green_screen(video_path, background_path=None):
    # Tạo file tạm để xuất video trước (tránh ghi đè khi đang đọc)
    base, ext = os.path.splitext(video_path)
    temp_output = base + "_temp" + ext

    # Load video
    clip = VideoFileClip(video_path)
    w, h = clip.size
    fps = clip.fps

    # Nếu không có background_path thì tạo background màu #00B140
    if background_path is None or not os.path.exists(background_path):
        bg_img = np.full((h, w, 3), (0, 177, 64), dtype=np.uint8)  # màu #00B140
    else:
        # Load background image
        bg_img = cv2.imread(background_path)
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        # Resize background theo video
        bg_img = cv2.resize(bg_img, (w, h))

    # Hàm xử lý từng frame
    def process_frame(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Ngưỡng màu xanh
        lower_green = np.array([35, 60, 60])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Làm mịn mask để tránh viền cứng
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Chuẩn hóa mask về 0-1 để blend
        mask_normalized = mask.astype(float) / 255.0
        mask_inv = 1 - mask_normalized

        # Blend mượt giữa foreground và background
        fg = frame.astype(float) * mask_inv[:, :, None]
        bg = bg_img.astype(float) * mask_normalized[:, :, None]

        combined = cv2.addWeighted(fg, 1.0, bg, 1.0, 0.0)
        return combined.astype(np.uint8)

    # Áp dụng filter
    new_clip = clip.fl_image(process_frame)

    # Giữ lại audio gốc
    new_clip = new_clip.set_audio(clip.audio)

    # Xuất ra file tạm
    new_clip.write_videofile(temp_output, codec="libx264", audio_codec="aac", fps=fps)

    # Ghi đè file gốc
    shutil.move(temp_output, video_path)

    return video_path


# replace_green_screen(
#     video_path="/content/merged_videoddd.mp4",
#     background_path="/content/OIP.png",
#     output_path="output.mp4"
# )
