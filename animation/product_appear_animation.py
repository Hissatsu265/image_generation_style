!pip install face_recognition

import cv2
import face_recognition
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip

def process_video_with_product(
    video_path, product_img_path, start_time, end_time, output_path="output.mp4", hold_duration=1.5
):
    video = VideoFileClip(video_path)
    fps = video.fps
    w, h = video.size

    frame_img = video.get_frame(0)
    face_locations = face_recognition.face_locations(frame_img)

    if not face_locations:
        print("Không tìm thấy khuôn mặt.")
        return

    top, right, bottom, left = face_locations[0]
    face_center_y = (top + bottom) // 2
    distance_top = top
    distance_bottom = h - bottom

    insert_y = (distance_top // 2) if distance_top > distance_bottom else bottom + (distance_bottom // 2)
    insert_x = w // 2

    product_img = cv2.imread(product_img_path, cv2.IMREAD_UNCHANGED)
    if product_img is None or product_img.shape[2] != 4:
        print("Ảnh sản phẩm phải có nền trong suốt (PNG 4 kênh).")
        return

    max_product_height = int((bottom - top) * 1.2)  # ⬅️ tăng kích thước hơn chút
    scale = max_product_height / product_img.shape[0]
    product_w = int(product_img.shape[1] * scale)
    product_h = int(product_img.shape[0] * scale)

    resized_product = cv2.resize(product_img, (product_w, product_h))

    def product_effect(t):
        if start_time <= t <= end_time:
            progress = (t - start_time) / (end_time - start_time)
            scale_factor = 0.3 + 1.0 * progress  # ⬅️ lớn hơn
            fade_alpha = int(255 * progress)
        elif end_time < t <= end_time + hold_duration:
            scale_factor = 1.3  # giữ nguyên
            fade_alpha = 255
        else:
            return None

        resized = cv2.resize(resized_product, (
            int(product_w * scale_factor),
            int(product_h * scale_factor)
        ))

        img = resized.copy()
        img[..., 3] = (img[..., 3] * (fade_alpha / 255)).astype(np.uint8)

        return ImageClip(img, transparent=True).set_position((
            insert_x - img.shape[1] // 2,
            insert_y - img.shape[0] // 2
        )).set_duration(0.05)

    product_clips = []
    t = start_time
    step = 1.0 / fps
    while t <= end_time + hold_duration:
        clip = product_effect(t)
        if clip:
            clip = clip.set_start(t)
            product_clips.append(clip)
        t += step

    final = CompositeVideoClip([video] + product_clips)
    final.write_videofile(output_path, codec="libx264", audio=True)

# === Gọi hàm chính ===
process_video_with_product(
    video_path="/content/merged_video (18).mp4",
    product_img_path="/content/ai-generated-dslr-photo-camera-object-photo-png.png",
    start_time=1.5,
    end_time=1.9,
    output_path="final_output.mp4",
    hold_duration=7.0  # ⬅️ giữ hình thêm 2 giây
)
