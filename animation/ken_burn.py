from moviepy.editor import ImageClip, concatenate_videoclips
import numpy as np
import cv2

def ken_burns_effect(image_path, output_path="ken_burns_output.mp4", duration=6, scale=0.8, fps=30):
    # Đọc ảnh
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Kích thước khung hình nhỏ hơn
    crop_h, crop_w = int(h * scale), int(w * scale)
    step_count = int(duration * fps)

    frames = []

    for i in range(step_count):
        progress = i / step_count

        # Giai đoạn đầu: từ trái qua phải (nửa thời gian đầu)
        if progress <= 0.5:
            x = int(progress * 2 * (w - crop_w))  # trái ➝ phải
            y = 0
        else:
            # Giai đoạn sau: phải qua trái ở dưới
            x = int((1 - (progress - 0.5) * 2) * (w - crop_w))  # phải ➝ trái
            y = h - crop_h

        # Cắt ảnh con
        cropped = image[y:y+crop_h, x:x+crop_w]
        resized = cv2.resize(cropped, (w, h))
        frames.append(resized)

    # Ghi video bằng MoviePy
    def make_frame(t):
        idx = min(int(t * fps), len(frames) - 1)
        return frames[idx][:, :, ::-1]  # BGR ➝ RGB

    clip = ImageClip(frames[0][:, :, ::-1], duration=duration)
    video = clip.set_make_frame(make_frame).set_duration(duration)
    video.write_videofile(output_path, fps=fps)

# Ví dụ sử dụng
ken_burns_effect("/content/phone.jpg", output_path="output_ken_burns.mp4", duration=6)
