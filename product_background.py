import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

# def remove_green_background(frame, background):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
#     # Mặt nạ cho màu xanh lá (HSV)
#     lower_green = np.array([35, 50, 50])
#     upper_green = np.array([85, 255, 255])
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     mask_inv = cv2.bitwise_not(mask)

#     fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
#     bg = cv2.bitwise_and(background, background, mask=mask)
#     combined = cv2.add(fg, bg)
#     return combined
# =================================================================
# def remove_green_background(frame, background):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
#     # Mặt nạ cho màu xanh lá (HSV)
#     lower_green = np.array([35, 50, 50])
#     upper_green = np.array([85, 255, 255])
#     mask = cv2.inRange(hsv, lower_green, upper_green)

#     # Làm sạch mặt nạ bằng erosion/dilation
#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.erode(mask, kernel, iterations=1)
#     mask = cv2.dilate(mask, kernel, iterations=2)

#     # Làm mờ mặt nạ để làm mềm biên
#     mask = cv2.GaussianBlur(mask, (5, 5), 0)

#     # Tạo mask nghịch
#     mask = mask.astype(np.float32) / 255.0
#     mask_inv = 1.0 - mask

#     # Kết hợp foreground và background với alpha blending
#     fg = frame.astype(np.float32) * mask_inv[..., np.newaxis]
#     bg = background.astype(np.float32) * mask[..., np.newaxis]
#     combined = fg + bg
#     return combined.astype(np.uint8)


# def apply_green_screen_on_image(video_path, background_path, output_path, start_time, end_time):
#     clip = VideoFileClip(video_path)
#     width, height = clip.w, clip.h

#     # Load và resize ảnh nền khớp với video
#     background = cv2.imread(background_path)
#     background = cv2.resize(background, (width, height))
#     background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

#     writer = FFMPEG_VideoWriter(output_path, (width, height), clip.fps)

#     for t in np.arange(0, clip.duration, 1.0 / clip.fps):
#         frame = clip.get_frame(t)
#         if start_time <= t <= end_time:
#             frame = remove_green_background(frame, background)
#         writer.write_frame(frame)

#     writer.close()
#     print(f"✅ Xuất video hoàn tất: {output_path}")
# ============================================================
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

def remove_green_background(frame, background):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Mặt nạ cho màu xanh lá (HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Làm sạch và làm mềm biên
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Alpha blending
    mask = mask.astype(np.float32) / 255.0
    mask_inv = 1.0 - mask

    fg = frame.astype(np.float32) * mask_inv[..., np.newaxis]
    bg = background.astype(np.float32) * mask[..., np.newaxis]
    combined = fg + bg
    return combined.astype(np.uint8)

def apply_green_screen_on_image(video_path, background_path, output_path, start_time, end_time):
    clip = VideoFileClip(video_path)
    width, height = clip.w, clip.h
    crop_percent = 0.3
    crop_pixels = int(height * crop_percent)

    # Load và resize ảnh nền
    background = cv2.imread(background_path)
    background = cv2.resize(background, (width, height))
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    writer = FFMPEG_VideoWriter(output_path, (width, height), clip.fps)

    for t in np.arange(0, clip.duration, 1.0 / clip.fps):
        frame = clip.get_frame(t)

        if start_time <= t <= end_time:
            # Cắt 15% dưới + pad 15% trên bằng màu xanh
            frame = frame[:height - crop_pixels, :, :]
            green_pad = np.full((crop_pixels, width, 3), (0, 255, 0), dtype=np.uint8)
            frame = np.vstack((green_pad, frame))

            frame = remove_green_background(frame, background)

        writer.write_frame(frame)

    writer.close()
    print(f"✅ Xuất video hoàn tất: {output_path}")

# Ví dụ sử dụng
apply_green_screen_on_image(
    video_path="/content/merged_video (15).mp4",
    background_path="/content/resized/cropped_image.jpg",
    output_path="output4.mp4",
    start_time=10.16,
    end_time=20.59
)
