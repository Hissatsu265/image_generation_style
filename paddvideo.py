from moviepy.editor import VideoFileClip, ColorClip, CompositeVideoClip

# async def add_green_background(input_video: str, output_video: str, target_w: int = 1280, target_h: int = 720):

#     clip = VideoFileClip(input_video)
#     orig_w, orig_h = clip.size

#     green_bg = ColorClip(size=(target_w, target_h), color=(0, 255, 0), duration=clip.duration)

#     scale = target_h / orig_h
#     new_w = int(orig_w * scale)
#     new_h = int(orig_h * scale)
#     resized_clip = clip.resize((new_w, new_h))

#     x_center = (target_w - new_w) // 2
#     y_center = (target_h - new_h) // 2

#     final = CompositeVideoClip([green_bg, resized_clip.set_position((x_center, y_center))])

#     final.write_videofile(output_video, codec="libx264", audio_codec="aac")

import cv2

def detect_face_center(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_center = (x + w // 2, y + h // 2)
    return face_center, frame.shape[1], frame.shape[0]  # (cx, cy), width, height


async def add_green_background(input_video: str, output_video: str, target_w: int = 1280, target_h: int = 720):
    clip = VideoFileClip(input_video)
    orig_w, orig_h = clip.size

    # Tính scale sao cho video vừa nhất trong khung (letterbox)
    scale_w = target_w / orig_w
    scale_h = target_h / orig_h
    scale = min(scale_w, scale_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    # print(new_h)
    # print(new_w)
    resized_clip = clip.resize((new_w, new_h))

    green_bg = ColorClip(size=(target_w, target_h), color=(0, 255, 0), duration=clip.duration)

    # Vị trí căn giữa (nếu có face thì có thể tinh chỉnh sau)
    # face_info = detect_face_center(input_video)
    # if face_info is not None:
    #     (cx, cy), frame_w, frame_h = face_info
    #     cx_scaled = int(cx * scale)
    #     cy_scaled = int(cy * scale)
    #     x_center = target_w // 2 - cx_scaled
    #     y_center = target_h // 2 - cy_scaled
    # else:
        # Căn giữa mặc định
    x_center = (target_w - new_w) // 2
    y_center = (target_h - new_h) // 2

    # Ghép video lên nền
    final = CompositeVideoClip([green_bg, resized_clip.set_position((x_center, y_center))])
    final.write_videofile(output_video, codec="libx264", audio_codec="aac")

# add_green_background("/content/26eaea4a_00001-audio.mp4", "output.mp4", 1920, 1080)
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

        lower_green = np.array([35, 60, 60])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological cleaning
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Feathering edges
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        # Normalize alpha mask
        alpha = mask.astype(float) / 255.0
        alpha_inv = 1 - alpha

        fg = frame.astype(float) * alpha_inv[:, :, None]
        bg = bg_img.astype(float) * alpha[:, :, None]

        combined = fg + bg
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
# ===============================================================================
import numpy as np
from PIL import Image

def crop_green_background(image_path: str, output_path: str, margin: float = 0.04):
    """
    Crop bỏ nền xanh, giữ lại foreground (người/vật thể).
    margin: phần trăm padding thêm quanh bounding box (0.05 = 5%).
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)


    mask_fg = cv2.bitwise_not(mask_green)

    # Contours
    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("❌ Không tìm thấy foreground")
        return

    # bounding box bao quanh toàn bộ foreground
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))

    
    pad_w = int(w * margin)
    pad_h = int(h * margin)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)
    cropped = img_rgb[y1:y2, x1:x2]

    Image.fromarray(cropped).save(output_path)
    print(f"✅ Saved cropped image: {output_path}, size={cropped.shape[1]}x{cropped.shape[0]}")
# crop_green_background("/content/3.png", "cropped.jpg", margin=0.05)


# =======================================================================
# from PIL import Image, ImageOps

# def resize_and_pad(image_path: str, output_path: str):

from PIL import Image
import math

def resize_and_pad(image_path: str, output_path: str):
    print("outputpath: ", output_path)
    img = Image.open(image_path)
    original_width, original_height = img.size
    
    print(f"Kích thước gốc: {original_width} x {original_height}")
    
    # Bước 1: Padding để các cạnh chia hết cho 16
    # Tính kích thước mới (làm tròn lên để chia hết cho 16)
    new_width = math.ceil(original_width / 16) * 16
    new_height = math.ceil(original_height / 16) * 16
    
    # Tính padding cần thiết
    width_padding = new_width - original_width
    height_padding = new_height - original_height
    
    # Padding ngang: chia đều 2 bên
    left_padding = width_padding // 2
    right_padding = width_padding - left_padding
    
    # Padding dọc: thêm ở trên
    top_padding = height_padding
    bottom_padding = 0
    
    # Tạo ảnh mới với padding màu xanh #00B140
    padded_img = Image.new('RGB', (new_width, new_height), '#00B140')
    
    # Dán ảnh gốc vào vị trí phù hợp
    padded_img.paste(img, (left_padding, top_padding))
    
    print(f"Sau padding: {new_width} x {new_height}")
    print(f"Padding: trái={left_padding}, phải={right_padding}, trên={top_padding}, dưới={bottom_padding}")
    
    # Bước 2: Resize để tích hai cạnh nằm trong khoảng mong muốn
    target_min_area = 278784  # Tích tối thiểu
    target_max_area = 409440  # Tích tối đa
    current_area = new_width * new_height
    
    print(f"Diện tích hiện tại: {current_area}")
    
    if current_area < target_min_area:
        # Cần phóng to
        scale_factor = math.sqrt(target_min_area / current_area)
        final_width = int(new_width * scale_factor)
        final_height = int(new_height * scale_factor)
        
        # Đảm bảo tích không vượt quá target_max_area
        if final_width * final_height > target_max_area:
            scale_factor = math.sqrt(target_max_area / current_area)
            final_width = int(new_width * scale_factor)
            final_height = int(new_height * scale_factor)
            
    elif current_area > target_max_area:
        # Cần thu nhỏ
        scale_factor = math.sqrt(target_max_area / current_area)
        final_width = int(new_width * scale_factor)
        final_height = int(new_height * scale_factor)
        
        # Đảm bảo tích không nhỏ hơn target_min_area
        if final_width * final_height < target_min_area:
            scale_factor = math.sqrt(target_min_area / current_area)
            final_width = int(new_width * scale_factor)
            final_height = int(new_height * scale_factor)
    else:
        # Diện tích đã phù hợp, không cần resize
        final_width = new_width
        final_height = new_height
    
    # Resize ảnh
    if final_width != new_width or final_height != new_height:
        final_img = padded_img.resize((final_width, final_height), Image.Resampling.LANCZOS)
        print(f"Sau resize: {final_width} x {final_height}")
    else:
        final_img = padded_img
        print("Không cần resize")
    
    final_area = final_width * final_height
    print(f"Diện tích cuối: {final_area}")
    print(f"Trong khoảng mục tiêu: {target_min_area <= final_area <= target_max_area}")
    
    # Lưu ảnh
    final_img.save(output_path, quality=95)
    print(f"Đã lưu ảnh tại: {output_path}")
    

# Ví dụ chạy
# resize_and_pad("/content/0f8d203c-dc10-4094-8b9e-45864d2cf337 sdsadasfafa.JPG", "output.jpg")
