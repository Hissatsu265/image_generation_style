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
from PIL import Image, ImageOps

def resize_and_pad(image_path: str, output_path: str):
    target_min, target_max = 496, 790
    max_area = 270336
    bg_color = (0, 177, 64)

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    aspect = orig_w / orig_h

    # B1: resize về khoảng [512, 790]
    scale = 1.0
    if orig_w < target_min or orig_h < target_min:
        scale = max(target_min / orig_w, target_min / orig_h)
    elif orig_w > target_max or orig_h > target_max:
        scale = min(target_max / orig_w, target_max / orig_h)

    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # B2: làm tròn 16
    def round_to_16(x): return (x + 15) // 16 * 16
    target_w, target_h = round_to_16(new_w), round_to_16(new_h)

    # B3: giữ trong giới hạn
    if target_w > target_max: target_w = target_max // 16 * 16
    if target_h > target_max: target_h = target_max // 16 * 16

    # B4: check diện tích
    while target_w * target_h >= max_area:
        if target_w >= target_h and target_w > target_min:
            target_w -= 16
        elif target_h > target_min:
            target_h -= 16
        else:
            break

    # B5: tạo nền và dán
    bg = Image.new("RGB", (target_w, target_h), bg_color)

    # Nếu cần pad ngang → căn giữa trái-phải
    if target_w > new_w and target_h == new_h:
        offset_x = (target_w - new_w) // 2
        offset_y = 0  # không pad bottom
    # Nếu cần pad dọc → pad lên trên
    elif target_h > new_h and target_w == new_w:
        offset_x = 0
        offset_y = target_h - new_h  # chỉ pad phía trên
    else:
        offset_x = 0
        offset_y = 0

    bg.paste(img, (offset_x, offset_y))
    bg.save(output_path)
    print(f"✅ Saved: {output_path}, size={bg.size}, area={bg.size[0]*bg.size[1]}")

# Ví dụ chạy
# resize_and_pad("/content/0f8d203c-dc10-4094-8b9e-45864d2cf337 sdsadasfafa.JPG", "output.jpg")
