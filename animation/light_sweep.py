import cv2
import numpy as np

def crop_and_resize_image(img, target_width, target_height):
    """
    Crop ảnh về tỉ lệ khung hình mong muốn và resize về kích thước cụ thể
    
    Args:
        img: Ảnh đầu vào
        target_width: Chiều rộng mục tiêu
        target_height: Chiều cao mục tiêu
    
    Returns:
        Ảnh đã được crop và resize
    """
    h, w = img.shape[:2]
    target_ratio = target_width / target_height
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        # Ảnh quá rộng, cần crop chiều rộng
        new_width = int(h * target_ratio)
        start_x = (w - new_width) // 2
        cropped = img[:, start_x:start_x + new_width]
    else:
        # Ảnh quá cao, cần crop chiều cao
        new_height = int(w / target_ratio)
        start_y = (h - new_height) // 2
        cropped = img[start_y:start_y + new_height, :]
    
    # Resize về kích thước mục tiêu
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    return resized

def apply_light_sweep(img, sweep_pos, sweep_width=100, intensity=1.5):
    """
    Áp dụng hiệu ứng quét ánh sáng
    
    Args:
        img: Ảnh đầu vào
        sweep_pos: Vị trí hiện tại của ánh sáng
        sweep_width: Độ rộng của vùng sáng
        intensity: Cường độ ánh sáng
    
    Returns:
        Ảnh với hiệu ứng ánh sáng
    """
    h, w = img.shape[:2]
    
    # Tạo mask sáng
    mask = np.zeros((h, w), dtype=np.float32)
    start = max(0, sweep_pos - sweep_width // 2)
    end = min(w, sweep_pos + sweep_width // 2)
    
    for i in range(start, end):
        brightness = 1 - abs(i - sweep_pos) / (sweep_width / 2)
        brightness = np.clip(brightness * intensity, 0, 1)
        mask[:, i] = brightness
    
    # Làm mượt mask
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15, sigmaY=15)
    
    # Nhân mask vào ảnh gốc
    light = img.astype(np.float32) * (1 + mask[:, :, None])
    light = np.clip(light, 0, 255).astype(np.uint8)
    return light

def create_light_sweep_video(input_path, output_path, 
                           target_width=1920, target_height=1080,
                           sweep_width=150, intensity=1.5,
                           video_duration=3.0, fps=30,
                           sweep_speed_multiplier=1.0):
    """
    Tạo video với hiệu ứng quét ánh sáng
    
    Args:
        input_path: Đường dẫn ảnh đầu vào
        output_path: Đường dẫn video đầu ra
        target_width: Chiều rộng video mục tiêu
        target_height: Chiều cao video mục tiêu
        sweep_width: Độ rộng vùng sáng
        intensity: Cường độ ánh sáng
        video_duration: Thời lượng video (giây)
        fps: Số khung hình/giây
        sweep_speed_multiplier: Hệ số tốc độ quét (1.0 = bình thường, >1 = nhanh hơn, <1 = chậm hơn)
    """
    
    # Load và xử lý ảnh
    print("📸 Đang load và xử lý ảnh...")
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Không thể load ảnh từ {input_path}")
    
    # Crop và resize ảnh
    img_processed = crop_and_resize_image(img, target_width, target_height)
    print(f"✅ Đã xử lý ảnh: {img.shape} -> {img_processed.shape}")
    
    # Tính toán thông số video
    total_frames = int(video_duration * fps)
    
    # Tính toán khoảng cách quét
    # Quét từ ngoài trái (-sweep_width) đến ngoài phải (width + sweep_width)
    total_sweep_distance = target_width + 2 * sweep_width
    
    # Áp dụng hệ số tốc độ
    actual_sweep_distance = total_sweep_distance * sweep_speed_multiplier
    
    print(f"🎬 Tạo video: {total_frames} frames, {fps} fps, {video_duration}s")
    print(f"⚡ Tốc độ quét: {sweep_speed_multiplier}x")
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    # Tạo từng frame
    for frame_idx in range(total_frames):
        # Tính vị trí ánh sáng hiện tại
        progress = frame_idx / (total_frames - 1)  # 0 -> 1
        sweep_pos = -sweep_width + progress * actual_sweep_distance
        sweep_pos = int(sweep_pos)
        
        # Tạo frame với hiệu ứng
        frame = apply_light_sweep(img_processed, sweep_pos, sweep_width, intensity)
        out.write(frame)
        
        # Hiển thị tiến độ
        if frame_idx % 10 == 0:
            percent = (frame_idx / total_frames) * 100
            print(f"⏳ Tiến độ: {percent:.1f}% ({frame_idx}/{total_frames})")
    
    out.release()
    print(f"✅ Hoàn thành! Đã lưu video: {output_path}")

# ====== CÁCH SỬ DỤNG ======

if __name__ == "__main__":
    # Cấu hình cơ bản
    input_image = "/content/coca_sp.jpg"  # Đường dẫn ảnh đầu vào
    output_video = "light_sweep_enhanced.mp4"  # Video đầu ra
    
    # Tùy chỉnh thông số
    CONFIG = {
        # Kích thước video
        "target_width": 448,      # Chiều rộng video
        "target_height": 782,     # Chiều cao video
        
        # Hiệu ứng ánh sáng
        "sweep_width": 200,        # Độ rộng vùng sáng (pixel)
        "intensity": 2.0,          # Cường độ sáng (1.0-3.0)
        
        # Thời gian và tốc độ
        "video_duration": 6.0,     # Thời lượng video (giây)
        "fps": 30,                 # Khung hình/giây
        "sweep_speed_multiplier": 0.8,  # Tốc độ quét (0.5=chậm, 1.0=bình thường, 2.0=nhanh)
    }
    
    try:
        create_light_sweep_video(
            input_path=input_image,
            output_path=output_video,
            **CONFIG
        )
        
        print("\n🎉 THÀNH CÔNG!")
        print(f"📺 Video: {output_video}")
        print(f"⏱️  Thời lượng: {CONFIG['video_duration']}s")
        print(f"📐 Kích thước: {CONFIG['target_width']}x{CONFIG['target_height']}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

# ====== CÁC VÍ DỤ PRESET ======

def create_quick_demo():
    """Tạo video demo nhanh (2 giây, 720p)"""
    create_light_sweep_video(
        input_path="phone.jpg",
        output_path="demo_quick.mp4",
        target_width=1280, target_height=720,
        video_duration=2.0,
        sweep_speed_multiplier=1.5
    )

def create_cinematic():
    """Tạo video cinematic chậm rãi (6 giây, 4K)"""
    create_light_sweep_video(
        input_path="phone.jpg", 
        output_path="cinematic.mp4",
        target_width=3840, target_height=2160,
        sweep_width=300, intensity=1.8,
        video_duration=6.0, fps=60,
        sweep_speed_multiplier=0.6
    )

def create_social_media():
    """Tạo video cho social media (vuông, 3 giây)"""
    create_light_sweep_video(
        input_path="phone.jpg",
        output_path="social_media.mp4", 
        target_width=1080, target_height=1080,
        video_duration=3.0,
        sweep_speed_multiplier=1.2
    )
