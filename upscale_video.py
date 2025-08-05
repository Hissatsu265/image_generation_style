import os
import torch
from PIL import Image
import sys
import cv2
import subprocess
import shutil
import glob
import numpy as np
# sys.path.append('/content/drive/MyDrive/20_6upscale_video/RealESRGAN-20250620T093117Z-1-001/RealESRGAN')
current_dir = os.path.dirname(os.path.abspath(__file__))
realesrgan_path = os.path.join(current_dir, 'RealESRGAN')
sys.path.append(realesrgan_path)
from RealESRGAN import RealESRGAN
import time

def blend_images(original, upscaled, blend_ratio=0.2):
    """Trộn ảnh gốc với ảnh upscaled để giảm hiệu ứng hoạt hình"""
    # Resize ảnh gốc lên cùng kích thước với ảnh upscaled
    original_resized = original.resize(upscaled.size, Image.LANCZOS)

    # Convert sang numpy để xử lý
    orig_np = np.array(original_resized, dtype=np.float32)
    upsc_np = np.array(upscaled, dtype=np.float32)

    # Trộn theo tỷ lệ
    blended = (1 - blend_ratio) * upsc_np + blend_ratio * orig_np
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)

def apply_sharpening(image, strength=0.3):
    """Áp dụng sharpening nhẹ để tăng độ sắc nét tự nhiên"""
    img_np = np.array(image)

    # Gaussian blur
    blurred = cv2.GaussianBlur(img_np, (3, 3), 1.0)

    # Unsharp masking
    sharpened = cv2.addWeighted(img_np, 1 + strength, blurred, -strength, 0)

    return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))

def upscale_image_enhanced(input_path: str, output_path: str = "results", scale: int = 4,
                          model_type: str = "realesr-general", blend_ratio: float = 0.0,
                          sharpen: bool = False):
    """Upscale một ảnh đơn lẻ với các tùy chọn cải tiến"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RealESRGAN(device, scale=scale)

    # Chọn model phù hợp
    if model_type == "realesr-general":
        model_path = f'/content/drive/MyDrive/20_6upscale_video/RealESRGAN-20250620T093117Z-1-001/RealESRGAN/weights/realesr-general-x4v3.pth'
    else:
        model_path = f'/workspace/multitalk_verquant/RealESRGAN/RealESRGAN/weights/RealESRGAN_x2.pth'

    model.load_weights(model_path, download=False)

    os.makedirs(output_path, exist_ok=True)
    image_name = os.path.basename(input_path)
    print(f"\n🚀 Đang xử lý: {image_name}")
    print(f"🎨 Model: {model_type}")
    print(f"🔀 Blend ratio: {blend_ratio}")
    print(f"✨ Sharpening: {'Có' if sharpen else 'Không'}")

    original_image = Image.open(input_path).convert('RGB')

    start = time.time()
    sr_image = model.predict(original_image)

    if blend_ratio > 0:
        sr_image = blend_images(original_image, sr_image, blend_ratio)
        print(f"🔀 Đã áp dụng blending với tỷ lệ {blend_ratio}")

    if sharpen:
        sr_image = apply_sharpening(sr_image)
        print(f"✨ Đã áp dụng sharpening")

    elapsed = time.time() - start
    save_path = os.path.join(output_path, f"upscaled_{image_name}")
    sr_image.save(save_path)

    print(f"✅ Đã upscale xong ảnh: {image_name}")
    print(f"⏱️ Thời gian xử lý: {elapsed:.2f} giây")
    print(f"💾 Đã lưu ảnh tại: {save_path}")

    return save_path

def get_video_info(video_path: str):
    """Lấy thông tin video (fps, duration, etc.)"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }

def extract_frames(video_path: str, output_dir: str):
    """Tách video thành các frame bằng ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'ffmpeg', '-i', video_path,
        '-q:v', '1',
        '-pix_fmt', 'rgb24',
        os.path.join(output_dir, 'frame_%06d.png'),
        '-y'
    ]

    print(f"🎬 Đang tách video thành frames...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Lỗi khi tách frames: {result.stderr}")
        return False

    frame_files = glob.glob(os.path.join(output_dir, 'frame_*.png'))
    print(f"✅ Đã tách được {len(frame_files)} frames")
    return True

def upscale_frames_enhanced(frames_dir: str, output_dir: str, scale: int = 4,
                           model_type: str = "realesr-general", blend_ratio: float = 0.0,
                           sharpen: bool = False):
    """Upscale tất cả các frame với các tùy chọn cải tiến"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Sử dụng device: {device}")
    print(f"🎨 Model: {model_type}")
    print(f"🔀 Blend ratio: {blend_ratio}")
    print(f"✨ Sharpening: {'Có' if sharpen else 'Không'}")

    model = RealESRGAN(device, scale=scale)

    if model_type == "realesr-general":
        model_path = f'/content/drive/MyDrive/20_6upscale_video/RealESRGAN-20250620T093117Z-1-001/RealESRGAN/weights/realesr-general-x4v3.pth'
    else:
        model_path = f'/workspace/multitalk_verquant/RealESRGAN/RealESRGAN/weights/RealESRGAN_x2.pth'

    model.load_weights(model_path, download=False)

    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
    total_frames = len(frame_files)

    if total_frames == 0:
        print("❌ Không tìm thấy frame nào để xử lý!")
        return False

    print(f"🚀 Bắt đầu upscale {total_frames} frames...")
    start_time = time.time()

    for i, frame_path in enumerate(frame_files, 1):
        frame_start = time.time()

        original_image = Image.open(frame_path).convert('RGB')
        sr_image = model.predict(original_image)

        # Áp dụng các kỹ thuật cải tiến
        if blend_ratio > 0:
            sr_image = blend_images(original_image, sr_image, blend_ratio)

        if sharpen:
            sr_image = apply_sharpening(sr_image)

        frame_name = os.path.basename(frame_path)
        output_path = os.path.join(output_dir, frame_name)
        sr_image.save(output_path)

        frame_elapsed = time.time() - frame_start
        total_elapsed = time.time() - start_time
        avg_time = total_elapsed / i
        eta = avg_time * (total_frames - i)

        print(f"✅ Frame {i}/{total_frames} - {frame_elapsed:.2f}s - ETA: {eta:.1f}s")

    total_time = time.time() - start_time
    print(f"🎉 Hoàn thành upscale tất cả frames trong {total_time:.2f} giây")
    return True

def create_video_from_frames(frames_dir: str, output_video_path: str, fps: float, original_video_path: str = None):
    """Ghép các frame thành video và copy audio từ video gốc"""

    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        '-preset', 'medium',
        output_video_path + '_no_audio.mp4',
        '-y'
    ]

    print(f"🎬 Đang tạo video từ frames...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Lỗi khi tạo video: {result.stderr}")
        return False

    if original_video_path and os.path.exists(original_video_path):
        print(f"🔊 Đang copy audio từ video gốc...")
        cmd_audio = [
            'ffmpeg',
            '-i', output_video_path + '_no_audio.mp4',
            '-i', original_video_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_video_path,
            '-y'
        ]

        result = subprocess.run(cmd_audio, capture_output=True, text=True)

        if result.returncode == 0:
            os.remove(output_video_path + '_no_audio.mp4')
            print(f"✅ Đã thêm audio vào video")
        else:
            shutil.move(output_video_path + '_no_audio.mp4', output_video_path)
            print(f"⚠️ Không thể copy audio, video chỉ có hình ảnh")
    else:
        shutil.move(output_video_path + '_no_audio.mp4', output_video_path)

    print(f"✅ Video đã được tạo: {output_video_path}")
    return True

def upscale_video_enhanced(video_path: str, output_path: str = None, scale: int = 2,
                          model_type: str = "realesr-general", blend_ratio: float = 0.0,
                          sharpen: bool = False, keep_temp_files: bool = False):
    """Upscale toàn bộ video với các tùy chọn cải tiến"""

    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy file video: {video_path}")
        return

    if output_path is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        suffix = f"_{model_type}" if model_type != "realesr-general" else ""
        output_path = os.path.join(video_dir, f"{video_name}_upscaled_x{scale}{suffix}.mp4")

    temp_dir = "temp_video_processing"
    frames_dir = os.path.join(temp_dir, "original_frames")
    upscaled_frames_dir = os.path.join(temp_dir, "upscaled_frames")

    try:
        print(f"📹 Đang phân tích video: {os.path.basename(video_path)}")
        video_info = get_video_info(video_path)
        print(f"📊 Thông tin video:")
        print(f"   - Độ phân giải: {video_info['width']}x{video_info['height']}")
        print(f"   - FPS: {video_info['fps']:.2f}")
        print(f"   - Số frame: {video_info['frame_count']}")
        print(f"   - Thời lượng: {video_info['duration']:.2f} giây")
        print(f"   - Độ phân giải sau upscale: {video_info['width']*scale}x{video_info['height']*scale}")

        if not extract_frames(video_path, frames_dir):
            return

        if not upscale_frames_enhanced(frames_dir, upscaled_frames_dir, scale,
                                      model_type, blend_ratio, sharpen):
            return

        if not create_video_from_frames(upscaled_frames_dir, output_path, video_info['fps'], video_path):
            return

        print(f"\n🎉 HOÀN THÀNH!")
        print(f"📁 Video gốc: {video_path}")
        print(f"💾 Video đã upscale: {output_path}")
        print(f"📏 Scale: x{scale}")
        print(f"🎨 Model: {model_type}")

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {str(e)}")

    finally:
        if not keep_temp_files and os.path.exists(temp_dir):
            print(f"🧹 Đang dọn dẹp files tạm...")
            shutil.rmtree(temp_dir)
            print(f"✅ Đã xóa files tạm")

if __name__ == '__main__':
    print("=" * 60)
    print("🎥 REALESRGAN VIDEO UPSCALER - ENHANCED VERSION")
    print("=" * 60)

    video_path = input("📁 Nhập đường dẫn video (hoặc Enter để dùng mặc định): ").strip()

    if not video_path:
        video_path = "/content/drive/MyDrive/20 6 upscale video/single_long_mediumvram_8step.mp4"

    if not os.path.exists(video_path):
        print(f"❌ File không tồn tại: {video_path}")
        print("Bạn có muốn upscale ảnh thay thế không? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            image_path = "/content/drive/MyDrive/20 6 upscale video/blurimg.png"

            # Tùy chọn cho ảnh
            print("\n🎨 Chọn model:")
            print("1. realesr-general (tự nhiên hơn, ít hoạt hình)")
            print("2. RealESRGAN (sắc nét hơn, có thể hoạt hình)")
            model_choice = input("Lựa chọn (1/2) [mặc định: 1]: ").strip()
            model_type = "realesr-general" if model_choice != "2" else "realesrgan"

            blend_input = input("🔀 Blend ratio (0.0-0.5, càng cao càng tự nhiên) [mặc định: 0.1]: ").strip()
            blend_ratio = 0.1
            try:
                if blend_input:
                    blend_ratio = max(0.0, min(0.5, float(blend_input)))
            except:
                pass

            sharpen_choice = input("✨ Áp dụng sharpening? (y/n) [mặc định: n]: ").strip().lower()
            sharpen = sharpen_choice == 'y'

            upscale_image_enhanced(image_path, model_type=model_type,
                                 blend_ratio=blend_ratio, sharpen=sharpen)
    else:
        scale_input = input("📏 Nhập scale factor (2, 4, 8) [mặc định: 4]: ").strip()
        scale = 4
        if scale_input in ['2', '4', '8']:
            scale = int(scale_input)

        print("\n🎨 Chọn model:")
        print("1. realesr-general (tự nhiên hơn, ít bị hoạt hình hóa)")
        print("2. RealESRGAN (sắc nét hơn, có thể bị hoạt hình hóa)")
        model_choice = input("Lựa chọn (1/2) [mặc định: 1]: ").strip()
        model_type = "realesr-general" if model_choice != "2" else "realesrgan"

        blend_input = input("🔀 Blend ratio (0.0-0.5, càng cao càng tự nhiên) [mặc định: 0.1]: ").strip()
        blend_ratio = 0.1
        try:
            if blend_input:
                blend_ratio = max(0.0, min(0.5, float(blend_input)))
        except ValueError:
            pass

        sharpen_choice = input("✨ Áp dụng sharpening? (y/n) [mặc định: n]: ").strip().lower()
        sharpen = sharpen_choice == 'y'

        print(f"\n🚀 Bắt đầu xử lý với các tham số:")
        print(f"   - Model: {model_type}")
        print(f"   - Scale: x{scale}")
        print(f"   - Blend ratio: {blend_ratio}")
        print(f"   - Sharpening: {'Có' if sharpen else 'Không'}")

        upscale_video_enhanced(video_path, scale=scale, model_type=model_type,
                              blend_ratio=blend_ratio, sharpen=sharpen)