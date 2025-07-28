import os
import torch
from PIL import Image
import sys
import cv2
import numpy as np
# sys.path.append('/workspace/multitalk_verquant/RealESRGAN')
current_dir = os.path.dirname(os.path.abspath(__file__))
realesrgan_path = os.path.join(current_dir, 'RealESRGAN')
sys.path.append(realesrgan_path)

from RealESRGAN import RealESRGAN
import time

def blend_images(original, upscaled, blend_ratio=0.2):
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
        model_path = f'RealESRGAN/weights/RealESRGAN_x{scale}.pth'

    model.load_weights(model_path, download=True)

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

if __name__ == '__main__':
    print("=" * 60)
    print("🖼️ REALESRGAN IMAGE UPSCALER")
    print("=" * 60)

    image_path = input("📁 Nhập đường dẫn ảnh (hoặc Enter để dùng mặc định): ").strip()

    if not image_path:
        image_path = "/content/drive/MyDrive/20 6 upscale video/blurimg.png"

    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại: {image_path}")
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

        upscale_image_enhanced(image_path, model_type=model_type,
                             blend_ratio=blend_ratio, sharpen=sharpen)