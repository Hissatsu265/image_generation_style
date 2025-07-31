from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
import cv2
import os

def create_product_showcase(video_path, product_media_path, output_path, scale_ratio=0.25, position="bottom_left"):
    """
    Tạo video showcase với product media (ảnh hoặc video) làm nền và video chính ở góc
    
    Args:
        video_path: đường dẫn video chính
        product_media_path: đường dẫn ảnh hoặc video sản phẩm
        output_path: đường dẫn file đầu ra
        scale_ratio: tỷ lệ thu nhỏ video chính (mặc định 0.25 = 25%)
        position: vị trí video nhỏ ("bottom_left", "bottom_right", "top_left", "top_right")
    """
    
    # === BƯỚC 1: Load video chính để lấy tỷ lệ khung hình ===
    main_video = VideoFileClip(video_path)
    video_ratio = main_video.w / main_video.h
    video_duration = main_video.duration
    
    # === BƯỚC 2: Kiểm tra product media là ảnh hay video ===
    file_extension = os.path.splitext(product_media_path)[1].lower()
    is_video = file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    if is_video:
        # Xử lý video sản phẩm
        product_video = VideoFileClip(product_media_path)
        
        # === BƯỚC 3: Crop video sản phẩm theo tỷ lệ video chính ===
        product_ratio = product_video.w / product_video.h
        
        if product_ratio > video_ratio:
            # Video sản phẩm quá rộng → cắt chiều ngang
            new_w = int(product_video.h * video_ratio)
            x1 = (product_video.w - new_w) // 2
            product_cropped = product_video.crop(x1=x1, x2=x1+new_w)
        else:
            # Video sản phẩm quá cao → cắt chiều dọc  
            new_h = int(product_video.w / video_ratio)
            y1 = (product_video.h - new_h) // 2
            product_cropped = product_video.crop(y1=y1, y2=y1+new_h)
        
        # === BƯỚC 4: Xử lý thời lượng video sản phẩm ===
        if product_cropped.duration < video_duration:
            # Video sản phẩm ngắn hơn → lặp lại
            loops_needed = int(video_duration / product_cropped.duration) + 1
            background_clip = product_cropped.loop(duration=video_duration)
        else:
            # Video sản phẩm dài hơn → cắt
            background_clip = product_cropped.subclip(0, video_duration)
            
        # Resize background về kích thước video chính
        background_clip = background_clip.resize((main_video.w, main_video.h))
        
    else:
        # Xử lý ảnh sản phẩm (code gốc)
        image = cv2.imread(product_media_path)
        img_h, img_w = image.shape[:2]
        img_ratio = img_w / img_h

        if img_ratio > video_ratio:
            # Ảnh quá rộng → cắt chiều ngang
            new_w = int(img_h * video_ratio)
            x1 = (img_w - new_w) // 2
            image_cropped = image[:, x1:x1+new_w]
        else:
            # Ảnh quá cao → cắt chiều dọc
            new_h = int(img_w / video_ratio)
            y1 = (img_h - new_h) // 2
            image_cropped = image[y1:y1+new_h, :]

        # Lưu ảnh tạm thời
        temp_img_path = "temp_cropped.jpg" 
        cv2.imwrite(temp_img_path, image_cropped)
        
        # Tạo background clip từ ảnh
        background_clip = ImageClip(temp_img_path).set_duration(video_duration)
        background_clip = background_clip.resize((main_video.w, main_video.h))

    # === BƯỚC 5: Thu nhỏ video chính và đặt vị trí ===
    small_video = main_video.resize(scale_ratio)
    
    # Tính toán vị trí dựa trên tham số position
    margin = 20
    if position == "bottom_left":
        pos_x = margin
        pos_y = background_clip.h - small_video.h - margin
    elif position == "bottom_right":
        pos_x = background_clip.w - small_video.w - margin
        pos_y = background_clip.h - small_video.h - margin
    elif position == "top_left":
        pos_x = margin
        pos_y = margin
    elif position == "top_right":
        pos_x = background_clip.w - small_video.w - margin
        pos_y = margin
    else:
        # Mặc định bottom_left
        pos_x = margin
        pos_y = background_clip.h - small_video.h - margin
    
    small_video = small_video.set_position((pos_x, pos_y))

    # === BƯỚC 6: Ghép video cuối cùng ===
    final = CompositeVideoClip([background_clip, small_video])
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    # Dọn dẹp file tạm
    if not is_video and os.path.exists("temp_cropped.jpg"):
        os.remove("temp_cropped.jpg")
    
    # Đóng các clip để giải phóng bộ nhớ
    main_video.close()
    if is_video:
        product_video.close()
        product_cropped.close()
    background_clip.close()
    small_video.close()
    final.close()
    
    print(f"✅ Đã tạo video showcase: {output_path}")

# === SỬ DỤNG ===
# if __name__ == "__main__":
#     video_path = "/content/merged_video (4).mp4"
#     product_media_path = "/content/phone.jpg"  
#     output_path = "output_product_showcase.mp4"
    
#     create_product_showcase(
#         video_path=video_path,
#         product_media_path=product_media_path, 
#         output_path=output_path,
#         scale_ratio=0.3,  
#         position="bottom_left" 
#     )
