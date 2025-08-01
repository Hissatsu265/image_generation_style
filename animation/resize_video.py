from moviepy.editor import VideoFileClip

def crop_and_resize_video(input_path, output_path, target_width, target_height):
    # Tính tỉ lệ cần cắt
    target_ratio = target_width / target_height

    clip = VideoFileClip(input_path)
    original_w, original_h = clip.w, clip.h
    original_ratio = original_w / original_h

    # Tính kích thước crop phù hợp
    if original_ratio > target_ratio:
        # Video quá rộng => crop hai bên
        new_w = int(original_h * target_ratio)
        x1 = (original_w - new_w) // 2
        y1 = 0
        x2 = x1 + new_w
        y2 = original_h
    else:
        # Video quá cao => crop trên dưới
        new_h = int(original_w / target_ratio)
        y1 = (original_h - new_h) // 2
        x1 = 0
        x2 = original_w
        y2 = y1 + new_h

    # Cắt rồi resize
    cropped = clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)
    resized = cropped.resize(newsize=(target_width, target_height))

    resized.write_videofile(output_path, codec='libx264', audio_codec='aac')
    print(f"✅ Xuất video hoàn tất: {output_path}")

# Ví dụ sử dụng
crop_and_resize_video(
    input_path="/content/output_video (8).mp4",
    output_path="output_cropped_resized.mp4",
    target_width=448,
    target_height=782
)
