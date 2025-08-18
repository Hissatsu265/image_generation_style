# from moviepy.editor import VideoFileClip
# import numpy as np

# # Load video gốc
# clip = VideoFileClip("/content/clip_1.mp4")

# # === Cấu hình zoom ===
# zoom_duration = 0.75              # Thời gian hiệu ứng zoom (giây)
# zoom_start_time = 1.0             # Thời điểm bắt đầu zoom (giây)
# zoom_percent = 0.5                # Zoom vào 30% kích thước video (0.3 = 30%)

# # Kích thước video gốc và tỷ lệ
# W, H = clip.size
# aspect_ratio = W / H

# # Tâm điểm vùng cần zoom (có thể thay đổi theo ý bạn)
# cx, cy = 200, 500

# # Kích thước vùng zoom nhỏ nhất (sau khi zoom xong)
# min_zoom_w = int(W * zoom_percent)
# min_zoom_h = int(min_zoom_w / aspect_ratio)

# # Hàm xác định vùng crop theo thời gian
# def dynamic_crop(t):
#     if t < zoom_start_time:
#         # Trước khi bắt đầu zoom: hiển thị toàn khung
#         return 0, 0, W, H

#     elif t >= zoom_start_time + zoom_duration:
#         # Sau khi zoom xong: giữ vùng zoom cố định
#         crop_w = min_zoom_w
#         crop_h = min_zoom_h
#     else:
#         # Trong quá trình zoom: scale từ 1.0 đến target zoom
#         alpha = (t - zoom_start_time) / zoom_duration
#         crop_w = int(W - alpha * (W - min_zoom_w))
#         crop_h = int(H - alpha * (H - min_zoom_h))

#         # Đảm bảo đúng tỉ lệ
#         crop_h = int(crop_w / aspect_ratio)

#     # Tính vị trí crop từ tâm điểm
#     x1 = max(0, cx - crop_w // 2)
#     y1 = max(0, cy - crop_h // 2)
#     x2 = min(W, x1 + crop_w)
#     y2 = min(H, y1 + crop_h)

#     # Cập nhật lại để đảm bảo đúng kích thước
#     x1 = x2 - crop_w
#     y1 = y2 - crop_h

#     return x1, y1, x2, y2

# # Áp dụng crop động từng frame
# zoomed = clip.fl(lambda gf, t: 
#     clip.crop(*dynamic_crop(t))
#         .resize((W, H))
#         .get_frame(t)
# )

# # Ghi kết quả ra file
# zoomed.set_duration(clip.duration).write_videofile("output_zoomed.mp4", codec="libx264")
from moviepy.editor import VideoFileClip

def apply_zoom_effect(
    input_path,
    output_path="output_zoomed.mp4",
    zoom_duration=0.75,
    zoom_start_time=1.0,
    zoom_percent=0.5,
    center=(200, 500),
    end_effect=None,
    remove_mode="instant"  # "instant" hoặc "smooth"
):
    """
    Áp dụng hiệu ứng zoom động cho video.
    """
    clip = VideoFileClip(input_path)

    # Kích thước video gốc và tỷ lệ
    W, H = clip.size
    duration_video = clip.duration
    cx, cy = center

    # ===== Validation =====
    if zoom_duration <= 0:
        print("[LỖI] zoom_duration phải > 0")
        return clip.write_videofile(output_path, codec="libx264")

    if zoom_start_time < 0 or zoom_start_time >= duration_video:
        print("[LỖI] zoom_start_time không hợp lệ (ngoài khoảng video)")
        return clip.write_videofile(output_path, codec="libx264")

    if end_effect is not None:
        if end_effect <= zoom_start_time:
            print("[LỖI] end_effect phải lớn hơn zoom_start_time")
            return clip.write_videofile(output_path, codec="libx264")
        if end_effect > duration_video:
            print("[LỖI] end_effect vượt quá độ dài video")
            return clip.write_videofile(output_path, codec="libx264")

    if not (0 < zoom_percent <= 1):
        print("[LỖI] zoom_percent phải nằm trong (0, 1] (ví dụ: 0.5 = 50%)")
        return clip.write_videofile(output_path, codec="libx264")

    if not (0 <= cx <= W and 0 <= cy <= H):
        print("[LỖI] center nằm ngoài khung hình")
        return clip.write_videofile(output_path, codec="libx264")

    if remove_mode not in ["instant", "smooth"]:
        print("[LỖI] remove_mode chỉ được là 'instant' hoặc 'smooth'")
        return clip.write_videofile(output_path, codec="libx264")

    aspect_ratio = W / H
    min_zoom_w = int(W * zoom_percent)
    min_zoom_h = int(min_zoom_w / aspect_ratio)

    # Hàm xác định vùng crop theo thời gian
    def dynamic_crop(t):
        if t < zoom_start_time:
            return 0, 0, W, H

        # Nếu có end_effect
        if end_effect is not None and t >= end_effect:
            if remove_mode == "instant":
                return 0, 0, W, H
            elif remove_mode == "smooth":
                if t >= end_effect + zoom_duration:
                    return 0, 0, W, H
                else:
                    alpha = (t - end_effect) / zoom_duration
                    crop_w = int(min_zoom_w + alpha * (W - min_zoom_w))
                    crop_h = int(crop_w / aspect_ratio)
        elif t >= zoom_start_time + zoom_duration:
            crop_w, crop_h = min_zoom_w, min_zoom_h
        else:
            alpha = (t - zoom_start_time) / zoom_duration
            crop_w = int(W - alpha * (W - min_zoom_w))
            crop_h = int(crop_w / aspect_ratio)

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(W, x1 + crop_w)
        y2 = min(H, y1 + crop_h)

        x1 = x2 - crop_w
        y1 = y2 - crop_h
        return x1, y1, x2, y2

    # Áp dụng crop động
    zoomed = clip.fl(lambda gf, t: 
        clip.crop(*dynamic_crop(t))
            .resize((W, H))
            .get_frame(t)
    )

    zoomed.set_duration(clip.duration).write_videofile(output_path, codec="libx264")


# apply_zoom_effect(
#     input_path="/content/55c95f56_clip_0_cut_11.49s.mp4",
#     output_path="zoomed.mp4",
#     zoom_duration=1.5,
#     zoom_start_time=2.0,
#     zoom_percent=0.4,
#     center=(300, 400),
#     end_effect=5.0,
#     remove_mode="smooth"
# )

# # Zoom in lúc 2s, giữ đến 5s, rồi bỏ zoom ngay lập tức
# apply_zoom_effect(
#     input_path="/content/55c95f56_clip_0_cut_11.49s.mp4",
#     output_path="zoomed_instant.mp4",
#     zoom_duration=1.5,
#     zoom_start_time=2.0,
#     zoom_percent=0.4,
#     center=(300, 400),
#     end_effect=5.0,
#     remove_mode="instant"
# )
