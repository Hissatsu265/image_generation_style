from moviepy.editor import VideoFileClip, CompositeVideoClip

def apply_zoom_out_effect(input_path, output_path, zoom_start_time, zoom_end_time):
    clip = VideoFileClip(input_path)
    w, h = clip.size

    duration = zoom_end_time - zoom_start_time

    def zoom_out_frame(get_frame, t):
        frame = get_frame(t)
        if zoom_start_time <= t <= zoom_end_time:
            progress = (t - zoom_start_time) / duration  # từ 0 đến 1
            zoom = 1.2 - 0.2 * progress  # ví dụ zoom từ 1.2x về 1.0x
            new_w = int(w / zoom)
            new_h = int(h / zoom)
            x1 = (w - new_w) // 2
            y1 = (h - new_h) // 2
            frame = frame[y1:y1+new_h, x1:x1+new_w]
        return frame

    # Áp dụng hiệu ứng bằng hàm lambda tùy biến frame
    zoomed_clip = clip.fl(lambda gf, t: zoom_out_frame(gf, t), apply_to=['video'])

    # Resize lại từng frame đã crop về kích thước ban đầu
    zoomed_clip = zoomed_clip.resize((w, h))

    zoomed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    print(f"✅ Video đã xuất với hiệu ứng zoom out: {output_path}")

# Ví dụ sử dụng
apply_zoom_out_effect(
    input_path="/content/output_video_with_audioddd.mp4",
    output_path="output_zoomout.mp4",
    zoom_start_time=0,   # Thời gian bắt đầu hiệu ứng
    zoom_end_time=1      # Thời gian kết thúc hiệu ứng
)
