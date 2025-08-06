import gradio as gr
import os
import json
from moviepy.editor import VideoFileClip
# ===================================================================
import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips, ImageSequenceClip
from moviepy.video.fx import resize, speedx
import tempfile
import shutil

class VideoTransitionTool:
    def __init__(self):
        self.transition_duration = 1.0  # Default 1 second transition

    def merge_videos_with_transition(self, video1_path, video2_path, output_path,
                                   transition_type="crossfade", transition_duration=1.0):
        """
        Merge two videos with specified transition effect

        Args:
            video1_path: Path to first video
            video2_path: Path to second video
            output_path: Output video path
            transition_type: Type of transition effect
            transition_duration: Duration of transition in seconds
        """
        self.transition_duration = transition_duration

        # Load videos
        clip1 = VideoFileClip(video1_path)
        clip2 = VideoFileClip(video2_path)

        print(f"Video 1: {clip1.size}, Duration: {clip1.duration}s")
        print(f"Video 2: {clip2.size}, Duration: {clip2.duration}s")

        # Apply transition based on type
        if transition_type == "crossfade":
            result = self._crossfade_transition(clip1, clip2)
        elif transition_type == "slide_horizontal":
            result = self._slide_transition(clip1, clip2, direction="horizontal")
        elif transition_type == "slide_vertical":
            result = self._slide_transition(clip1, clip2, direction="vertical")
        elif transition_type == "zoom_in":
            result = self._zoom_transition(clip1, clip2, zoom_type="in")
        elif transition_type == "zoom_out":
            result = self._zoom_transition(clip1, clip2, zoom_type="out")
        elif transition_type == "flash_cut":
            result = self._flash_cut_transition(clip1, clip2)
        elif transition_type == "push_blur":
            result = self._push_blur_transition(clip1, clip2)
        elif transition_type == "rgb_split":
            result = self._rgb_split_transition(clip1, clip2)
        elif transition_type == "circle_mask":
            result = self._mask_transition(clip1, clip2, mask_type="circle")
        elif transition_type == "square_mask":
            result = self._mask_transition(clip1, clip2, mask_type="square")
        else:
            # Default to simple concatenation
            result = concatenate_videoclips([clip1, clip2])

        # Export result
        print(f"Exporting video with {transition_type} transition...")
        result.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # Clean up
        clip1.close()
        clip2.close()
        result.close()

        print(f"Video saved to: {output_path}")

    def _crossfade_transition(self, clip1, clip2):
        """Crossfade transition effect"""
        # Trim clips for transition
        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip1_fade = clip1.subclip(clip1.duration - self.transition_duration)
        clip2_fade = clip2.subclip(0, self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)

        # Create fade effects
        clip1_fade = clip1_fade.fadeout(self.transition_duration)
        clip2_fade = clip2_fade.fadein(self.transition_duration)

        # Composite the fading parts
        transition_part = CompositeVideoClip([clip1_fade, clip2_fade])

        return concatenate_videoclips([clip1_main, transition_part, clip2_main])

    def _slide_transition(self, clip1, clip2, direction="horizontal"):
        """Slide transition effect"""
        w, h = clip1.size
        fps = clip1.fps

        # Create frames for transition
        transition_frames = []
        steps = int(fps * self.transition_duration)

        # Get last frame of clip1 and first frame of clip2
        last_frame = clip1.get_frame(clip1.duration - 0.1)
        first_frame = clip2.get_frame(0.1)

        for i in range(steps):
            progress = i / steps

            if direction == "horizontal":
                # Slide horizontally
                offset = int(w * progress)
                frame = np.zeros((h, w, 3), dtype=np.uint8)

                # Clip1 sliding out (left)
                if offset < w:
                    clip1_part = last_frame[:, offset:]
                    frame[:, :w-offset] = clip1_part

                # Clip2 sliding in (right)
                if offset > 0:
                    clip2_part = first_frame[:, :offset]
                    frame[:, w-offset:] = clip2_part

            else:  # vertical
                # Slide vertically
                offset = int(h * progress)
                frame = np.zeros((h, w, 3), dtype=np.uint8)

                # Clip1 sliding out (up)
                if offset < h:
                    clip1_part = last_frame[offset:, :]
                    frame[:h-offset, :] = clip1_part

                # Clip2 sliding in (down)
                if offset > 0:
                    clip2_part = first_frame[:offset, :]
                    frame[h-offset:, :] = clip2_part

            transition_frames.append(frame)

        # Create transition clip
        transition_clip = ImageSequenceClip(transition_frames, fps=fps)

        # Combine clips
        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)

        return concatenate_videoclips([clip1_main, transition_clip, clip2_main])

    def _zoom_transition(self, clip1, clip2, zoom_type="in"):
        """Zoom in/out transition effect"""
        fps = clip1.fps
        steps = int(fps * self.transition_duration)
        transition_frames = []

        last_frame = clip1.get_frame(clip1.duration - 0.1)
        first_frame = clip2.get_frame(0.1)

        for i in range(steps):
            progress = i / steps

            if zoom_type == "in":
                # Zoom into clip1, then reveal clip2
                if progress < 0.5:
                    # Zoom into clip1
                    scale = 1 + progress * 2  # Scale from 1 to 2
                    frame = self._zoom_frame(last_frame, scale)
                else:
                    # Zoom out from clip2
                    scale = 2 - (progress - 0.5) * 2  # Scale from 2 to 1
                    frame = self._zoom_frame(first_frame, scale)
            else:  # zoom_out
                # Zoom out from clip1, then zoom into clip2
                if progress < 0.5:
                    scale = 1 - progress * 0.8  # Scale from 1 to 0.2
                    frame = self._zoom_frame(last_frame, scale)
                else:
                    scale = 0.2 + (progress - 0.5) * 1.6  # Scale from 0.2 to 1
                    frame = self._zoom_frame(first_frame, scale)

            transition_frames.append(frame)

        transition_clip = ImageSequenceClip(transition_frames, fps=fps)

        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)

        return concatenate_videoclips([clip1_main, transition_clip, clip2_main])

    def _zoom_frame(self, frame, scale):
        """Helper function to zoom a frame"""
        h, w = frame.shape[:2]

        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))

        # Create output frame
        result = np.zeros((h, w, 3), dtype=np.uint8)

        # Calculate crop/pad positions
        if scale > 1:  # Zoom in - crop
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            result = resized[start_y:start_y+h, start_x:start_x+w]
        else:  # Zoom out - pad
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            result[start_y:start_y+new_h, start_x:start_x+new_w] = resized

        return result

    def _flash_cut_transition(self, clip1, clip2):
        """Flash cut transition with white/black flash"""
        fps = clip1.fps
        flash_frames = max(1, int(fps * 0.1))  # 0.1 second flash

        # Create white flash frames
        h, w = clip1.size[1], clip1.size[0]  # MoviePy uses (width, height)
        white_frame = np.ones((h, w, 3), dtype=np.uint8) * 255
        flash_frames_list = [white_frame] * flash_frames

        flash_clip = ImageSequenceClip(flash_frames_list, fps=fps)

        # Simple concatenation with flash
        clip1_main = clip1.subclip(0, clip1.duration)
        clip2_main = clip2.subclip(0)

        return concatenate_videoclips([clip1_main, flash_clip, clip2_main])

    def _push_blur_transition(self, clip1, clip2):
        """Push transition with blur effect"""
        w, h = clip1.size
        fps = clip1.fps
        steps = int(fps * self.transition_duration)
        transition_frames = []

        last_frame = clip1.get_frame(clip1.duration - 0.1)
        first_frame = clip2.get_frame(0.1)

        for i in range(steps):
            progress = i / steps
            offset = int(w * progress)
            blur_strength = int(15 * (1 - abs(progress - 0.5) * 2))  # Max blur at middle

            # Create composite frame
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            # Clip1 part (with blur)
            if offset < w:
                clip1_part = last_frame[:, offset:]
                if blur_strength > 0:
                    clip1_part = cv2.GaussianBlur(clip1_part, (blur_strength*2+1, blur_strength*2+1), 0)
                frame[:, :w-offset] = clip1_part

            # Clip2 part (with blur)
            if offset > 0:
                clip2_part = first_frame[:, :offset]
                if blur_strength > 0:
                    clip2_part = cv2.GaussianBlur(clip2_part, (blur_strength*2+1, blur_strength*2+1), 0)
                frame[:, w-offset:] = clip2_part

            transition_frames.append(frame)

        transition_clip = ImageSequenceClip(transition_frames, fps=fps)

        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)

        return concatenate_videoclips([clip1_main, transition_clip, clip2_main])

    def _rgb_split_transition(self, clip1, clip2):
        """RGB split glitch transition"""
        w, h = clip1.size
        fps = clip1.fps
        steps = int(fps * self.transition_duration)
        transition_frames = []

        last_frame = clip1.get_frame(clip1.duration - 0.1)
        first_frame = clip2.get_frame(0.1)

        for i in range(steps):
            progress = i / steps

            # Create RGB split effect
            if progress < 0.5:
                # Split clip1
                base_frame = last_frame.copy()
                split_intensity = int(20 * progress)
            else:
                # Split clip2
                base_frame = first_frame.copy()
                split_intensity = int(20 * (1 - progress))

            # Apply RGB split
            if split_intensity > 0:
                # Separate RGB channels
                b, g, r = cv2.split(base_frame)

                # Shift channels
                r_shifted = np.roll(r, split_intensity, axis=1)
                b_shifted = np.roll(b, -split_intensity, axis=1)

                # Merge back
                frame = cv2.merge([b_shifted, g, r_shifted])
            else:
                frame = base_frame

            # Add color flash at peak
            if 0.4 < progress < 0.6:
                flash_intensity = 1 - abs(progress - 0.5) * 4
                if progress < 0.5:
                    # Red flash
                    frame[:, :, 2] = np.clip(frame[:, :, 2] + flash_intensity * 100, 0, 255)
                else:
                    # Blue flash
                    frame[:, :, 0] = np.clip(frame[:, :, 0] + flash_intensity * 100, 0, 255)

            transition_frames.append(frame)

        transition_clip = ImageSequenceClip(transition_frames, fps=fps)

        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)

        return concatenate_videoclips([clip1_main, transition_clip, clip2_main])

    def _mask_transition(self, clip1, clip2, mask_type="circle"):
        """Mask-based transition with creative shapes"""
        w, h = clip1.size
        fps = clip1.fps
        steps = int(fps * self.transition_duration)
        transition_frames = []

        last_frame = clip1.get_frame(clip1.duration - 0.1)
        first_frame = clip2.get_frame(0.1)

        center_x, center_y = w // 2, h // 2
        max_radius = int(np.sqrt(w**2 + h**2) / 2)

        for i in range(steps):
            progress = i / steps

            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)

            if mask_type == "circle":
                radius = int(max_radius * progress)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)

            elif mask_type == "square":
                size = int(min(w, h) * progress)
                x1 = center_x - size // 2
                y1 = center_y - size // 2
                x2 = center_x + size // 2
                y2 = center_y + size // 2
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            # Apply mask
            mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0

            # Blend frames
            frame = last_frame * (1 - mask_3d) + first_frame * mask_3d
            frame = frame.astype(np.uint8)

            transition_frames.append(frame)

        transition_clip = ImageSequenceClip(transition_frames, fps=fps)

        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)

        return concatenate_videoclips([clip1_main, transition_clip, clip2_main])


# ===================================================================
# Global variables để lưu trữ thông tin
current_clips = []
clip_settings = {}
video_queue = []  # Hàng chờ các video đã chỉnh sửa
current_preview_video = None  # Video hiện tại đang preview
transition_settings = {}  # Lưu cài đặt chuyển cảnh giữa các clip

from moviepy.editor import VideoFileClip

def split_video_by_times(video_path, time_list, output_prefix="clip"):

    video = VideoFileClip(video_path)

    # Lấy độ dài video
    video_duration = video.duration

    # Tự động thêm thời điểm bắt đầu (0) và kết thúc (độ dài video)
    full_time_list = [0] + sorted(time_list) + [video_duration]

    # Loại bỏ các thời điểm trùng lặp
    full_time_list = sorted(list(set(full_time_list)))

    print(f"Độ dài video: {video_duration:.2f} giây")
    print(f"Danh sách thời điểm cắt: {full_time_list}")



    # Lặp qua từng cặp thời điểm để cắt
    for i in range(len(full_time_list) - 1):
        start = full_time_list[i]
        end = full_time_list[i + 1]

        if start >= end:
            print(f"Bỏ qua đoạn không hợp lệ: {start} >= {end}")
            continue

        # Cắt đoạn video
        subclip = video.subclip(start, end)

        # Tạo tên file và lưu lại
        output_path = f"{output_prefix}_{i+1}.mp4"
        subclip.write_videofile(output_path, codec="libx264")
        current_clips.append(output_path)
        print(f"Đã lưu đoạn {i+1}: {start:.2f} -> {end:.2f} giây vào {output_path}")

    video.close()
    print(f"Hoàn thành! Đã cắt thành {len(full_time_list) - 1} đoạn video.")
    return current_clips

def process_zoom_effect(video_path, x, y, start_time, duration, zoom_level):
    """
    Hàm xử lý hiệu ứng zoom vào video

    Args:
        video_path: Đường dẫn video đầu vào
        x, y: Tọa độ tâm zoom
        start_time: Thời điểm bắt đầu zoom (giây)
        duration: Thời gian kéo dài zoom (giây)
        zoom_level: Mức độ zoom (0.3 = 30%)

    Returns:
        str: Đường dẫn video sau khi zoom
    """
    import numpy as np

    # Tạo tên file output
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{base_name}_zoomed_{int(x)}_{int(y)}_{zoom_level}.mp4"

    try:
        # Load video gốc
        clip = VideoFileClip(video_path)

        # === Cấu hình zoom ===
        zoom_duration = duration          # Thời gian hiệu ứng zoom (giây)
        zoom_start_time = start_time      # Thời điểm bắt đầu zoom (giây)
        zoom_percent = zoom_level         # Zoom vào theo tỷ lệ (0.3 = 30%)

        # Kích thước video gốc và tỷ lệ
        W, H = clip.size
        aspect_ratio = W / H

        # Tâm điểm vùng cần zoom
        cx, cy = int(x), int(y)

        # Kích thước vùng zoom nhỏ nhất (sau khi zoom xong)
        min_zoom_w = int(W * zoom_percent)
        min_zoom_h = int(min_zoom_w / aspect_ratio)

        # Hàm xác định vùng crop theo thời gian
        def dynamic_crop(t):
            if t < zoom_start_time:
                # Trước khi bắt đầu zoom: hiển thị toàn khung
                return 0, 0, W, H
            elif t >= zoom_start_time + zoom_duration:
                # Sau khi zoom xong: giữ vùng zoom cố định
                crop_w = min_zoom_w
                crop_h = min_zoom_h
            else:
                # Trong quá trình zoom: scale từ 1.0 đến target zoom
                alpha = (t - zoom_start_time) / zoom_duration
                crop_w = int(W - alpha * (W - min_zoom_w))
                crop_h = int(H - alpha * (H - min_zoom_h))
                # Đảm bảo đúng tỉ lệ
                crop_h = int(crop_w / aspect_ratio)

            # Tính vị trí crop từ tâm điểm
            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - crop_h // 2)
            x2 = min(W, x1 + crop_w)
            y2 = min(H, y1 + crop_h)

            # Cập nhật lại để đảm bảo đúng kích thước
            if x2 - x1 != crop_w:
                x1 = max(0, x2 - crop_w)
            if y2 - y1 != crop_h:
                y1 = max(0, y2 - crop_h)

            return x1, y1, x2, y2

        # Áp dụng crop động từng frame
        def zoom_frame_function(gf, t):
            x1, y1, x2, y2 = dynamic_crop(t)
            return clip.crop(x1, y1, x2, y2).resize((W, H)).get_frame(t)

        zoomed = clip.fl(zoom_frame_function)

        zoomed.set_duration(clip.duration).write_videofile(
            output_path,
            codec="libx264",
            verbose=False,  # Tắt log để giao diện sạch hơn
            logger=None
        )

        clip.close()
        zoomed.close()

        print(f"✅ Đã xử lý zoom: {video_path} -> {output_path}")
        print(f"📍 Tọa độ: ({x}, {y}), ⏰ Thời gian: {start_time}s, ⏱️ Thời lượng: {duration}s, 🔍 Zoom: {zoom_level}")

        return output_path

    except Exception as e:
        print(f"❌ Lỗi khi xử lý zoom: {str(e)}")
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path

def process_video(video_file, time_list_str):
    """Xử lý video upload và cắt thành các clip nhỏ"""
    try:
        # Parse danh sách thời điểm
        time_list = [float(x.strip()) for x in time_list_str.split(',')]

        # Cắt video
        clips = split_video_by_times(video_file.name, time_list)

        # Tạo gallery để hiển thị các clip - Gallery cần format đúng
        video_gallery = clips  # Gallery sẽ tự động hiển thị video files

        return video_gallery, f"Đã cắt thành {len(clips)} clip"

    except Exception as e:
        return [], f"Lỗi: {str(e)}"

def update_clip_gallery():
    """Cập nhật gallery hiển thị các clip"""
    return current_clips

def sub_split_clip(clip_index, sub_times_str):
    """Cắt nhỏ thêm một clip cụ thể"""
    # TODO: Implement sub-splitting logic
    return f"Đã cắt nhỏ thêm clip {clip_index + 1} tại các thời điểm: {sub_times_str}"

def add_zoom_effect(clip_index, x, y, start_time, duration, zoom_level):
    """Thêm hiệu ứng zoom vào clip"""
    global current_preview_video

    try:
        if clip_index < 0 or clip_index >= len(current_clips):
            return None, f"Clip index không hợp lệ: {clip_index}"

        # Lấy video gốc
        original_video = current_clips[clip_index]

        # Xử lý zoom effect
        zoomed_video = process_zoom_effect(original_video, x, y, start_time, duration, zoom_level)

        # Lưu video đã zoom để preview
        current_preview_video = zoomed_video

        return zoomed_video, f"Đã thêm zoom vào clip {clip_index + 1}: tọa độ ({x}, {y}), thời gian {start_time}s, thời lượng {duration}s, mức zoom {zoom_level}"

    except Exception as e:
        return None, f"Lỗi khi thêm zoom: {str(e)}"

def save_to_queue(clip_choice):
    global video_queue, current_preview_video, current_clips

    video_to_save = None
    clip_name = ""

    if current_preview_video:
        # Nếu có video đã được chỉnh sửa (có hiệu ứng), ưu tiên lưu video này
        video_to_save = current_preview_video
        clip_name = os.path.basename(current_preview_video)
    elif clip_choice and current_clips:
        # Nếu không có video chỉnh sửa, lưu clip gốc được chọn
        clip_idx = int(clip_choice.split()[-1]) - 1
        if 0 <= clip_idx < len(current_clips):
            video_to_save = current_clips[clip_idx]
            clip_name = os.path.basename(current_clips[clip_idx])

    if video_to_save:
        video_queue.append(video_to_save)
        queue_display = update_queue_display()
        return queue_display, f"Đã lưu '{clip_name}' vào hàng chờ. Tổng cộng: {len(video_queue)} video"
    else:
        return [], "Vui lòng chọn clip để lưu vào hàng chờ"

def update_queue_display():
    """Cập nhật hiển thị hàng chờ"""
    return video_queue

def clear_queue():
    global video_queue, transition_settings
    video_queue = []
    transition_settings = {}
    return [], "Đã xóa toàn bộ hàng chờ"

def remove_from_queue(queue_index):
    global video_queue, transition_settings
    try:
        if 0 <= queue_index < len(video_queue):
            removed = video_queue.pop(queue_index)
            # Cập nhật lại transition settings sau khi xóa
            new_transition_settings = {}
            for i, (key, value) in enumerate(transition_settings.items()):
                if i < queue_index:
                    new_transition_settings[i] = value
                elif i > queue_index:
                    new_transition_settings[i-1] = value
            transition_settings = new_transition_settings
            return update_queue_display(), f"Đã xóa video khỏi hàng chờ: {os.path.basename(removed)}"
        else:
            return video_queue, "Index không hợp lệ"
    except Exception as e:
        return video_queue, f"Lỗi khi xóa: {str(e)}"

def get_transition_pairs():
    """Lấy danh sách các cặp clip để thêm chuyển cảnh"""
    if len(video_queue) < 2:
        return []

    pairs = []
    for i in range(len(video_queue) - 1):
        clip1_name = os.path.basename(video_queue[i])
        clip2_name = os.path.basename(video_queue[i + 1])
        pairs.append(f"Giữa Clip {i+1} ({clip1_name}) → Clip {i+2} ({clip2_name})")

    return pairs

def update_transition_dropdown():
    """Cập nhật dropdown chọn cặp clip để thêm transition"""
    pairs = get_transition_pairs()
    return gr.Dropdown(choices=pairs, value=pairs[0] if pairs else None)

def set_transition(pair_choice, transition_type):
    """Thiết lập hiệu ứng chuyển cảnh cho cặp clip"""
    global transition_settings

    if not pair_choice:
        return "Vui lòng chọn cặp clip", ""

    # Parse index từ pair_choice
    pair_index = int(pair_choice.split("Giữa Clip ")[1].split(" ")[0]) - 1

    # Lưu cài đặt transition
    transition_settings[pair_index] = transition_type

    # Tạo summary transition settings
    summary = create_transition_summary()

    return f"✅ Đã thiết lập hiệu ứng '{transition_type}' giữa clip {pair_index + 1} và {pair_index + 2}", summary

def create_transition_summary():
    """Tạo tóm tắt các cài đặt chuyển cảnh"""
    if not transition_settings:
        return "Chưa có hiệu ứng chuyển cảnh nào được thiết lập."

    summary_lines = []
    for pair_index, transition_type in transition_settings.items():
        summary_lines.append(f"• Clip {pair_index + 1} → Clip {pair_index + 2}: {transition_type}")

    return "\n".join(summary_lines)

def clear_all_transitions():
    """Xóa tất cả cài đặt chuyển cảnh"""
    global transition_settings
    transition_settings = {}
    return "Đã xóa tất cả cài đặt chuyển cảnh.", ""

def add_transition_effect(clip_index, transition_type):
    return f"Đã thêm hiệu ứng chuyển cảnh '{transition_type}' cho clip {clip_index + 1}"

def export_final_video():
    global video_queue, transition_settings

    if not video_queue:
        return "Hàng chờ trống. Không có video để xuất."

    if len(video_queue) == 1:
        return f"✅ Chỉ có 1 video trong hàng chờ. Video đã sẵn sàng: {os.path.basename(video_queue[0])}"

    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips

        processed_clips = []

        for i in range(len(video_queue)):
            current_video_path = video_queue[i]
            current_clip = VideoFileClip(current_video_path)

            if i == 0:
                processed_clips.append(current_clip)
                print(f"✅ Đã thêm video đầu tiên: {os.path.basename(current_video_path)}")
            else:
                transition_index = i - 1
                if transition_index in transition_settings:
                    transition_type = transition_settings[transition_index]
                    previous_video_path = processed_clips[-1].filename

                    print(f"🎭 Áp dụng hiệu ứng '{transition_type}' giữa:")
                    print(f"   - Video {i}: {os.path.basename(previous_video_path)}")
                    print(f"   - Video {i+1}: {os.path.basename(current_video_path)}")

                    transition_result = apply_transition_effect(
                        previous_video_path,
                        current_video_path,
                        transition_type
                    )

                    # if transition_result:

                    #     processed_clips.pop()  # Xóa previous clip

                    #     transition_clip = VideoFileClip(transition_result)
                    #     processed_clips.append(transition_clip)
                    if transition_result:
                        if len(processed_clips) > 0:
                            processed_clips.pop()  # Chỉ xóa nếu clip trước là clip gốc chưa merge
                        transition_clip = VideoFileClip(transition_result)
                        processed_clips.append(transition_clip)
                        print(f"✅ Đã áp dụng transition, kết quả: {os.path.basename(transition_result)}")
                    else:
                        processed_clips.append(current_clip)
                        print(f"⚠️ Transition thất bại, nối bình thường")
                else:
                    processed_clips.append(current_clip)
                    print(f"✅ Nối bình thường: {os.path.basename(current_video_path)}")

        # Nối tất cả các clip lại với nhau
        print(f"🔗 Bắt đầu nối {len(processed_clips)} clip...")
        final_video = concatenate_videoclips(processed_clips)

        # Xuất video cuối cùng
        output_path = "final_video.mp4"
        final_video.write_videofile(
            output_path,
            codec="libx264",
            verbose=False,
            logger=None
        )


        for clip in processed_clips:
            clip.close()
        final_video.close()

        export_info = [
            f"🎬 Đã xuất video thành công: {output_path}",
            f"📊 Tổng cộng {len(video_queue)} video được nối:",
            ""
        ]

        for i, video_path in enumerate(video_queue):
            video_name = os.path.basename(video_path)
            export_info.append(f"  {i+1}. {video_name}")

            if i in transition_settings:
                transition = transition_settings[i]
                export_info.append(f"      ↳ Chuyển cảnh: {transition}")

    #     return "\n".join(export_info)

    # except Exception as e:
    #     return f"❌ Lỗi khi xuất video: {str(e)}"
        return output_path, export_info
    except Exception as e:
        return None, f"❌ Lỗi khi xuất video: {str(e)}"

def apply_transition_effect(video1_path, video2_path, transition_type):

    output_path = f"transition_{transition_type}_{len(video_queue)}_{hash(video1_path + video2_path) % 10000}.mp4"
    if transition_type=="None":
      return None
    try:

        tool = VideoTransitionTool()
        tool.merge_videos_with_transition(
            video1_path,
            video2_path,
            output_path,
            transition_type=transition_type,
            transition_duration=0.5
        )
        return output_path
    except Exception as e:
        print(f"❌ Lỗi khi áp dụng transition {transition_type}: {str(e)}")
        return None

# Các hàm hiệu ứng chuyển cảnh (để trống như yêu cầu)
# def crossfade_transition():
#     pass

# def slide_horizontal_transition():
#     pass

# def slide_vertical_transition():
#     pass

# def zoom_in_transition():
#     pass

# def zoom_out_transition():
#     pass

# def flash_cut_transition():
#     pass

# def push_blur_transition():
#     pass

# def rgb_split_transition():
#     pass

# def circle_mask_transition():
#     pass

# def square_mask_transition():
#     pass

with gr.Blocks(title="Video Editor", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎬 Video Editor Tool")
    gr.Markdown("Upload video và nhập danh sách thời điểm để cắt video thành các clip nhỏ")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(
                label="📁 Upload Video",
                file_types=[".mp4", ".avi", ".mov", ".mkv"]
            )

            time_input = gr.Textbox(
                label="⏰ Danh sách thời điểm (phân cách bằng dấu phẩy)",
                placeholder="0, 3.95, 7.91, 11.94",
                value="0, 3.95, 7.91, 11.94"
            )

            split_btn = gr.Button("✂️ Cắt Video", variant="primary")

            result_msg = gr.Textbox(label="📝 Kết quả", interactive=False)

        with gr.Column(scale=2):
            clips_gallery = gr.Gallery(
                label="🎥 Video Clips",
                show_label=True,
                elem_id="clips_gallery",
                columns=2,
                rows=2,
                height="auto"
            )

    gr.Markdown("## 🛠️ Tùy chỉnh từng clip")

    with gr.Row():
      with gr.Column(scale=1):
          clip_selector = gr.Dropdown(
              label="📋 Chọn clip để chỉnh sửa",
              choices=[],
              interactive=True
          )

          with gr.Tabs():
              with gr.Tab("🔍 Zoom"):
                  with gr.Row():
                      zoom_x = gr.Number(label="Tọa độ X", value=0)
                      zoom_y = gr.Number(label="Tọa độ Y", value=0)
                  with gr.Row():
                      zoom_start = gr.Number(label="Thời gian bắt đầu (s)", value=0)
                      zoom_duration= gr.Number(label="Thời gian kéo dài (s)", value=0.5)
                      zoom_level = gr.Number(label="Mức độ zoom vào (0.3=30%)", value=0.3)

                  zoom_btn = gr.Button("Thêm Zoom")
                  zoom_result = gr.Textbox(label="Kết quả", interactive=False)

              # with gr.Tab("✨ Hiệu ứng chuyển cảnh"):
              #     transition_type = gr.Dropdown(
              #         label="Chọn hiệu ứng",
              #         choices=[
              #             "crossfade",
              #             "slide_horizontal",
              #             "slide_vertical",
              #             "zoom_in",
              #             "zoom_out",
              #             "flash_cut",
              #             "push_blur",
              #             "rgb_split",
              #             "circle_mask",
              #             "square_mask"
              #         ],
              #         value="crossfade"
              #     )
              #     transition_btn = gr.Button("Thêm hiệu ứng chuyển cảnh")
              #     transition_result = gr.Textbox(label="Kết quả", interactive=False)
              with gr.Tab("🛍️ Video Sản phẩm"):
                  product_image = gr.File(
                      label="📸 Upload ảnh sản phẩm",
                      file_types=[".jpg", ".jpeg", ".png"],
                      type="filepath"
                  )
                  product_effect = gr.Dropdown(
                      label="Chọn hiệu ứng",
                      choices=[
                          "sweep_light",
                          "kenburn",
                          "zoom_pan",
                          "fade_in_out"
                      ],
                      value="sweep_light"
                  )
                  product_btn = gr.Button("Tạo video sản phẩm", variant="primary")
                  product_result = gr.Textbox(label="Kết quả", interactive=False)

      with gr.Column(scale=1):
          clip_preview = gr.Video(label="👁️ Preview Clip", interactive=False)

          # Nút lưu vào hàng chờ
          save_queue_btn = gr.Button("💾 Lưu lại cảnh này", variant="secondary", size="lg")
          save_queue_result = gr.Textbox(label="Trạng thái lưu", interactive=False)

    # Phần hàng chờ video
    gr.Markdown("## 📋 Hàng chờ Video")

    with gr.Row():
        with gr.Column(scale=3):
            # Hiển thị hàng chờ
            queue_gallery = gr.Gallery(
                label="🎬 Video trong hàng chờ",
                show_label=True,
                elem_id="queue_gallery",
                columns=3,
                rows=2,
                height="auto"
            )

        with gr.Column(scale=1):
            # Các nút quản lý hàng chờ
            queue_info = gr.Textbox(label="📊 Thông tin hàng chờ", interactive=False)
            clear_queue_btn = gr.Button("🗑️ Xóa toàn bộ hàng chờ", variant="stop")
            queue_index_input = gr.Number(label="Index để xóa", value=0, precision=0)
            remove_queue_btn = gr.Button("❌ Xóa video theo index", variant="secondary")

    # Phần quản lý hiệu ứng chuyển cảnh giữa các clip
    gr.Markdown("## 🎭 Quản lý Hiệu ứng Chuyển cảnh")
    gr.Markdown("*Thiết lập hiệu ứng chuyển cảnh giữa các clip trước khi xuất video cuối cùng*")

    with gr.Row():
        with gr.Column(scale=2):
            # Chọn cặp clip để thêm transition
            transition_pair_selector = gr.Dropdown(
                label="🔗 Chọn cặp clip để thêm chuyển cảnh",
                choices=[],
                interactive=True
            )

            # Chọn loại hiệu ứng
            transition_effect_selector = gr.Dropdown(
                label="✨ Chọn hiệu ứng chuyển cảnh",
                choices=[
                    "crossfade",
                    "slide_horizontal",
                    "slide_vertical",
                    "zoom_out",
                    "push_blur",
                    "rgb_split",
                    "circle_mask",
                    "None"
                ],
                value="crossfade"
            )
            # transition_duration_selector=

            # Các nút điều khiển
            with gr.Row():
                set_transition_btn = gr.Button("➕ Thiết lập Chuyển cảnh", variant="primary")
                clear_transitions_btn = gr.Button("🗑️ Xóa Tất cả", variant="secondary")

            transition_set_result = gr.Textbox(label="Kết quả thiết lập", interactive=False)

        with gr.Column(scale=2):
            transition_summary = gr.Textbox(
                label="📋 Tóm tắt Chuyển cảnh",
                placeholder="Chưa có hiệu ứng chuyển cảnh nào được thiết lập.",
                lines=8,
                interactive=False
            )


    # Nút xuất video cuối cùng
    gr.Markdown("## 📤 Xuất video")
    export_btn = gr.Button("🎬 Xuất Video Cuối Cùng", variant="primary", size="lg")
    export_result = gr.Textbox(label="Kết quả xuất", interactive=False, lines=10)
    final_video_preview = gr.Video(label="🎞 Video cuối cùng", interactive=False)


    def update_clip_dropdown():
        if current_clips:
            choices = [f"Clip {i+1}" for i in range(len(current_clips))]
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
        return gr.Dropdown(choices=[], value=None)

    def on_clip_select(clip_choice):
        """Khi chọn clip, hiển thị preview"""
        if clip_choice and current_clips:
            clip_idx = int(clip_choice.split()[-1]) - 1
            if 0 <= clip_idx < len(current_clips):
                return current_clips[clip_idx]
        return None

    def on_split_video(video, times):
        gallery, msg = process_video(video, times)
        print("current_clips: ",current_clips)
        clip_choices = [f"Clip {i+1}" for i in range(len(current_clips))]
        preview_video = current_clips[0] if current_clips else None
        return gallery, msg, gr.Dropdown(choices=clip_choices, value=clip_choices[0] if clip_choices else None), preview_video

    def on_zoom(clip_choice, x, y, start, duration, level):
        if clip_choice:
            clip_idx = int(clip_choice.split()[-1]) - 1
            preview_video, result_msg = add_zoom_effect(clip_idx, x, y, start, duration, level)
            return preview_video, result_msg
        return None, "Chọn clip trước"

    def on_transition(clip_choice, trans_type):
        if clip_choice:
            clip_idx = int(clip_choice.split()[-1]) - 1
            return add_transition_effect(clip_idx, trans_type)
        return "Chọn clip trước"

    def on_save_queue(clip_choice):
        queue_display, msg = save_to_queue(clip_choice)
        # Cập nhật transition pair selector khi có video mới được thêm
        transition_pairs = get_transition_pairs()
        return (
            queue_display,
            msg,
            f"Số video trong hàng chờ: {len(video_queue)}",
            gr.Dropdown(choices=transition_pairs, value=transition_pairs[0] if transition_pairs else None)
        )

    def on_clear_queue():
        queue_display, msg = clear_queue()
        return (
            queue_display,
            msg,
            "Hàng chờ trống",
            gr.Dropdown(choices=[], value=None),
            ""
        )

    def on_remove_from_queue(index):
        queue_display, msg = remove_from_queue(int(index))
        transition_pairs = get_transition_pairs()
        summary = create_transition_summary()
        return (
            queue_display,
            msg,
            f"Số video trong hàng chờ: {len(video_queue)}",
            gr.Dropdown(choices=transition_pairs, value=transition_pairs[0] if transition_pairs else None),
            summary
        )

    def on_set_transition(pair_choice, transition_type):
        result_msg, summary = set_transition(pair_choice, transition_type)
        return result_msg, summary

    def on_clear_transitions():
        result_msg, summary = clear_all_transitions()
        return result_msg, summary
    def on_product_video(clip_choice, image_path, effect_type):
        if not clip_choice:
            return None, "❌ Chưa chọn clip nguồn"

        if not image_path:
            return None, "❌ Chưa upload ảnh sản phẩm"

        clip_idx = int(clip_choice.split()[-1]) - 1
        if clip_idx < 0 or clip_idx >= len(current_clips):
            return None, "❌ Clip không hợp lệ"

        source_video = current_clips[clip_idx]  # đường dẫn video gốc

        try:
            # Gọi hàm xử lý của bạn, ví dụ:
            output_path = f"product_video_{clip_idx}.mp4"
            create_product_video(
                source_video,
                image_path,
                effect_type,
                output_path
            )
            return output_path, f"✅ Đã tạo video sản phẩm: {output_path}"
        except Exception as e:
            return None, f"❌ Lỗi khi tạo video: {str(e)}"
# =================================================================================
    from moviepy.editor import ImageClip, concatenate_videoclips
    import numpy as np
    import cv2
    def ken_burns_effect(image_path, output_path="ken_burns_output.mp4", duration=6, scale=0.8, fps=30):
        # Đọc ảnh
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        # Kích thước khung hình nhỏ hơn
        crop_h, crop_w = int(h * scale), int(w * scale)
        step_count = int(duration * fps)

        frames = []

        for i in range(step_count):
            progress = i / step_count

            # Giai đoạn đầu: từ trái qua phải (nửa thời gian đầu)
            if progress <= 0.5:
                x = int(progress * 2 * (w - crop_w))  # trái ➝ phải
                y = 0
            else:
                # Giai đoạn sau: phải qua trái ở dưới
                x = int((1 - (progress - 0.5) * 2) * (w - crop_w))  # phải ➝ trái
                y = h - crop_h

            # Cắt ảnh con
            cropped = image[y:y+crop_h, x:x+crop_w]
            resized = cv2.resize(cropped, (w, h))
            frames.append(resized)

        # Ghi video bằng MoviePy
        def make_frame(t):
            idx = min(int(t * fps), len(frames) - 1)
            return frames[idx][:, :, ::-1]  # BGR ➝ RGB

        clip = ImageClip(frames[0][:, :, ::-1], duration=duration)
        video = clip.set_make_frame(make_frame).set_duration(duration)
        video.write_videofile(output_path, fps=fps)
# ===========================================================================
    def create_product_video(video_path, image_path, effect_type, output_path):

        ken_burns_effect(image_path, output_path=output_path, duration=6)


        # import shutil
        # shutil.copy(video_path, output_path)
    split_btn.click(
        on_split_video,
        inputs=[video_input, time_input],
        outputs=[clips_gallery, result_msg, clip_selector, clip_preview]
    )

    clip_selector.change(
        on_clip_select,
        inputs=clip_selector,
        outputs=clip_preview
    )

    zoom_btn.click(
        on_zoom,
        inputs=[clip_selector, zoom_x, zoom_y, zoom_start, zoom_duration, zoom_level],
        outputs=[clip_preview, zoom_result]
    )

    product_btn.click(
        on_product_video,
        inputs=[clip_selector, product_image, product_effect],
        outputs=[clip_preview, product_result]
    )


    save_queue_btn.click(
        on_save_queue,
        inputs=clip_selector,  # Truyền clip_selector làm input
        outputs=[queue_gallery, save_queue_result, queue_info, transition_pair_selector]
    )

    clear_queue_btn.click(
        on_clear_queue,
        outputs=[queue_gallery, save_queue_result, queue_info, transition_pair_selector, transition_summary]
    )

    remove_queue_btn.click(
        on_remove_from_queue,
        inputs=queue_index_input,
        outputs=[queue_gallery, save_queue_result, queue_info, transition_pair_selector, transition_summary]
    )

    set_transition_btn.click(
        on_set_transition,
        inputs=[transition_pair_selector, transition_effect_selector],
        outputs=[transition_set_result, transition_summary]
    )

    clear_transitions_btn.click(
        on_clear_transitions,
        outputs=[transition_set_result, transition_summary]
    )

    # export_btn.click(
    #     export_final_video,
    #     outputs=export_result
    # )
    export_btn.click(
        export_final_video,
        outputs=[final_video_preview, export_result]
    )


if __name__ == "__main__":
    app.launch(debug=True, share=True)
