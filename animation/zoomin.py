# !pip install opencv-python mediapipe pillow numpy moviepy

import cv2
import numpy as np
from PIL import Image
import os
import mediapipe as mp
import random
import math

class VideoFaceZoom:
    def __init__(self, input_video_path, output_video_path):
        self.input_path = input_video_path
        self.output_path = output_video_path

        # Khởi tạo MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

        # Đọc video
        self.cap = cv2.VideoCapture(input_video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Thiết lập video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))

    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * self.width)
                y = int(bbox.ymin * self.height)
                w = int(bbox.width * self.width)
                h = int(bbox.height * self.height)
                padding_left = 0.3    
                padding_right = 0.3   
                padding_top = 0.5     
                padding_bottom = 0.2  
                if self.height > self.width:
                    padding_left = padding_right = padding_top = padding_bottom = 0.1
                    # print("======H>W===")
                     

                x_expand_left = int(w * padding_left)
                x_expand_right = int(w * padding_right)
                y_expand_top = int(h * padding_top)
                y_expand_bottom = int(h * padding_bottom)

                x = max(0, x - x_expand_left)
                y = max(0, y - y_expand_top)
                w = min(self.width - x, w + x_expand_left + x_expand_right)
                h = min(self.height - y, h + y_expand_top + y_expand_bottom)

                faces.append((x, y, x + w, y + h))

        return faces

    def get_largest_face(self, faces):
        if not faces:
            return None
        largest_face = max(faces, key=lambda face: (face[2] - face[0]) * (face[3] - face[1]))
        return largest_face

    # def calculate_zoom_region(self, face_bbox, zoom_factor=1.5):
    #     x1, y1, x2, y2 = face_bbox
    #     face_center_x = (x1 + x2) // 2
    #     face_center_y = (y1 + y2) // 2

    #     zoom_width = int(self.width / zoom_factor)
    #     zoom_height = int(self.height / zoom_factor)

    #     zoom_x1 = max(0, face_center_x - zoom_width // 2)
    #     zoom_y1 = max(0, face_center_y - zoom_height // 2)
    #     zoom_x2 = min(self.width, zoom_x1 + zoom_width)
    #     zoom_y2 = min(self.height, zoom_y1 + zoom_height)

    #     if zoom_x2 - zoom_x1 < zoom_width:
    #         zoom_x1 = max(0, zoom_x2 - zoom_width)
    #     if zoom_y2 - zoom_y1 < zoom_height:
    #         zoom_y1 = max(0, zoom_y2 - zoom_height)

    #     return (zoom_x1, zoom_y1, zoom_x2, zoom_y2)
    def calculate_zoom_region(self, face_bbox, zoom_factor=1.5):
        x1, y1, x2, y2 = face_bbox
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2

        zoom_width = int(self.width / zoom_factor)
        zoom_height = int(self.height / zoom_factor)

        # ⚠️ Điều chỉnh để vùng zoom lệch xuống dưới nếu là video dọc
        if self.height > self.width:
            face_center_y += int(0.1 * self.height)  # đẩy vùng zoom xuống 10% chiều cao

        zoom_x1 = max(0, face_center_x - zoom_width // 2)
        zoom_y1 = max(0, face_center_y - zoom_height // 2)
        zoom_x2 = min(self.width, zoom_x1 + zoom_width)
        zoom_y2 = min(self.height, zoom_y1 + zoom_height)

        if zoom_x2 - zoom_x1 < zoom_width:
            zoom_x1 = max(0, zoom_x2 - zoom_width)
        if zoom_y2 - zoom_y1 < zoom_height:
            zoom_y1 = max(0, zoom_y2 - zoom_height)

        return (zoom_x1, zoom_y1, zoom_x2, zoom_y2)

    def smooth_transition(self, current_region, target_region, alpha=0.1):
        if current_region is None:
            return target_region

        smooth_region = []
        for i in range(4):
            smooth_val = int(current_region[i] * (1 - alpha) + target_region[i] * alpha)
            smooth_region.append(smooth_val)

        return tuple(smooth_region)

    def apply_shake_effect(self, zoom_region, shake_intensity=5):
        
        if zoom_region is None:
            return zoom_region
            
        x1, y1, x2, y2 = zoom_region
        
        # Tạo độ lệch ngẫu nhiên
        shake_x = random.randint(-shake_intensity, shake_intensity)
        shake_y = random.randint(-shake_intensity, shake_intensity)
        
        # Áp dụng shake nhưng đảm bảo không vượt biên
        new_x1 = max(0, min(self.width - (x2 - x1), x1 + shake_x))
        new_y1 = max(0, min(self.height - (y2 - y1), y1 + shake_y))
        new_x2 = new_x1 + (x2 - x1)
        new_y2 = new_y1 + (y2 - y1)
        
        return (new_x1, new_y1, new_x2, new_y2)

    def calculate_gradual_zoom_factor(self, current_frame, start_frame, end_frame, start_zoom=1.0, end_zoom=1.5):
        """
        Tính zoom factor cho zoom từ từ
        """
        if current_frame < start_frame:
            return start_zoom
        elif current_frame >= end_frame:
            return end_zoom
        else:
            # Tính toán zoom factor theo thời gian (easing)
            progress = (current_frame - start_frame) / (end_frame - start_frame)
            # Sử dụng easing function để smooth hơn
            eased_progress = 1 - math.cos(progress * math.pi / 2)  # ease-out sine
            return start_zoom + (end_zoom - start_zoom) * eased_progress

    def apply_zoom_effect(self, frame, zoom_region, zoom_factor=1.5):
        x1, y1, x2, y2 = zoom_region
        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        return zoomed

    def process_video(self, zoom_start_frame=None, zoom_duration_frames=None, zoom_factor=1.5, 
                     zoom_type="instant", gradual_start_frame=None, gradual_end_frame=None, 
                     gradual_hold_frames=None, enable_shake=False, shake_intensity=3, shake_start_delay=0.5):

        
        frame_count = 0
        zoom_region = None
        target_face = None
        shake_start_frame = None
        shake_end_frame = None
        shake_duration_frames = int(0.5 * self.fps)  # Shake chỉ kéo dài 0.5 giây
        
        if zoom_type == "instant":
            print(f"Instant zoom từ frame {zoom_start_frame} đến {zoom_start_frame + zoom_duration_frames}")
            if enable_shake:
                shake_start_frame = zoom_start_frame + int(shake_start_delay * self.fps)
                shake_end_frame = shake_start_frame + shake_duration_frames
                print(f"Shake effect từ frame {shake_start_frame} đến {shake_end_frame}")
        else:
            gradual_total_end = gradual_end_frame + (gradual_hold_frames or 0)
            print(f"Gradual zoom từ frame {gradual_start_frame} đến {gradual_end_frame}")
            print(f"Giữ mức zoom đến frame {gradual_total_end}")
            if enable_shake:
                shake_start_frame = gradual_end_frame + int(shake_start_delay * self.fps)
                shake_end_frame = shake_start_frame + shake_duration_frames
                print(f"Shake effect từ frame {shake_start_frame} đến {shake_end_frame}")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            current_zoom_factor = 1.0
            should_zoom = False

            # Xác định có nên zoom không và zoom factor
            if zoom_type == "instant":
                if zoom_start_frame <= frame_count < zoom_start_frame + zoom_duration_frames:
                    should_zoom = True
                    current_zoom_factor = zoom_factor
            elif zoom_type == "gradual":
                if gradual_start_frame <= frame_count <= gradual_end_frame:
                    # Đang trong giai đoạn zoom từ từ
                    should_zoom = True
                    current_zoom_factor = self.calculate_gradual_zoom_factor(
                        frame_count, gradual_start_frame, gradual_end_frame, 1.0, zoom_factor
                    )
                elif gradual_hold_frames and gradual_end_frame < frame_count <= gradual_end_frame + gradual_hold_frames:
                    # Đang trong giai đoạn giữ mức zoom tối đa
                    should_zoom = True
                    current_zoom_factor = zoom_factor

            # Xử lý zoom
            if should_zoom and current_zoom_factor > 1.0:
                if faces:
                    current_largest = self.get_largest_face(faces)

                    if target_face is None:
                        target_face = current_largest
                    else:
                        target_area = (target_face[2] - target_face[0]) * (target_face[3] - target_face[1])
                        current_area = (current_largest[2] - current_largest[0]) * (current_largest[3] - current_largest[1])

                        if current_area > target_area * 1.2:
                            target_face = current_largest

                    target_zoom_region = self.calculate_zoom_region(target_face, current_zoom_factor)
                    zoom_region = self.smooth_transition(zoom_region, target_zoom_region, alpha=0.15)

                    # Áp dụng shake effect chỉ trong khoảng thời gian nhất định
                    if (enable_shake and shake_start_frame and shake_end_frame and 
                        shake_start_frame <= frame_count < shake_end_frame):
                        zoom_region = self.apply_shake_effect(zoom_region, shake_intensity)

                    frame = self.apply_zoom_effect(frame, zoom_region, current_zoom_factor)

                elif zoom_region is not None:
                    # Áp dụng shake effect chỉ trong khoảng thời gian nhất định
                    if (enable_shake and shake_start_frame and shake_end_frame and 
                        shake_start_frame <= frame_count < shake_end_frame):
                        zoom_region = self.apply_shake_effect(zoom_region, shake_intensity)
                    
                    frame = self.apply_zoom_effect(frame, zoom_region, current_zoom_factor)
            else:
                # Reset khi không zoom
                if not should_zoom:
                    target_face = None
                    zoom_region = None

            self.out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Đã xử lý: {frame_count}/{self.total_frames} frames")

        print("Hoàn thành xử lý video!")

    def release(self):
        self.cap.release()
        self.out.release()

def create_face_zoom_video(input_video, output_video, zoom_type="instant", **kwargs):
    
    
    processor = VideoFaceZoom(input_video, output_video)
    
    # Thiết lập parameters
    zoom_factor = kwargs.get('zoom_factor', 1.8)
    enable_shake = kwargs.get('enable_shake', False)
    shake_intensity = kwargs.get('shake_intensity', 3)
    shake_start_delay = kwargs.get('shake_start_delay', 0.5)
    
    try:
        if zoom_type == "instant":
            zoom_start_time = kwargs.get('zoom_start_time', 0)
            zoom_duration = kwargs.get('zoom_duration', 2)
            
            zoom_start_frame = int(zoom_start_time * processor.fps)
            zoom_duration_frames = int(zoom_duration * processor.fps)
            
            processor.process_video(
                zoom_start_frame=zoom_start_frame,
                zoom_duration_frames=zoom_duration_frames,
                zoom_factor=zoom_factor,
                zoom_type="instant",
                enable_shake=enable_shake,
                shake_intensity=shake_intensity,
                shake_start_delay=shake_start_delay
            )
            
        elif zoom_type == "gradual":
            gradual_start_time = kwargs.get('gradual_start_time', 0)
            gradual_end_time = kwargs.get('gradual_end_time', 3)
            hold_duration = kwargs.get('hold_duration', 2)  # Mặc định giữ 2 giây
            
            gradual_start_frame = int(gradual_start_time * processor.fps)
            gradual_end_frame = int(gradual_end_time * processor.fps)
            gradual_hold_frames = int(hold_duration * processor.fps)
            
            processor.process_video(
                zoom_factor=zoom_factor,
                zoom_type="gradual",
                gradual_start_frame=gradual_start_frame,
                gradual_end_frame=gradual_end_frame,
                gradual_hold_frames=gradual_hold_frames,
                enable_shake=enable_shake,
                shake_intensity=shake_intensity,
                shake_start_delay=shake_start_delay
            )
        
    finally:
        processor.release()

from moviepy.editor import VideoFileClip

def replace_audio(video_path, audio_source_path, output_path):
    video = VideoFileClip(video_path)
    new_audio = VideoFileClip(audio_source_path).audio
    video_with_new_audio = video.set_audio(new_audio)
    video_with_new_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path


# if __name__ == "__main__":
#     input_video_path = "/workspace/marketing-video-ai/e06b2a6c_clip_1_cut_10.3s.mp4"
    
#     # Ví dụ 1: Zoom đột ngột với hiệu ứng rung
#     print("=== Tạo zoom đột ngột với shake effect ===")
#     create_face_zoom_video(
#         input_video=input_video_path,
#         output_video="output_instant_zoom_shake.mp4",
#         zoom_type="instant",
#         zoom_start_time=1,
#         zoom_duration=4,
#         zoom_factor=1.8,
#         enable_shake=True,
#         shake_intensity=1,
#         shake_start_delay=0.3
#     )
    
#     #Ví dụ 2: Zoom từ từ với giữ mức zoom
#     print("=== Tạo zoom từ từ với giữ mức zoom ===")
#     create_face_zoom_video(
#         input_video=input_video_path,
#         output_video="output_gradual_zoom.mp4",
#         zoom_type="gradual",
#         gradual_start_time=1,
#         gradual_end_time=1.5,  # Zoom từ giây 1 đến giây 3
#         hold_duration=2,     # Giữ mức zoom trong 2 giây (đến giây 5)
#         zoom_factor=1.3
#     )
    
#     # Ví dụ 3: Zoom từ từ với shake effect (chỉ 0.5 giây)
#     print("=== Tạo zoom từ từ với shake effect ===")
#     create_face_zoom_video(
#         input_video=input_video_path,
#         output_video="output_gradual_zoom_shake.mp4",
#         zoom_type="gradual",
#         gradual_start_time=1,
#         gradual_end_time=3,    # Zoom từ giây 1 đến giây 3
#         hold_duration=2,       # Giữ mức zoom đến giây 5
#         zoom_factor=1.8,
#         enable_shake=True,
#         shake_intensity=3,
#         shake_start_delay=0.2  # Shake bắt đầu sau 0.2s khi zoom xong, kéo dài 0.5s
#     )
    
    # Thay audio
    # output = replace_audio("output_gradual_zoom.mp4", "/content/25_7padd.mp4", "final_output.mp4")
    # print("Video cuối cùng:", output)
