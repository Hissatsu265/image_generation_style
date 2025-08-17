import os
import uuid
from pathlib import Path
from typing import List
from app.models.schemas import TransitionEffect, DollyEffect
import asyncio
import time

class VideoEffectService:
    def __init__(self):
        pass  # Không cần khởi tạo gì đặc biệt

    async def apply_effects(self, 
                          video_path: str,
                          transition_times: List[float],
                          transition_effects: List[TransitionEffect],
                          transition_durations: List[float],
                          dolly_effects: List[DollyEffect] = None,
                          job_id: str = None) -> str:
        """
        Áp dụng hiệu ứng cho video
        
        Args:
            video_path: Đường dẫn video input
            transition_times: Danh sách thời điểm chuyển cảnh (giây)
            transition_effects: Danh sách hiệu ứng chuyển cảnh
            transition_durations: Danh sách thời gian duration từng hiệu ứng (giây)
            dolly_effects: Danh sách hiệu ứng dolly (tùy chọn)
            job_id: ID của job
            
        Returns:
            str: Đường dẫn video output đã xử lý
        """
        
        # Validate input
        if len(transition_times) != len(transition_effects) != len(transition_durations):
            raise ValueError("transition_times, transition_effects, transition_durations must have same length")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        # Tạo output path
        from config import OUTPUT_DIR
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        output_filename = f"effect_{job_id or uuid.uuid4().hex}.mp4"
        output_path = output_dir / output_filename
        
        # === TỰ XỬ LÝ HIỆU ỨNG TẠI ĐÂY ===
        # Bạn có thể truy cập tất cả thông tin:
        # - video_path: đường dẫn video gốc
        # - transition_times: [1.5, 3.0, 5.2, ...] - thời điểm chuyển cảnh
        # - transition_effects: [TransitionEffect.FADE, TransitionEffect.ZOOM_IN, ...] - loại hiệu ứng
        # - transition_durations: [0.5, 1.0, 0.8, ...] - thời gian hiệu ứng
        # - dolly_effects: list các DollyEffect object với thông tin chi tiết
        # - job_id: ID của job để track progress
        print("=========================================================")
        print(f"Processing video effects for job {job_id}")
        print(f"Input video: {video_path}")
        print(f"Transition times: {transition_times}")
        print(f"Transition effects: {transition_effects}")
        print(f"Transition durations: {transition_durations}")
        print(f"Dolly effects: {len(dolly_effects or [])} effects")
        print(f"Output will be: {output_path}")
        

        time.sleep(7)
        
        print("=========================================================")
        # TODO: Thực hiện xử lý video với các hiệu ứng
        # Ví dụ có thể sử dụng:
        # - FFmpeg với filter complex
        # - OpenCV cho xử lý frame by frame
        # - MoviePy cho Python-based video processing
        # - Các library khác...
        
        # Tạm thời copy file để test (XÓA DÒNG NÀY KHI IMPLEMENT THẬT)
        await self._mock_process_video(video_path, str(output_path), job_id)
        
        return str(output_path)

    async def _mock_process_video(self, input_path: str, output_path: str, job_id: str):
        """
        Mock function để test - XÓA KHI IMPLEMENT THẬT
        """
        import shutil
        
        # Giả lập thời gian xử lý
        await asyncio.sleep(2)  # 2 giây để test
        
        # Copy file để có output (chỉ để test)
        shutil.copy2(input_path, output_path)
        print(f"Mock processing completed for job {job_id}")

    async def get_video_duration(self, video_path: str) -> float:
        """
        Lấy duration của video - có thể sử dụng ffprobe hoặc library khác
        """
        
        # TODO: Implement lấy duration thật
        # Có thể dùng:
        # - ffprobe command
        # - OpenCV cv2.VideoCapture
        # - MoviePy VideoFileClip
        
        # Mock duration cho test
        return 60.0  # 60 giây

    def validate_effects_timing(self, 
                              transition_times: List[float],
                              dolly_effects: List[DollyEffect],
                              video_duration: float):
        """
        Validate timing của các effects không vượt quá video duration
        """
        
        # Kiểm tra transition times
        for time_point in transition_times:
            if time_point > video_duration:
                raise ValueError(f"Transition time {time_point}s exceeds video duration {video_duration}s")
        
        # Kiểm tra dolly effects
        for dolly in dolly_effects or []:
            if dolly.start_time + dolly.duration > video_duration:
                raise ValueError(f"Dolly effect (start: {dolly.start_time}s, duration: {dolly.duration}s) exceeds video duration {video_duration}s")