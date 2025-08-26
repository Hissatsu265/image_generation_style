import os
import uuid
from pathlib import Path
from typing import List

from app.models.schemas import TransitionEffect, DollyEffect, DollyEffectType, DollyEndType
from animation.full_transition_effect import apply_effect
from animation.zoomin_at_one_point import apply_zoom_effect
from animation.zoomin import safe_create_face_zoom_video

import asyncio

class VideoEffectService:
    def __init__(self):
        pass  # Không cần khởi tạo gì đặc biệt

    def _ensure_dolly_objects(self, dolly_effects):
        """Convert dict objects to DollyEffect objects if needed"""
        if not dolly_effects:
            return []
        
        result = []
        for dolly in dolly_effects:
            if isinstance(dolly, dict):
                # Convert dict to DollyEffect object
                dolly_obj = DollyEffect(
                    scene_index=dolly.get('scene_index'),
                    start_time=dolly.get('start_time', 0.0),
                    duration=dolly.get('duration', 0.5),
                    zoom_percent=dolly.get('zoom_percent', 0),
                    effect_type=DollyEffectType(dolly.get('effect_type')),
                    x_coordinate=dolly.get('x_coordinate'),
                    y_coordinate=dolly.get('y_coordinate'),
                    end_time=dolly.get('end_time'),
                    end_type=DollyEndType(dolly.get('end_type', 'smooth'))
                )
                result.append(dolly_obj)
            elif isinstance(dolly, DollyEffect):
                result.append(dolly)
            else:
                raise TypeError(f"Invalid dolly effect type: {type(dolly)}")
        
        return result

    async def apply_effects(self, 
                          video_path: str,
                          transition_times: List[float],
                          transition_effects: List[TransitionEffect],
                          transition_durations: List[float],
                          dolly_effects: List[DollyEffect] = None,
                          job_id: str = None) -> str:
        """
        Áp dụng hiệu ứng cho video - ASYNC version (deprecated, use apply_effects_sync)
        """
        return self.apply_effects_sync(
            video_path, transition_times, transition_effects, 
            transition_durations, dolly_effects, job_id
        )

    def apply_effects_sync(self, 
                          video_path: str,
                          transition_times: List[float],
                          transition_effects: List[TransitionEffect],
                          transition_durations: List[float],
                          dolly_effects: List[DollyEffect] = None,
                          job_id: str = None) -> str:
        # Validate input
        if len(transition_times) != len(transition_effects) != len(transition_durations):
            raise ValueError("transition_times, transition_effects, transition_durations must have same length")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        # Convert dict to DollyEffect objects if needed
        dolly_effects = self._ensure_dolly_objects(dolly_effects)
        
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
        # =======================Transition================================
        for i in range(len(transition_times)):
            
            if i>0: video_path = str(output_path)  
            start_time = transition_times[i] - transition_durations[i] / 2
            end_time = transition_times[i] + transition_durations[i] / 2
            effect_name = transition_effects[i]
            apply_effect(
                video_path=video_path,
                output_path=str(output_path),
                start_time=start_time,
                end_time=end_time,
                effect_name=effect_name
            )
            print(f"Applied effect {effect_name} from {start_time}s to {end_time}s on video {video_path}")
            print(f"Output saved to {output_path}")
            print("================================")
        # =======================Dolly================================
        k=0
        print("Dolly effects processing is currently disabled in the code.")
        print(f"Received {len(dolly_effects)} dolly effects:")
        # print(dolly_effects)
        for i, dolly in enumerate(dolly_effects):
            # if i>0:
            video_path = str(output_path)
            print(f"  Effect {i+1}: type={dolly.effect_type}, start={dolly.start_time}s, duration={dolly.duration}s")
            if dolly.effect_type == DollyEffectType.MANUAL_ZOOM:
                print(f"    Manual zoom at ({dolly.x_coordinate}, {dolly.y_coordinate}) with zoom percent {dolly.zoom_percent}%")
                apply_zoom_effect(
                    input_path=video_path,
                    output_path=str(output_path),
                    zoom_duration=dolly.duration,
                    zoom_start_time=dolly.start_time,
                    zoom_percent=dolly.zoom_percent / 100.0,  # Chuyển đổi sang tỷ lệ
                    center=(dolly.x_coordinate, dolly.y_coordinate),  # Giả sử tọa độ trong khoảng [0, 1]
                    end_effect=dolly.end_time,  # Kết thúc tại thời gian đã chỉ định
                    remove_mode=dolly.end_type.value  # "smooth" hoặc "instant"
                )   
            elif dolly.effect_type == DollyEffectType.AUTO_ZOOM:
                print(f"    Auto zoom with zoom percent {dolly.zoom_percent}%")
                create_face_zoom_video(
                    input_video=video_path,
                    output_video=str(output_path),
                    zoom_type="instant",
                    zoom_start_time=dolly.start_time,
                    zoom_duration=dolly.end_time-   dolly.start_time,
                    zoom_factor=2 - dolly.zoom_percent / 100.0,
                    enable_shake=False,
                    shake_intensity=1,
                    shake_start_delay=0.3
                )
            # elif dolly.effect_type == DollyEffectType.DOUBLE_ZOOM:
            #     print(f"    Double zoom with zoom percent {dolly.zoom_percent}%")
            #     create_face_zoom_video(
            #         input_video=video_path,
            #         output_video=str(output_path),
            #         zoom_type="instant",
            #         zoom_start_time=dolly.start_time,
            #         zoom_duration=dolly.end_time - dolly.start_time,
            #         zoom_factor=1.3,
            #         enable_shake=False,
            #         shake_intensity=2,
            #         shake_start_delay=0.5
            #     )
            #     create_face_zoom_video(
            #         input_video=video_path,
            #         output_video=str(output_path),
            #         zoom_type="instant",
            #         zoom_start_time=dolly.start_time+1.0,
            #         zoom_duration=dolly.end_time-   dolly.start_time - 1.0,
            #         zoom_factor=1.3,
            #         enable_shake=False,
            #         shake_intensity=2,
            #         shake_start_delay=0.5
            #     )



            print("================================")
        # for dolly in dolly_effects or []:
        #     if k!=0:
        #         video_path = str(output_path)
        #     # if dolly.scene_index is not None:
        #     if dolly.effect_type == DollyEffectType.MANUAL_ZOOM:
        #         # if dolly.start_time is None:
        #         #     dolly.start_time = transition_times[0] if transition_times else 0
        #         apply_zoom_effect(
        #             input_path=video_path,
        #             output_path=str(output_path),
        #             zoom_duration=dolly.duration,
        #             zoom_start_time=dolly.start_time,
        #             zoom_percent=dolly.zoom_percent / 100.0,  # Chuyển đổi sang tỷ lệ
        #             center=(dolly.x_coordinate, dolly.y_coordinate),  # Giả sử tọa độ trong khoảng [0, 1]
        #             end_effect=dolly.end_time,  # Kết thúc tại thời gian đã chỉ định
        #             remove_mode=dolly.end_type.value  # "smooth" hoặc "instant"
        #         )   
        #     elif dolly.effect_type == DollyEffectType.AUTO_ZOOM:
        #         create_face_zoom_video(
        #             input_video=video_path,
        #             output_video=str(output_path),
        #             zoom_type="instant",
        #             zoom_start_time=dolly.start_time,
        #             zoom_duration=dolly.duration,
        #             zoom_factor=1.3,
        #             enable_shake=False,
        #             shake_intensity=1,
        #             shake_start_delay=0.3
        #         )
        #     elif dolly.effect_type == DollyEffectType.DOUBLE_ZOOM:
        #         create_face_zoom_video(
        #             input_video=video_path,
        #             output_video=str(output_path),
        #             zoom_type="smooth",
        #             zoom_start_time=dolly.start_time,
        #             zoom_duration=dolly.duration,
        #             zoom_factor=1.6,
        #             enable_shake=True,
        #             shake_intensity=2,
        #             shake_start_delay=0.5
        #         )
        #         create_face_zoom_video(
        #             input_video=video_path,
        #             output_video=str(output_path),
        #             zoom_type="instant",
        #             zoom_start_time=dolly.start_time+1.0,
        #             zoom_duration=dolly.duration-1.0,
        #             zoom_factor=1.3,
        #             enable_shake=False,
        #             shake_intensity=1,
        #             shake_start_delay=0.3
        #         )
        #     k+=1
        print(f"Processing video effects for job {job_id}")
        print(f"Input video: {video_path}")
        print(f"Transition times: {transition_times}")
        print(f"Transition effects: {transition_effects}")
        print(f"Transition durations: {transition_durations}")
        print(f"Dolly effects: {len(dolly_effects or [])} effects")
        print(f"Output will be: {output_path}")
        # import time
        # time.sleep(7)
        print("================================")
        # =======================Test================================
        # self._mock_process_video_sync(video_path, str(output_path), job_id)
        
        return str(output_path)

    def _mock_process_video_sync(self, input_path: str, output_path: str, job_id: str):
  
        import shutil
        import time
        
        time.sleep(2)  # 2 giây để test
        
        shutil.copy2(input_path, output_path)
        print(f"Mock processing completed for job {job_id}")

    def get_video_duration_sync(self, video_path: str) -> float:
        """
        Lấy duration của video - SYNC version để chạy trong thread pool
        """
        
        # TODO: Implement lấy duration thật với subprocess.run()
        # Ví dụ:
        # import subprocess
        # result = subprocess.run([
        #     "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        #     "-of", "csv=p=0", video_path
        # ], capture_output=True, text=True)
        # if result.returncode != 0:
        #     raise RuntimeError(f"FFprobe failed: {result.stderr}")
        # return float(result.stdout.strip())
        
        # Mock duration cho test
        return 60.0  # 60 giây

    async def get_video_duration(self, video_path: str) -> float:
        """
        Lấy duration của video - sử dụng subprocess để không block
        """
        
        # TODO: Implement lấy duration thật với subprocess
        # Ví dụ:
        # process = await asyncio.create_subprocess_exec(
        #     "ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
        #     "-of", "csv=p=0", video_path,
        #     stdout=asyncio.subprocess.PIPE,
        #     stderr=asyncio.subprocess.PIPE
        # )
        # stdout, stderr = await process.communicate()
        # return float(stdout.decode('utf-8').strip())
        
        # Mock duration cho test
        return 60.0  # 60 giây

    def validate_effects_timing(self, 
                              transition_times: List[float],
                              dolly_effects: List[DollyEffect],
                              video_duration: float):
        # Convert dict to DollyEffect objects if needed
        dolly_effects = self._ensure_dolly_objects(dolly_effects)
        
        # Kiểm tra transition times
        for time_point in transition_times:
            if time_point > video_duration:
                raise ValueError(f"Transition time {time_point}s exceeds video duration {video_duration}s")
        
        # Kiểm tra dolly effects
        for dolly in dolly_effects or []:
            if dolly.start_time + dolly.duration > video_duration:
                raise ValueError(f"Dolly effect (start: {dolly.start_time}s, duration: {dolly.duration}s) exceeds video duration {video_duration}s")