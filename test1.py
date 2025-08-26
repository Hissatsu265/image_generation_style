import os
import uuid
from pathlib import Path
from typing import List

from app.models.schemas import TransitionEffect, DollyEffect, DollyEffectType, DollyEndType
from animation.full_transition_effect import apply_effect
from animation.zoomin_at_one_point import apply_zoom_effect
from animation.zoomin import safe_create_face_zoom_video

import asyncio

dolly_effects= [
      {
        "scene_index": 0,
        "start_time": 1.5,
        "duration": 2.0,
        "zoom_percent": 50,
        "effect_type": "auto_zoom",
        "end_time": 5.0,
        "end_type": "smooth"
      },
      {
        "scene_index": 0,
        "start_time": 5,
        "duration": 2.0,
        "zoom_percent": 50,
        "effect_type": "auto_zoom",
        "end_time": 5.0,
        "end_type": "smooth"
      },

    #   {
    #     "scene_index": 1,
    #     "start_time": 3.0,
    #     "duration": 1.5,
    #     "zoom_percent": 50,
    #     "effect_type": "manual_zoom",
    #     "x_coordinate": 100,
    #     "y_coordinate": 100,
    #     "end_time": 4.0,
    #     "end_type": "instant"
    #   },
    #   {
    #     "scene_index": 1,
    #     "start_time": 5.0,
    #     "duration": 1.5,
    #     "zoom_percent": 50,
    #     "effect_type": "manual_zoom",
    #     "x_coordinate": 200,
    #     "y_coordinate": 200,
    #     "end_time": 9.0,
    #     "end_type": "instant"
    #   },

    ]

video_path="/workspace/marketing-video-ai/55c95f56_clip_0_cut_11.49s.mp4"
job_id="sfsgagfgdfgdfg"
# if len(transition_times) != len(transition_effects) != len(transition_durations):
#             raise ValueError("transition_times, transition_effects, transition_durations must have same length")
        
#         if not os.path.exists(video_path):
#             raise FileNotFoundError(f"Input video not found: {video_path}")
        
#         # Convert dict to DollyEffect objects if needed
#         dolly_effects = self._ensure_dolly_objects(dolly_effects)
        
        # Tạo output path
from config import OUTPUT_DIR
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(exist_ok=True)

output_filename = f"effect_{job_id or uuid.uuid4().hex}.mp4"
output_path = output_dir / output_filename

# outputpath_raw_eff=output_dir / f"raw_effect_{job_id or uuid.uuid4().hex}_step.mp4"
for i, dolly in enumerate(dolly_effects):
    if i > 0:
        video_path = str(output_path)
    outputpath_raw_eff=output_dir / f"raw_effect_{job_id or uuid.uuid4().hex}_step{i+1}.mp4"

    print(f"  Effect {i+1}: type={dolly['effect_type']}, start={dolly['start_time']}s, duration={dolly['duration']}s")

    if dolly["effect_type"] == "manual_zoom":
        print(f"    Manual zoom at ({dolly['x_coordinate']}, {dolly['y_coordinate']}) with zoom percent {dolly['zoom_percent']}%")
        apply_zoom_effect(
            input_path=video_path,
            output_path=str(outputpath_raw_eff),
            zoom_duration=dolly["duration"],
            zoom_start_time=dolly["start_time"],
            zoom_percent=dolly["zoom_percent"] / 100.0,
            center=(dolly["x_coordinate"], dolly["y_coordinate"]),
            end_effect=dolly["end_time"],
            remove_mode=dolly["end_type"]
        )
    elif dolly["effect_type"] == "auto_zoom":
        print(f"  Auto zoom with zoom percent {dolly['zoom_percent']}%")
        safe_create_face_zoom_video(
            input_video=video_path,
            output_video=str(outputpath_raw_eff),
            zoom_type="instant",
            zoom_start_time=dolly["start_time"],
            zoom_duration=dolly["end_time"] - dolly["start_time"],
            zoom_factor=2 - dolly["zoom_percent"] / 100.0,
            enable_shake=False,
            shake_intensity=1,
            shake_start_delay=0.3
        )
    # if os.path.exists(output_path):
    #     os.remove(output_path)
    #     print(f"Đã xóa file: {output_path}")
    # else:
    #     print("File không tồn tại")
    # os.rename(outputpath_raw_eff, output_path)
    
# print(f"Processing video effects for job {job_id}")
# print(f"Input video: {video_path}")
# print(f"Transition times: {transition_times}")
# print(f"Transition effects: {transition_effects}")
# print(f"Transition durations: {transition_durations}")
print(f"Dolly effects: {len(dolly_effects or [])} effects")
print(f"Output will be: {output_path}")
# import time
# time.sleep(7)
print("================================")