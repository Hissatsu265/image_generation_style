import asyncio
import os
import uuid
from pathlib import Path
from typing import List
import subprocess
import json
import random

from divide_audio import process_audio_file
from multiperson_imageedit import crop_with_ratio_expansion
from merge_video import concat_videos
from take_lastframe import save_last_frame
from cut_video import cut_video,cut_audio,cut_audio_from_time
from audio_duration import get_audio_duration
from add03seconds import add_silence_to_audio
from animation.animation_decision import select_peak_segment
from animation.zoomin import create_face_zoom_video
from keepratio import ImagePadder
import asyncio

padder = ImagePadder()
class VideoService:
    def __init__(self):
        from config import OUTPUT_DIR
        self.output_dir = OUTPUT_DIR

    def generate_output_filename(self) -> str:
        """Tạo tên file output unique"""
        unique_id = str(uuid.uuid4())[:8]
        timestamp = int(asyncio.get_event_loop().time())
        return unique_id, f"video_{timestamp}_{unique_id}.mp4"

    async def create_video(self, image_paths: List[str], prompts: List[str], audio_path: str, resolution: str, job_id: str) -> str:
        jobid, output_filename = self.generate_output_filename()
        output_path = self.output_dir / output_filename
        print("fdfsdfsdf")
        try:
            
            from app.services.job_service import job_service
            await job_service.update_job_status(job_id, "processing", progress=30)
            print("dfsdf")
            await run_job(jobid, prompts, image_paths, audio_path, output_path)   
            print(f"Video created successfully: {output_path}")
            print(f"Job ID: {job_id}, Output Path: {output_path}")

            # await asyncio.sleep(2)  
            # await job_service.update_job_status(job_id, "processing", progress=60)
            
            # await asyncio.sleep(2)  
            # await job_service.update_job_status(job_id, "processing", progress=90)
            
            # ==== END CODE ====
            if output_path.exists():
                return str(output_path)
            else:
                raise Exception("Video creation failed - output file not found")
                
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise e
async def run_job(job_id, prompts, cond_images, cond_audio_path,output_path_video):
    print("sdf")
    generate_output_filename = output_path_video
    print("sdf")
    if get_audio_duration(cond_audio_path) > 15:
        output_directory = "output_segments"
        os.makedirs(output_directory, exist_ok=True)
        output_paths,durations, result = process_audio_file(cond_audio_path, output_directory)
        results=[]
        last_value=None
        for i, output_path in enumerate(output_paths):
            # ==============Random image for each scene=============
            choices = [x for x in range(len(prompts)) if x != last_value]  # loại bỏ giá trị lần trước
            current_value = random.choice(choices)  # chọn ngẫu nhiên
            print(current_value)
            last_value = current_value  # lưu 
            # ===============================================================================
            print(f"Audio segment {i+1}: {output_path} (Duration: {durations[i]}s)")

            print(cond_images)
            print(f"Image: {cond_images[current_value]}")
            print(f"Prompt: {prompts[current_value]}")
            # clip_name03second=os.path.join(os.getcwd(), f"{job_id}_clip03second_{i}.mp4")
            clip_name=os.path.join(os.getcwd(), f"{job_id}_clip_{i}.mp4")
            # print(i)
            # print(type(current_value))
            output=await generate_video_cmd(
                prompt=prompts[current_value],
                cond_image=cond_images[current_value],# 
                cond_audio_path=output_path, 
                output_path=clip_name,
                job_id=job_id
            )
            output_file=cut_video(clip_name, durations[i]-0.3) 
            results.append(output_file)
            try:
                os.remove(output_path)
            except Exception as e:
                print(f"❌ Error removing temporary file {output_path}: {str(e)}")

        concat_name=os.path.join(os.getcwd(), f"{job_id}_concat_{i}.mp4")
        output_file1 = concat_videos(results, concat_name)
        from merge_video_audio import replace_audio_trimmed
        output_file = replace_audio_trimmed(output_file1,cond_audio_path,output_path_video)
        try:
            os.remove(output_file1)
            for file in results:
                os.remove(file)
        except Exception as e:
            print(f"❌ Error removing temporary files: {str(e)}")
        return True
    else:
        generate_output_filename=os.path.join(os.getcwd(), f"{job_id}_noaudio.mp4")
        output=await generate_video_cmd(
            prompt=prompts[0], 
            cond_image=cond_images[0], 
            cond_audio_path=cond_audio_path, 
            output_path=generate_output_filename,
            job_id=job_id
        )  
        from merge_video_audio import replace_audio_trimmed
        output_file = replace_audio_trimmed(generate_output_filename,cond_audio_path,output_path_video)
        try:
            os.remove(generate_output_filename)
        except Exception as e:
            print(f"❌ Error removing temporary files: {str(e)}")
        return True
async def generate_video_cmd(prompt, cond_image, cond_audio_path,output_path,job_id):
    print("sdf2")
    generate_output_filename = output_path
    print(generate_output_filename)
    # json_filename = generate_output_filename.replace(".mp4", ".json")
    json_filename=os.path.join(os.getcwd(), f"{job_id}_filenamejson.json")
    print("prompt")
    print(cond_image)
    print(cond_audio_path)
    print(output_path)
    print(job_id)
    # ======================================================
    json_filenamepadder = os.path.join(os.getcwd(), f"{job_id}_padder.json")
    image_namepadder = os.path.join(os.getcwd(), f"{job_id}_padder.jpg")
    video_namepadder = os.path.join(os.getcwd(), f"{job_id}_padder.mp4")
    print(json_filenamepadder)
    print(image_namepadder)
    print(video_namepadder)
    k="627"
    # if user_input['resolution'] == 'multitalk-720': k="960"
    info,output_path1 = padder.pad_image(
        image_path=cond_image,
        ratio_type=k, 
        output_path=image_namepadder,
        info_path=json_filenamepadder
    )

    json_data = {
        "prompt": prompt,
        "cond_image": image_namepadder,
        "cond_audio": {
            "person1": cond_audio_path
        }
    }
    # =========================================================
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    print("sdfsdfsdfsdf")    
    cmd = [
        "python", "-u", 
        "generate_multitalk.py",
        "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", "weights/chinese-wav2vec2-base",
        "--input_json", json_filename,
        "--quant", "int8",
        "--quant_dir", "weights/MeiGen-MultiTalk",
        "--lora_dir", "weights/MeiGen-MultiTalk/quant_models/quant_model_int8_FusionX.safetensors",
        "--sample_text_guide_scale", "1.0",
        "--sample_audio_guide_scale", "2.0",
        "--sample_steps", "8",
        "--size", "multitalk-480",#cần sửa resolution sau này
        "--mode", "streaming",
        "--num_persistent_param_in_dit","0",
        "--save_file",video_namepadder.replace(".mp4", ""), 
        "--sample_shift", "2"
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # stdout, stderr = await process.communicate()
    async def handle_output():
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            output = line.decode().strip()
            if output:
                print(f"📝 {output}")

    async def handle_error():
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            error = line.decode().strip()
            if error:
                print(f"⚠️ {error}")

    # Chạy đồng thời
    await asyncio.gather(
        handle_output(),
        handle_error(),
        process.wait()
    )


    if process.returncode != 0:
        raise RuntimeError(f"Video creation failed: {stderr.decode()}")
    print("========================================================================")

    restored_video = padder.restore_video_ratio(
                                video_path=video_namepadder,
                                padding_info_path=json_filenamepadder,
                                output_path=output_path
                            )
    try:
        os.remove(json_filename)
        os.remove(json_filenamepadder)
        os.remove(image_namepadder)
        os.remove(video_namepadder)
    except Exception as e:
        print(f"❌ Error removing temporary files: {str(e)}")

    return output_path