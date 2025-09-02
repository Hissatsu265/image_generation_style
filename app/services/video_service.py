import os
import uuid
from pathlib import Path
from typing import List
import subprocess
import json
import random

import aiohttp
import aiofiles
import websockets
import glob
import time
from config import SERVER_COMFYUI,WORKFLOW_INFINITETALK_PATH,BASE_DIR
from PIL import Image
server_address = SERVER_COMFYUI

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
from audio_processing_infinite import trim_video_start,add_silence_to_start
from check_audio_safe import wait_for_audio_ready
from paddvideo import add_green_background,replace_green_screen,crop_green_background,resize_and_pad
# from app.services.create_video_infinitetalk import load_workflow,wait_for_completion,queue_prompt,find_latest_video
import asyncio

padder = ImagePadder()
class VideoService:
    def __init__(self):
        from config import OUTPUT_DIR
        self.output_dir = OUTPUT_DIR

    def generate_output_filename(self) -> str:
        unique_id = str(uuid.uuid4())[:8]
        timestamp = int(asyncio.get_event_loop().time())
        return unique_id, f"video_{timestamp}_{unique_id}.mp4"

    async def create_video(self, image_paths: List[str], prompts: List[str], audio_path: str, resolution: str, job_id: str,background:None) -> str:
        jobid, output_filename = self.generate_output_filename()
        output_path = self.output_dir / output_filename
        # print("fdfsdfsdf")
        try:
            
            from app.services.job_service import job_service
            await job_service.update_job_status(job_id, "processing", progress=30)
            # print("dfsdf")

            list_scene = await run_job(jobid, prompts, image_paths, audio_path, output_path,resolution,background)   

            print(f"Video created successfully: {output_path}")
            print(f"Job ID: {job_id}, Output Path: {output_path}")
            # =====================Test=================================
            # await asyncio.sleep(2)  
            # await job_service.update_job_status(job_id, "processing", progress=60)
            
            # await asyncio.sleep(2)  
            # await job_service.update_job_status(job_id, "processing", progress=90)
            
            # ==== END CODE ====
            if output_path.exists():
                return str(output_path),list_scene
            else:
                raise Exception("Video creation failed - output file not found")
                
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise e
async def run_job(job_id, prompts, cond_images, cond_audio_path,output_path_video,resolution,background):
    print("resolution: ",resolution)
    generate_output_filename = output_path_video
    # print("sdf2")
    list_scene=[]
    if get_audio_duration(cond_audio_path) > 15:
    # if False:
        output_directory = "output_segments"
        os.makedirs(output_directory, exist_ok=True)
        output_paths,durations, result = process_audio_file(cond_audio_path, output_directory)
        results=[]
        last_value=None
        for i, output_path in enumerate(output_paths):
            if i<len(output_paths)-1:
                list_scene.append(get_audio_duration(output_path))
            # ==============Random image for each scene=============
            choices = [x for x in range(len(prompts)) if x != last_value] 
            current_value = random.choice(choices)  # chọn ngẫu nhiên
            # print(current_value)
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
            audiohavesecondatstart = add_silence_to_start(output_path, job_id, duration_ms=0)
            # if wait_for_audio_ready(audiohavesecondatstart, min_size_mb=0.02, max_wait_time=60, min_duration=2.0):
            #     print("Detailed check passed!")
            # print(prompts[current_value])
            # print(cond_images[current_value])
            # print("dfsdfsdfsd:   ", audiohavesecondatstart)
            audiohavesecondatstart=str(BASE_DIR / audiohavesecondatstart)
            print("dfsdfsdfsd:   ", audiohavesecondatstart)
            print(type(audiohavesecondatstart))
            # print(clip_name)
            # print(job_id)
            crop_green_background(cond_images[current_value], str(cond_images[current_value].replace(".png", "_crop.png")), margin=0.04)
            resize_and_pad(str(cond_images[current_value].replace(".png", "_crop.png")), str(cond_images[current_value].replace(".png", "_pad.png")))
            
            output=await generate_video_cmd(
                prompt=prompts[current_value],
                cond_image=str(cond_images[current_value].replace(".png", "_pad.png")),# 
                cond_audio_path=audiohavesecondatstart, 
                output_path=clip_name,
                job_id=job_id,
                resolution=resolution
            )
            trim_video_start(clip_name, duration=0.5)
            output_file=cut_video(clip_name, get_audio_duration(output_path)-0.5) 
            results.append(output_file)
            try:
                os.remove(output_path)
                os.remove(clip_name)
                os.remove(audiohavesecondatstart)
            except Exception as e:
                print(f"❌ Error removing temporary file {output_path}: {str(e)}")

        concat_name=os.path.join(os.getcwd(), f"{job_id}_concat_{i}.mp4")
        output_file1 = concat_videos(results, concat_name)
        from merge_video_audio import replace_audio_trimmed
        output_file = replace_audio_trimmed(output_file1,cond_audio_path,output_path_video)
        replace_green_screen(
            video_path=str(output_path_video),
            background_path=background,  
        )
        try:
            os.remove(output_file1)
            for file in results:
                os.remove(file)
        except Exception as e:
            print(f"❌ Error removing temporary files: {str(e)}")
        return list_scene
    else:
        audiohavesecondatstart = add_silence_to_start(cond_audio_path, job_id, duration_ms=500)
        generate_output_filename=os.path.join(os.getcwd(), f"{job_id}_noaudio.mp4")
        if wait_for_audio_ready(audiohavesecondatstart, min_size_mb=0.02, max_wait_time=60, min_duration=2.0):
            print("Detailed check passed!")
        crop_green_background(cond_images[0], str(cond_images[0].replace(".png", "_crop.png")), margin=0.04)
        resize_and_pad(str(cond_images[0].replace(".png", "_crop.png")), str(cond_images[0].replace(".png", "_pad.png")))
        print("sdf3")
        print(str(cond_images[0].replace(".png", "_pad.png")))
        output=await generate_video_cmd(
            prompt=prompts[0], 
            cond_image=str(cond_images[0].replace(".png", "_pad.png")), 
            cond_audio_path=audiohavesecondatstart, 
            output_path=generate_output_filename,
            job_id=job_id,
            resolution=resolution
        )  
        # print("sdf4")
        from merge_video_audio import replace_audio_trimmed
        # print("dfdsf")
        # print(generate_output_filename)
        tempt=trim_video_start(generate_output_filename, duration=0.5)
        output_file = replace_audio_trimmed(generate_output_filename,cond_audio_path,output_path_video)
        try:
            os.remove(generate_output_filename)
            os.remove(audiohavesecondatstart)
        except Exception as e:
            print(f"❌ Error removing temporary files: {str(e)}")
        # print("sdf5")
        replace_green_screen(
            video_path=str(output_path_video),
            background_path=background,  
        )
        # print("sdf6")

        return list_scene
# ============================================================================================


async def load_workflow(path="workflow.json"):
    """Load workflow file bất đồng bộ"""
    async with aiofiles.open(path, "r", encoding='utf-8') as f:
        content = await f.read()
        return json.loads(content)

async def queue_prompt(workflow):
    """Gửi workflow đến ComfyUI server bất đồng bộ"""
    client_id = str(uuid.uuid4())
    
    payload = {
        "prompt": workflow, 
        "client_id": client_id
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{server_address}/prompt",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                result["client_id"] = client_id
                return result
            else:
                raise Exception(f"Failed to queue prompt: {response.status}")

async def wait_for_completion(prompt_id, client_id):
    print(f"Đang kết nối WebSocket để theo dõi tiến trình...")
    
    websocket_url = f"ws://{server_address}/ws?clientId={client_id}"
    
    try:
        async with websockets.connect(websocket_url) as websocket:
            print("✅ Đã kết nối WebSocket")
            
            total_nodes = 0
            completed_nodes = 0
            
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    
                    if isinstance(msg, str):
                        data = json.loads(msg)                        
                        print(f"📨 Nhận message: {data.get('type', 'unknown')}")
                        
                        if data["type"] == "execution_start":
                            print(f"🚀 Bắt đầu thực thi workflow với prompt_id: {data.get('data', {}).get('prompt_id')}")
                        
                        elif data["type"] == "executing":
                            node_id = data["data"]["node"]
                            # current_prompt_id = data["data"]["prompt_id"]
                            current_prompt_id = data.get("data", {}).get("prompt_id")

                            if current_prompt_id == prompt_id:
                                if node_id is None:
                                    print("🎉 Workflow hoàn thành!")
                                    return True
                                else:
                                    completed_nodes += 1
                                    print(f"⚙️  Đang xử lý node: {node_id} ({completed_nodes} nodes đã hoàn thành)")
                        
                        elif data["type"] == "progress":
                            progress_data = data.get("data", {})
                            value = progress_data.get("value", 0)
                            max_value = progress_data.get("max", 100)
                            node = progress_data.get("node")
                            percentage = (value / max_value * 100) if max_value > 0 else 0
                            print(f"📊 Node {node}: {value}/{max_value} ({percentage:.1f}%)")
                        
                        elif data["type"] == "execution_error":
                            print(f"❌ Lỗi thực thi: {data}")
                            return False
                            
                        elif data["type"] == "execution_cached":
                            cached_nodes = data.get("data", {}).get("nodes", [])
                            print(f"💾 {len(cached_nodes)} nodes được cache")
                
                except asyncio.TimeoutError:
                    print("⏰ WebSocket timeout, tiếp tục đợi...")
                    continue
                except Exception as e:
                    print(f"❌ Lỗi WebSocket: {e}")
                    break
                    
    except Exception as e:
        print(f"❌ Không thể kết nối WebSocket: {e}")
        print("🔄 Fallback: Kiểm tra file output định kỳ...")
        
        # Fallback: kiểm tra file output bất đồng bộ
        return await wait_for_completion_fallback(prompt_id)

async def wait_for_completion_fallback(prompt_id):
    start_time = time.time()
    
    while True:
        await asyncio.sleep(2)  # Chờ bất đồng bộ
        
        video_path = await find_latest_video("my_custom_video")
        if video_path and os.path.exists(video_path):
            file_time = os.path.getmtime(video_path)
            if file_time > start_time:
                print("✅ Phát hiện video mới được tạo!")
                return True
        
        if time.time() - start_time > 300:
            print("⏰ Timeout: Quá 5 phút không thấy kết quả")
            return False

# ========== Hàm tìm video mới nhất bất đồng bộ ==========
async def find_latest_video(prefix, output_dir=str(BASE_DIR / "ComfyUI/output")):    
    # Chạy file operations trong executor để không block event loop
    def _find_files():
        patterns = [
            f"{prefix}*.mp4",
            f"{prefix}*audio*.mp4", 
            f"{prefix}_*-audio.mp4"
        ]
        
        all_files = []
        for pattern in patterns:
            files = glob.glob(os.path.join(output_dir, pattern))
            all_files.extend(files)
        
        if not all_files:
            print(f"🔍 Không tìm thấy file nào với prefix '{prefix}' trong {output_dir}")
            # List tất cả file .mp4 để debug
            all_mp4 = glob.glob(os.path.join(output_dir, "*.mp4"))
            if all_mp4:
                print(f"📁 Các file .mp4 hiện có:")
                for f in sorted(all_mp4, key=os.path.getmtime, reverse=True)[:5]:
                    print(f"   {f} (modified: {time.ctime(os.path.getmtime(f))})")
            return None
        
        latest_file = max(all_files, key=os.path.getmtime)
        print(f"📁 Tìm thấy file mới nhất: {latest_file}")
        return latest_file
    
    # Chạy trong executor
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _find_files)

# ========== Hàm chính được cập nhật ==========
async def generate_video_cmd(prompt, cond_image, cond_audio_path, output_path, job_id,resolution):

    print("🔄 Đang load workflow...")

    workflow = await load_workflow(str(BASE_DIR) + "/" + WORKFLOW_INFINITETALK_PATH)
    # ============================================================
    # await crop_green_background(cond_image, str(cond_image.replace(".png", "_crop.png")))
    workflow["203"]["inputs"]["image"] = cond_image
    # =============================================================
    workflow["125"]["inputs"]["audio"] = cond_audio_path
    
    if prompt.strip() == "" or prompt is None or prompt == "none":
        workflow["135"]["inputs"]["positive_prompt"] = "Mouth moves in sync with speech. A person is sitting in a side-facing position, with their face turned toward the left side of the frame and the eyes look naturally forward in that left-facing direction without shifting. Speaking naturally, as if having a conversation. He always kept his posture and gaze straight without turning his head."    
    else:
        workflow["135"]["inputs"]["positive_prompt"] = prompt
        
    workflow["135"]["inputs"]["negative_prompt"] = "change perspective, bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    wf_h=448
    wf_w=448
    if resolution == "1080x1920":
        wf_w = 1080
        wf_h = 1920
    elif resolution=="1920x1080":
        wf_w = 1920
        wf_h = 1080
    elif resolution=="720x1280":
        wf_w = 720
        wf_h = 1280
        # workflow["208"]["inputs"]["frame_window_size"] = 41
    elif resolution=="480x854": 
        wf_w = 480
        wf_h = 854
    elif resolution=="854x480": 
        wf_w = 854
        wf_h = 480
    elif resolution=="1280x720":    
        wf_w = 1280
        wf_h = 720 
      
        # workflow["208"]["inputs"]["frame_window_size"] = 41
    img = Image.open(cond_image)
    width_real, height_real = img.size
    workflow["211"]["inputs"]["value"] = width_real
    workflow["212"]["inputs"]["value"] = height_real

    # workflow["211"]["inputs"]["value"] = 608
    # workflow["212"]["inputs"]["value"] = 608
    img.close()

    prefix = job_id
    workflow["131"]["inputs"]["filename_prefix"] = prefix

    print("📤 Đang gửi workflow đến ComfyUI...")

    resp = await queue_prompt(workflow)
    prompt_id = resp["prompt_id"]
    client_id = resp["client_id"]
    print(f"✅ Đã gửi workflow! Prompt ID: {prompt_id}")
    
    success = await wait_for_completion(prompt_id, client_id)
    
    if not success:
        print("❌ Workflow thất bại")
        return None

    print("🔍 Đang tìm video đã tạo...")
    video_path = await find_latest_video(prefix)
    
    if video_path:
        # await delete_file_async(str(cond_image.replace(".png", "_crop.png")))  
        await add_green_background(video_path, str(video_path.replace(".mp4", "_greenbg.mp4")), target_w=wf_w, target_h=wf_h)
        # await delete_file_async(video_path)
        video_path = str(video_path.replace(".mp4", "_greenbg.mp4"))
        print(f"🎬 Video được tạo tại: {video_path}")
        file_size = os.path.getsize(video_path)
        print(f"📏 Kích thước file: {file_size / (1024*1024):.2f} MB")
        
        await move_file_async(str(video_path),str(output_path))
        print("dfsdfs-----")
        return output_path
    else:
        print("❌ Không tìm thấy video")
        return None

async def move_file_async(src_path, dst_path):
    def move_file():
        os.rename(src_path, dst_path)
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, move_file)
async def delete_file_async(file_path: str):
    def delete_file():
        if os.path.exists(file_path):
            os.remove(file_path)
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, delete_file)
