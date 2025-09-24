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
import shutil

import asyncio
from directus.file_upload import Uploadfile_directus
class VideoService:
    def __init__(self):
        from config import OUTPUT_DIR
        self.output_dir = OUTPUT_DIR

    def generate_output_filename(self) -> str:
        unique_id = str(uuid.uuid4())[:8]
        timestamp = int(asyncio.get_event_loop().time())
        return unique_id, f"video_{timestamp}_{unique_id}_1.png",f"video_{timestamp}_{unique_id}_2.png"

    async def create_video(self, image_paths: List[str], prompts: List[str], style: str, resolution: str, job_id: str) -> str:
        jobid, output_filename1,output_filename2 = self.generate_output_filename()
        # output_path1 = self.output_dir / output_filename1
        # output_path2 = self.output_dir / output_filename2
        # print("===============================")
        # print("hi: ",output_path1)
        # print("hi: ",output_path2)
        print("=============================================")
        print("promtp",prompts)
        print("image_paths",image_paths)
        print("style",style)
        print("resolution",resolution)
        print("===============================")

        try:
            from app.services.job_service import job_service
            await job_service.update_job_status(job_id, "processing", progress=99)
            # print("60")
            img = await run_job(jobid, prompts, image_paths, resolution,style)   
            # for path in img:
            #     print(path,"=====heh")
            # print("61")
            list_img=[]
            for path in img:
                print(path,"=====heh")
                path_directus= Uploadfile_directus(str(path))
                list_img.append(path_directus)

            folder = os.path.dirname(img[0])
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"Đã xóa thư mục: {folder}")
            else:
                print(f"Thư mục không tồn tại: {folder}")

            return list_img
            # if path_directus is not None and output_path.exists() :
            #     print(f"Video upload successfully: {path_directus}")
            #     print(f"Job ID: {job_id}, Output Path: {path_directus}")
            #     os.remove(str(output_path))
            #     return str(path_directus)
            # else:
            #     raise Exception("Cannot upload video to Directus or Video creation failed - output file not found")
           
        except Exception as e:
            # if output_path1.exists():
            #     output_path1.unlink()
            raise e
async def run_job(job_id, prompts, cond_images,resolution,style):
    current_value=0
    img=await generate_video_cmd(
                prompt=prompts[current_value],
                cond_image=cond_images[current_value], 
                style=style, 
                job_id=job_id,
                resolution=resolution
            )
    # try:
    #     os.remove(cond_images[current_value])
    # except Exception as e:
    #     print(f"❌ Error removing temporary files: {str(e)}")
    return  img


# ============================================================================================

import asyncio
import signal

async def start_comfyui():
    process = await asyncio.create_subprocess_exec(
        "python3", "main.py",
        cwd=str(BASE_DIR / "ComfyUI"),  # chỗ chứa main.py
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    print("🚀 ComfyUI started (PID:", process.pid, ")")
    return process
async def stop_comfyui(process):
    if process and process.returncode is None:
        print("🛑 Stopping ComfyUI...")
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            print("⚠️ Force killing ComfyUI...")
            process.kill()
            await process.wait()
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
        
        video_path = await find_images_by_id("my_custom_video")
        if video_path and os.path.exists(video_path):
            file_time = os.path.getmtime(video_path)
            if file_time > start_time:
                print("✅ Phát hiện video mới được tạo!")
                return True
        
        if time.time() - start_time > 300:
            print("⏰ Timeout: Quá 5 phút không thấy kết quả")
            return False

# ========== Hàm tìm video mới nhất bất đồng bộ ==========
# async def find_latest_video(prefix, output_dir=str(BASE_DIR / "ComfyUI/output")):    
#     # Chạy file operations trong executor để không block event loop
#     def _find_files():
#         patterns = [
#             f"{prefix}*audio*.mp4", 
#             f"{prefix}_*-audio.mp4"
#         ]
        
#         all_files = []
#         for pattern in patterns:
#             files = glob.glob(os.path.join(output_dir, pattern))
#             all_files.extend(files)
        
#         if not all_files:
#             print(f"🔍 Không tìm thấy file nào với prefix '{prefix}' trong {output_dir}")
#             # List tất cả file .mp4 để debug
#             all_mp4 = glob.glob(os.path.join(output_dir, "*.mp4"))
#             if all_mp4:
#                 print(f"📁 Các file .mp4 hiện có:")
#                 for f in sorted(all_mp4, key=os.path.getmtime, reverse=True)[:5]:
#                     print(f"   {f} (modified: {time.ctime(os.path.getmtime(f))})")
#             return None
        
#         latest_file = max(all_files, key=os.path.getmtime)
#         print(f"📁 Tìm thấy file mới nhất: {latest_file}")
#         return latest_file
    
#     # Chạy trong executor
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(None, _find_files)
# ============================================================
async def find_images_by_id(image_id, output_dir=str(BASE_DIR / "ComfyUI/output")):
# async def find_images_by_id(image_id, output_dir="/home/toan/image_gen/ComfyUI/output"):
    # output_dir="/home/toan/image_gen/ComfyUI/output"
    def _find_files():
        target_dir = os.path.join(output_dir, str(image_id))
        if not os.path.exists(target_dir):
            print(f"❌ Không tìm thấy thư mục: {target_dir}")
            return []
        print("===============")
        pattern = os.path.join(target_dir, f"{image_id}*.png")
        files = glob.glob(pattern)
        print(pattern)
        if not files:
            print(f"🔍 Không tìm thấy file nào với id '{image_id}' trong {target_dir}")
            return []
        files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
        return files_sorted[:2]

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _find_files)

# ========== Hàm chính được cập nhật ==========
async def generate_video_cmd(prompt, cond_image, style, job_id,resolution):
    # comfy_process = await start_comfyui()
    # await asyncio.sleep(10)  
    try:

        print("🔄 Đang load workflow...")
        workflow = await load_workflow(str(BASE_DIR) + "/" + WORKFLOW_INFINITETALK_PATH)

        style_prefix = {
            "realistic": (
                "A highly detailed, ultra-realistic, high-resolution photograph of "
            ),
            "anime": (
                "An expressive, vibrant anime-style illustration of "
            ),
            "cartoon": (
                "A bold, colorful, exaggerated cartoon drawing of "
            ),
            "vintage": (
                "A nostalgic, retro, vintage-style photograph with warm tones of "
            ),
            "minimal": (
                "A clean, simple, minimalistic flat design of "
            ),
            "artistic": (
                "An imaginative, creative, artistic painting with unique textures of "
            )
        }
        rule = style_prefix.get(style, "")
        workflow["6"]["inputs"]["text"] = rule + prompt
        print(rule + prompt[0])

        wf_h=1024
        wf_w=1024
        if resolution == "1:1":
            wf_w = 1024
            wf_h = 1024
        elif resolution=="16:9":
            wf_w = 1280
            wf_h = 720
        elif resolution=="9:16":
            wf_w = 720
            wf_h = 1280
        
        workflow["162"]["inputs"]["width"] = wf_w
        workflow["162"]["inputs"]["height"] = wf_h
        

# ===========================================================
        prefix = job_id+"/"+job_id
        workflow["171"]["inputs"]["filename_prefix"] = prefix
        print("📤 Đang gửi workflow đến ComfyUI...")
        resp = await queue_prompt(workflow)

        # print("resp",resp)
        prompt_id = resp["prompt_id"]
        client_id = resp["client_id"]
        print(f"✅ Đã gửi workflow! Prompt ID: {prompt_id}")
        success = await wait_for_completion(prompt_id, client_id)
        
        if not success:
            print("❌ Workflow thất bại")
            return None

        print("🔍 Đang tìm image đã tạo...")
        img = await find_images_by_id(job_id)
        return img
    finally:
        # await stop_comfyui(comfy_process)
        print("done")

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