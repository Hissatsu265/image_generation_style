import json, urllib.request, websocket, io, os, glob
import uuid
import time
from config import SERVER_COMFYUI
server_address = SERVER_COMFYUI

def load_workflow(path="workflow.json"):
    with open(path, "r") as f:
        return json.load(f)

def queue_prompt(workflow):
    # Thêm client_id để track workflow
    client_id = str(uuid.uuid4())
    data = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{server_address}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        response = json.loads(resp.read())
        response["client_id"] = client_id
        return response

def wait_for_completion(prompt_id, client_id):
    print(f"Đang kết nối WebSocket để theo dõi tiến trình...")
    ws = websocket.WebSocket()
    
    try:
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
        print("✅ Đã kết nối WebSocket")
        
        total_nodes = 0
        completed_nodes = 0
        
        while True:
            try:
                msg = ws.recv()
                if isinstance(msg, str):
                    data = json.loads(msg)
                    
                    # In ra tất cả messages để debug
                    print(f"📨 Nhận message: {data.get('type', 'unknown')}")
                    
                    if data["type"] == "execution_start":
                        print(f"🚀 Bắt đầu thực thi workflow với prompt_id: {data.get('data', {}).get('prompt_id')}")
                    
                    elif data["type"] == "executing":
                        node_id = data["data"]["node"]
                        current_prompt_id = data["data"]["prompt_id"]
                        
                        if current_prompt_id == prompt_id:
                            if node_id is None:
                                print("🎉 Workflow hoàn thành!")
                                break
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
                        break
                        
                    elif data["type"] == "execution_cached":
                        cached_nodes = data.get("data", {}).get("nodes", [])
                        print(f"💾 {len(cached_nodes)} nodes được cache")
                        
            except websocket.WebSocketTimeoutException:
                print("⏰ WebSocket timeout, tiếp tục đợi...")
                continue
            except Exception as e:
                print(f"❌ Lỗi WebSocket: {e}")
                break
                
    except Exception as e:
        print(f"❌ Không thể kết nối WebSocket: {e}")
        print("🔄 Fallback: Kiểm tra file output định kỳ...")
        
        # Fallback: kiểm tra file output mỗi 2 giây
        start_time = time.time()
        while True:
            time.sleep(2)
            video_path = find_latest_video("my_custom_video")
            if video_path and os.path.exists(video_path):
                # Kiểm tra xem file có được tạo sau khi bắt đầu workflow không
                file_time = os.path.getmtime(video_path)
                if file_time > start_time:
                    print("✅ Phát hiện video mới được tạo!")
                    break
            
            # Timeout sau 5 phút
            if time.time() - start_time > 300:
                print("⏰ Timeout: Quá 5 phút không thấy kết quả")
                break
                
    finally:
        try:
            ws.close()
        except:
            pass

def find_latest_video(prefix, output_dir="/root/ComfyUI/output"):
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

# ==== MAIN =====================================================================
# =========================================================================
# print("🔄 Đang load workflow...")
# workflow = load_workflow("/root/wanvideo_infinitetalk_single_example_19_8 (1).json")

# # Update input nodes
# workflow["203"]["inputs"]["image"] = "/root/marketing-video-ai/girl_green/1.png"
# workflow["125"]["inputs"]["audio"] = "/root/marketing-video-ai/audio/english_girl_3s.wav"
# workflow["135"]["inputs"]["positive_prompt"] = "the girl is talking"
# workflow["211"]["inputs"]["value"] = 512
# workflow["212"]["inputs"]["value"] = 512

# # Update video output node
# prefix = "my_custom_video"
# workflow["131"]["inputs"]["filename_prefix"] = prefix

# print("📤 Đang gửi workflow đến ComfyUI...")

# # Queue workflow
# resp = queue_prompt(workflow)
# prompt_id = resp["prompt_id"]
# client_id = resp["client_id"]
# print(f"✅ Đã gửi workflow! Prompt ID: {prompt_id}")

# # Wait until workflow finished
# wait_for_completion(prompt_id, client_id)

# # Find generated video file
# print("🔍 Đang tìm video đã tạo...")
# video_path = find_latest_video(prefix)
# if video_path:
#     print(f"🎬 Video được tạo tại: {video_path}")
#     # Kiểm tra kích thước file
#     file_size = os.path.getsize(video_path)
#     print(f"📏 Kích thước file: {file_size / (1024*1024):.2f} MB")
# else:
#     print("❌ Không tìm thấy video")