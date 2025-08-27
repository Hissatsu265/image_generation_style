import os
import time
import cv2

def wait_for_file_ready(file_path, min_size_mb=0.1, max_wait_time=60, check_interval=1):
    """
    Kiểm tra file đã sẵn sàng để sử dụng
    
    Args:
        file_path: đường dẫn file cần kiểm tra
        min_size_mb: kích thước tối thiểu của file (MB)
        max_wait_time: thời gian chờ tối đa (giây)
        check_interval: khoảng thời gian giữa các lần kiểm tra (giây)
    
    Returns:
        bool: True nếu file sẵn sàng, False nếu timeout
    """
    print(f"🔍 Đang kiểm tra file: {file_path}")
    start_time = time.time()
    min_size_bytes = min_size_mb * 1024 * 1024
    last_size = 0
    stable_count = 0
    
    while time.time() - start_time < max_wait_time:
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            print(f"⏳ File chưa tồn tại. Chờ {check_interval}s...")
            time.sleep(check_interval)
            continue
        
        try:
            # Kiểm tra kích thước file
            current_size = os.path.getsize(file_path)
            print(f"📏 Kích thước file hiện tại: {current_size / (1024*1024):.2f} MB")
            
            # Kiểm tra file có đủ kích thước tối thiểu không
            if current_size < min_size_bytes:
                print(f"⚠️ File chưa đủ kích thước tối thiểu ({min_size_mb} MB). Chờ...")
                time.sleep(check_interval)
                continue
            
            # Kiểm tra file có đang được ghi không (kích thước ổn định)
            if current_size == last_size:
                stable_count += 1
                if stable_count >= 3:  # File ổn định trong 3 lần kiểm tra
                    print("📊 File ổn định, tiến hành kiểm tra tính toàn vẹn...")
                    
                    # Kiểm tra file có thể đọc được không
                    try:
                        cap = cv2.VideoCapture(file_path)
                        if cap.isOpened():
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                            
                            if frame_count > 0 and fps > 0:
                                print(f"✅ File hợp lệ - Frames: {frame_count}, FPS: {fps}")
                                return True
                            else:
                                print("❌ File video không hợp lệ")
                        else:
                            print("❌ Không thể mở file video")
                    except Exception as e:
                        print(f"❌ Lỗi khi kiểm tra file: {e}")
                    
                    time.sleep(check_interval)
            else:
                stable_count = 0
                last_size = current_size
                print(f"🔄 File đang thay đổi kích thước...")
                time.sleep(check_interval)
                
        except Exception as e:
            print(f"⚠️ Lỗi khi kiểm tra file: {e}")
            time.sleep(check_interval)
    
    print(f"❌ Timeout sau {max_wait_time}s")
    return False

# ========== CÁCH SỬ DỤNG ĐƠN GIẢN ==========

def simple_file_check(file_path):
    """
    Kiểm tra file đơn giản - chỉ cần gọi 1 dòng
    """
    return wait_for_file_ready(file_path, min_size_mb=0.5, max_wait_time=30)

# ========== TEMPLATE ÁP DỤNG ==========

# Template 1: Cách cơ bản
def your_processing_function():
    input_file = "input_video.mp4"
    output_file = "output_video.mp4"
    
    # Kiểm tra file input
    if not wait_for_file_ready(input_file):
        print("❌ File input không sẵn sàng")
        return False
    
    # Thực hiện xử lý của bạn ở đây
    # your_video_processing_code()
    
    # Kiểm tra file output
    if wait_for_file_ready(output_file):
        print("✅ Xử lý thành công!")
        return True
    else:
        print("❌ File output không được tạo")
        return False
