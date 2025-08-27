import os
import time
import librosa
import soundfile as sf
from mutagen import File
import wave
import subprocess

def wait_for_audio_ready(file_path, min_size_mb=0.01, max_wait_time=60, check_interval=1, min_duration=0.1):
    """
    Kiểm tra file audio đã sẵn sàng để sử dụng
    
    Args:
        file_path: đường dẫn file audio cần kiểm tra
        min_size_mb: kích thước tối thiểu của file (MB)
        max_wait_time: thời gian chờ tối đa (giây)
        check_interval: khoảng thời gian giữa các lần kiểm tra (giây)
        min_duration: thời lượng audio tối thiểu (giây)
    
    Returns:
        bool: True nếu file sẵn sàng, False nếu timeout
    """
    print(f"🎵 Đang kiểm tra file audio: {file_path}")
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
                    print("📊 File ổn định, tiến hành kiểm tra tính toàn vẹn audio...")
                    
                    # Kiểm tra file audio có thể đọc được không
                    if _validate_audio_file(file_path, min_duration):
                        return True
                    
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

def _validate_audio_file(file_path, min_duration=0.1):
    """
    Kiểm tra tính hợp lệ của file audio bằng nhiều phương pháp
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Phương pháp 1: Sử dụng librosa (tốt nhất cho hầu hết format)
    try:
        duration = librosa.get_duration(path=file_path)
        if duration >= min_duration:
            print(f"✅ File hợp lệ (librosa) - Thời lượng: {duration:.2f}s")
            return True
        else:
            print(f"⚠️ File quá ngắn: {duration:.2f}s < {min_duration}s")
    except Exception as e:
        print(f"⚠️ Lỗi librosa: {e}")
    
    # Phương pháp 2: Sử dụng soundfile
    try:
        with sf.SoundFile(file_path) as f:
            frames = len(f)
            samplerate = f.samplerate
            duration = frames / samplerate
            if duration >= min_duration:
                print(f"✅ File hợp lệ (soundfile) - Thời lượng: {duration:.2f}s, SR: {samplerate}Hz")
                return True
            else:
                print(f"⚠️ File quá ngắn: {duration:.2f}s < {min_duration}s")
    except Exception as e:
        print(f"⚠️ Lỗi soundfile: {e}")
    
    # Phương pháp 3: Sử dụng mutagen (tốt cho metadata)
    try:
        audio_file = File(file_path)
        if audio_file is not None and hasattr(audio_file, 'info'):
            duration = audio_file.info.length
            if duration >= min_duration:
                print(f"✅ File hợp lệ (mutagen) - Thời lượng: {duration:.2f}s")
                return True
            else:
                print(f"⚠️ File quá ngắn: {duration:.2f}s < {min_duration}s")
    except Exception as e:
        print(f"⚠️ Lỗi mutagen: {e}")
    
    # Phương pháp 4: Sử dụng wave (chỉ cho file WAV)
    if file_ext == '.wav':
        try:
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                if duration >= min_duration:
                    print(f"✅ File WAV hợp lệ - Thời lượng: {duration:.2f}s, SR: {sample_rate}Hz")
                    return True
                else:
                    print(f"⚠️ File quá ngắn: {duration:.2f}s < {min_duration}s")
        except Exception as e:
            print(f"⚠️ Lỗi wave: {e}")
    
    # Phương pháp 5: Sử dụng ffprobe (yêu cầu ffmpeg)
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', 
            file_path
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            if duration >= min_duration:
                print(f"✅ File hợp lệ (ffprobe) - Thời lượng: {duration:.2f}s")
                return True
            else:
                print(f"⚠️ File quá ngắn: {duration:.2f}s < {min_duration}s")
    except Exception as e:
        print(f"⚠️ Lỗi ffprobe: {e}")
    
    print("❌ Không thể xác thực file audio bằng bất kỳ phương pháp nào")
    return False

# ========== CÁCH SỬ DỤNG ĐƠN GIẢN ==========

def simple_audio_check(file_path):
    """
    Kiểm tra file audio đơn giản - chỉ cần gọi 1 dòng
    """
    return wait_for_audio_ready(file_path, min_size_mb=0.05, max_wait_time=30, min_duration=0.5)

def quick_audio_check(file_path, timeout=15):
    """
    Kiểm tra nhanh file audio với timeout ngắn
    """
    return wait_for_audio_ready(file_path, min_size_mb=0.01, max_wait_time=timeout, min_duration=0.1)

# ========== TEMPLATE ÁP DỤNG ==========

# Template 1: Xử lý audio cơ bản
def process_audio_with_check():
    input_file = "input_audio.wav"
    output_file = "output_audio.wav"
    
    # Kiểm tra file input
    if not wait_for_audio_ready(input_file):
        print("❌ File input không sẵn sàng")
        return False
    
    # Thực hiện xử lý audio của bạn ở đây
    print("🎵 Đang xử lý audio...")
    # your_audio_processing_code()
    
    # Kiểm tra file output
    if wait_for_audio_ready(output_file, min_duration=1.0):
        print("✅ Xử lý audio thành công!")
        return True
    else:
        print("❌ File output không được tạo hoặc không hợp lệ")
        return False

# Template 2: Batch processing với kiểm tra
def batch_audio_processing(input_files, output_dir):
    """
    Xử lý nhiều file audio với kiểm tra
    """
    results = []
    
    for input_file in input_files:
        print(f"\n🎵 Xử lý: {input_file}")
        
        # Kiểm tra file input
        if not wait_for_audio_ready(input_file):
            print(f"❌ Bỏ qua file: {input_file}")
            results.append((input_file, False, "Input không sẵn sàng"))
            continue
        
        # Tạo tên file output
        filename = os.path.basename(input_file)
        name, ext = os.path.splitext(filename)
        output_file = os.path.join(output_dir, f"{name}_processed{ext}")
        
        # Thực hiện xử lý
        try:
            # your_audio_processing_function(input_file, output_file)
            
            # Kiểm tra output
            if wait_for_audio_ready(output_file, min_duration=0.5):
                print(f"✅ Thành công: {output_file}")
                results.append((input_file, True, output_file))
            else:
                print(f"❌ Lỗi output: {output_file}")
                results.append((input_file, False, "Output không hợp lệ"))
                
        except Exception as e:
            print(f"❌ Lỗi xử lý: {e}")
            results.append((input_file, False, str(e)))
    
    return results

# Template 3: Theo dõi quá trình render/export audio
def monitor_audio_export(output_path, expected_duration=None, timeout=120):
    """
    Theo dõi quá trình export/render audio
    """
    print(f"🎵 Theo dõi export audio: {output_path}")
    
    # Tăng thời gian chờ cho file lớn
    if expected_duration and expected_duration > 60:
        timeout = max(timeout, expected_duration * 2)
    
    min_duration = expected_duration * 0.8 if expected_duration else 0.5
    
    success = wait_for_audio_ready(
        output_path, 
        min_size_mb=0.1,
        max_wait_time=timeout,
        min_duration=min_duration
    )
    
    if success:
        print("🎉 Export audio hoàn tất!")
        return True
    else:
        print("💥 Export audio thất bại hoặc timeout!")
        return False

# # ========== EXAMPLE USAGE ==========
# if __name__ == "__main__":
#     # Ví dụ sử dụng
#     audio_file = "test_audio.wav"
    
#     print("=== Test 1: Kiểm tra đơn giản ===")
#     if simple_audio_check(audio_file):
#         print("Ready to use!")
    
#     print("\n=== Test 2: Kiểm tra nhanh ===")
#     if quick_audio_check(audio_file):
#         print("Quick check passed!")
    
#     print("\n=== Test 3: Kiểm tra chi tiết ===")
#     if wait_for_audio_ready(audio_file, min_size_mb=0.1, max_wait_time=60, min_duration=2.0):
#         print("Detailed check passed!")