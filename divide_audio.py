import os
from pydub import AudioSegment
from pydub.silence import detect_silence
import math

def split_audio_file(input_path, output_dir="output", max_duration=14):
    try:
        # Tạo thư mục output nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load file audio
        audio = AudioSegment.from_file(input_path)
        
        # Chuyển đổi thời gian sang milliseconds
        max_duration_ms = max_duration * 1000
        
        # Tạo âm thanh im lặng 0.2 giây
        silence_200ms = AudioSegment.silent(duration=200)
        
        # Lấy tên file gốc (không có extension)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Danh sách chứa các đoạn audio
        segments = []
        current_audio = audio
        part_number = 1
        output_paths = []

        while len(current_audio) > max_duration_ms:
            # Tìm điểm cắt tối ưu trong khoảng 7-11 giây
            cut_point = find_optimal_cut_point(current_audio, max_duration_ms)
            
            # Cắt đoạn đầu
            first_segment = current_audio[:cut_point]
            
            # Cắt đoạn sau
            remaining_audio = current_audio[cut_point:]
            
            # Thêm im lặng 0.2s vào cuối đoạn đầu
            # first_segment_with_silence = first_segment + silence_200ms
            first_segment_with_silence = first_segment
            
            # Thêm im lặng 0.2s vào đầu đoạn sau
            # remaining_audio = silence_200ms + remaining_audio
            remaining_audio = remaining_audio
            # Lưu đoạn đầu
            output_path = os.path.join(output_dir, f"{base_name}_part_{part_number:03d}.mp3")
            first_segment_with_silence.export(output_path, format="mp3")
            output_paths.append(output_path)
            print(f"Saved: {output_path} (len: {len(first_segment_with_silence)/1000:.2f}s)")
            
            current_audio = remaining_audio
            part_number += 1
        if len(current_audio) > 0:
            output_path = os.path.join(output_dir, f"{base_name}_part_{part_number:03d}.mp3")
            current_audio.export(output_path, format="mp3")
            output_paths.append(output_path)
            print(f"Saved: {output_path} (len: {len(current_audio)/1000:.2f}s)")

        # print(f"Hoàn thành! Đã tạo {part_number} file từ {input_path}")
        return output_paths, True
        
    except Exception as e:
        print(f"Lỗi khi xử lý file: {str(e)}")
        return [], False

def find_optimal_cut_point(audio, max_duration_ms):
    start_check = 7 * 1000  # 7 giây
    end_check = 11 * 1000   # 11 giây
    
    end_check = min(end_check, len(audio))
    
    check_segment = audio[start_check:end_check]
   
    silence_ranges = detect_silence(check_segment, 
                                   min_silence_len=100, 
                                   silence_thresh=-40)
    
    if silence_ranges:
        first_silence = silence_ranges[0]
        silence_middle = (first_silence[0] + first_silence[1]) // 2
        optimal_cut = start_check + silence_middle
        
        print(f"Tìm thấy im lặng tại {optimal_cut/1000:.2f}s")
        return optimal_cut
    else:
        print(f"Không tìm thấy im lặng, cắt tại {start_check/1000:.2f}s")
        return start_check

def process_audio_file(input_path, output_dir="output"):
    return split_audio_file(input_path, output_dir, max_duration=14)


