import librosa
import soundfile as sf
import numpy as np
import os
from pathlib import Path

def detect_voice_activity(audio, sr, frame_length=2048, hop_length=512, energy_threshold=0.01):
    """
    Phát hiện hoạt động giọng nói trong audio
    
    Args:
        audio: mảng audio
        sr: sample rate
        frame_length: độ dài frame
        hop_length: khoảng cách giữa các frame
        energy_threshold: ngưỡng năng lượng để phát hiện giọng nói
    
    Returns:
        mảng boolean cho biết frame nào có giọng nói
    """
    # Tính năng lượng của từng frame
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Tính spectral centroid (trung tâm phổ tần)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    
    # Tính zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Kết hợp các đặc trưng để phát hiện giọng nói
    # Giọng nói thường có năng lượng cao, spectral centroid trung bình, và ZCR không quá cao
    voice_activity = (energy > energy_threshold) & (spectral_centroid > 500) & (zcr < 0.3)
    
    return voice_activity

def find_silence_regions(voice_activity, sr, hop_length=512, min_silence_duration=0.5):
    """
    Tìm các vùng im lặng trong audio
    
    Args:
        voice_activity: mảng boolean của hoạt động giọng nói
        sr: sample rate
        hop_length: khoảng cách giữa các frame
        min_silence_duration: thời gian im lặng tối thiểu (giây)
    
    Returns:
        danh sách các vùng im lặng (start_time, end_time)
    """
    # Chuyển đổi frame index sang thời gian
    times = librosa.frames_to_time(np.arange(len(voice_activity)), sr=sr, hop_length=hop_length)
    
    # Tìm các vùng im lặng
    silence_regions = []
    in_silence = False
    silence_start = 0
    
    for i, is_voice in enumerate(voice_activity):
        if not is_voice and not in_silence:
            # Bắt đầu vùng im lặng
            in_silence = True
            silence_start = times[i]
        elif is_voice and in_silence:
            # Kết thúc vùng im lặng
            in_silence = False
            silence_end = times[i]
            if silence_end - silence_start >= min_silence_duration:
                silence_regions.append((silence_start, silence_end))
    
    # Xử lý trường hợp kết thúc trong im lặng
    if in_silence:
        silence_end = times[-1]
        if silence_end - silence_start >= min_silence_duration:
            silence_regions.append((silence_start, silence_end))
    
    return silence_regions

def split_audio_intelligent(input_file, output_dir, target_min=8, target_max=14, fade_duration=0.5, silence_insert_duration=0.4):
    """
    Chia file âm thanh thành các đoạn nhỏ một cách thông minh
    
    Args:
        input_file: đường dẫn file âm thanh đầu vào
        output_dir: thư mục lưu các file đầu ra
        target_min: thời gian tối thiểu cho mỗi đoạn (giây)
        target_max: thời gian tối đa cho mỗi đoạn (giây)
        fade_duration: thời gian fade in/out (giây)
        silence_insert_duration: thời gian im lặng chèn vào (giây)
    """
    # Tải file âm thanh
    print(f"Đang tải file: {input_file}")
    audio, sr = librosa.load(input_file, sr=None)
    total_duration = len(audio) / sr
    
    print(f"Thời gian tổng: {total_duration:.2f}s")
    print(f"Sample rate: {sr}")
    
    # Phát hiện hoạt động giọng nói
    print("Đang phát hiện hoạt động giọng nói...")
    voice_activity = detect_voice_activity(audio, sr)
    
    # Tìm các vùng im lặng
    print("Đang tìm các vùng im lặng...")
    silence_regions = find_silence_regions(voice_activity, sr)
    
    print(f"Tìm thấy {len(silence_regions)} vùng im lặng")
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Chia file thành các đoạn
    segments = []
    current_start = 0
    
    while current_start < total_duration:
        # Tìm vùng im lặng tốt nhất trong khoảng target_min đến target_max
        best_cut_point = None
        best_silence_score = 0
        
        # Tìm kiếm trong khoảng thời gian mục tiêu
        search_end = min(current_start + target_max, total_duration)
        
        for silence_start, silence_end in silence_regions:
            silence_center = (silence_start + silence_end) / 2
            
            # Chỉ xét những vùng im lặng trong khoảng tìm kiếm
            if current_start + target_min <= silence_center <= search_end:
                # Tính điểm cho vùng im lặng này
                # Ưu tiên vùng im lặng dài hơn và gần với target_max
                silence_duration = silence_end - silence_start
                distance_to_target = abs(silence_center - (current_start + target_max))
                
                # Điểm càng cao càng tốt
                score = silence_duration - distance_to_target * 0.5
                
                if score > best_silence_score:
                    best_silence_score = score
                    best_cut_point = silence_center
        
        # Quyết định điểm cắt
        if best_cut_point is not None:
            # Tìm thấy vùng im lặng phù hợp
            segments.append((current_start, best_cut_point))
            current_start = best_cut_point
        else:
            # Không tìm thấy vùng im lặng phù hợp
            # Đảm bảo đoạn có độ dài ít nhất target_min
            if current_start + target_min <= total_duration:
                cut_point = min(current_start + target_max, total_duration)
                
                # Kiểm tra nếu phần còn lại quá ngắn, gộp vào đoạn hiện tại
                remaining = total_duration - cut_point
                if remaining > 0 and remaining < target_min:
                    cut_point = total_duration
                
                print(f"Không tìm thấy vùng im lặng phù hợp từ {current_start:.2f}s đến {cut_point:.2f}s")
                print(f"Sẽ chèn {silence_insert_duration}s im lặng vào cuối đoạn")
                
                segments.append((current_start, cut_point, True))  # True đánh dấu cần chèn im lặng
                current_start = cut_point
            else:
                # Nếu phần còn lại không đủ để tạo đoạn mới, gộp vào đoạn trước
                if segments:
                    last_segment = segments[-1]
                    if len(last_segment) == 3:  # Có flag chèn im lặng
                        segments[-1] = (last_segment[0], total_duration, last_segment[2])
                    else:
                        segments[-1] = (last_segment[0], total_duration)
                else:
                    segments.append((current_start, total_duration))
                break
    
    # Xử lý đoạn cuối cùng
    if current_start < total_duration:
        remaining_duration = total_duration - current_start
        if remaining_duration >= target_min:
            segments.append((current_start, total_duration))
        else:
            # Gộp với đoạn trước đó nếu đoạn cuối quá ngắn
            if segments:
                last_segment = segments[-1]
                if len(last_segment) == 3:  # Có flag chèn im lặng
                    segments[-1] = (last_segment[0], total_duration, last_segment[2])
                else:
                    segments[-1] = (last_segment[0], total_duration)
            else:
                # Nếu không có đoạn trước và đoạn cuối quá ngắn, vẫn tạo đoạn này
                segments.append((current_start, total_duration))
    
    # Kiểm tra và điều chỉnh các đoạn để đảm bảo thời gian phù hợp
    final_segments = []
    for segment_info in segments:
        start_time = segment_info[0]
        end_time = segment_info[1]
        need_silence = len(segment_info) > 2 and segment_info[2]
        
        duration = end_time - start_time
        
        # Nếu đoạn quá dài (>14s), chia nhỏ
        if duration > target_max:
            temp_start = start_time
            while temp_start < end_time:
                temp_end = min(temp_start + target_max, end_time)
                remaining = end_time - temp_end
                
                # Nếu phần còn lại quá ngắn, gộp vào đoạn hiện tại
                if remaining > 0 and remaining < target_min:
                    temp_end = end_time
                
                if temp_end == end_time and need_silence:
                    final_segments.append((temp_start, temp_end, True))
                else:
                    final_segments.append((temp_start, temp_end))
                
                temp_start = temp_end
        else:
            final_segments.append(segment_info)
    
    segments = final_segments
    
    # Xuất các đoạn âm thanh
    print(f"Đang xuất {len(segments)} đoạn âm thanh...")
    
    fade_samples = int(fade_duration * sr)
    silence_samples = int(silence_insert_duration * sr)
    
    for i, segment_info in enumerate(segments):
        start_time = segment_info[0]
        end_time = segment_info[1]
        need_silence = len(segment_info) > 2 and segment_info[2]
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Trích xuất đoạn âm thanh
        segment_audio = audio[start_sample:end_sample]
        
        # Thêm 0.4s im lặng vào đầu từ đoạn thứ 2 trở đi
        if i > 0:  # Từ segment thứ 2 (index 1) trở đi
            silence_prefix = np.zeros(silence_samples)
            segment_audio = np.concatenate([silence_prefix, segment_audio])
        
        # Chèn im lặng vào cuối nếu cần (cho các đoạn không có vùng im lặng tự nhiên)
        if need_silence:
            silence_suffix = np.zeros(silence_samples)
            segment_audio = np.concatenate([segment_audio, silence_suffix])
        
        # Thêm fade in/out
        if len(segment_audio) > 2 * fade_samples:
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            segment_audio[:fade_samples] *= fade_in
            
            # Fade out
            fade_out = np.linspace(1, 0, fade_samples)
            segment_audio[-fade_samples:] *= fade_out
        
        # Lưu file
        output_file = os.path.join(output_dir, f"segment_{i+1:03d}.wav")
        sf.write(output_file, segment_audio, sr)
        
        duration = len(segment_audio) / sr
        silence_notes = []
        if i > 0:
            silence_notes.append(f"0.4s im lặng đầu")
        if need_silence:
            silence_notes.append(f"0.4s im lặng cuối")
        
        silence_note = f" ({', '.join(silence_notes)})" if silence_notes else ""
        print(f"Đã tạo: {output_file} (thời gian: {duration:.2f}s{silence_note})")
    
    print(f"\nHoàn thành! Đã tạo {len(segments)} đoạn âm thanh trong thư mục: {output_dir}")
    print(f"Lưu ý: Từ đoạn thứ 2 trở đi đã được thêm 0.4s im lặng vào đầu")

def main():
    # Cấu hình
    input_file = "/workspace/multitalk_verquant/audio/folie_2_alterative.wav"  # Thay đổi đường dẫn file đầu vào
    output_dir = "output_segments"  # Thư mục lưu các đoạn
    
    # Kiểm tra file đầu vào
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file: {input_file}")
        print("Vui lòng thay đổi đường dẫn file trong biến 'input_file'")
        return
    
    try:
        split_audio_intelligent(
            input_file=input_file,
            output_dir=output_dir,
            target_min=8,  # Thời gian tối thiểu (giây)
            target_max=14,  # Thời gian tối đa (giây)
            fade_duration=0.5,  # Thời gian fade (giây)
            silence_insert_duration=0.4  # Thời gian im lặng chèn vào (giây)
        )
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()