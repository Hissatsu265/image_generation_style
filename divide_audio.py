
import os
from pydub import AudioSegment
from pydub.silence import detect_silence

def split_audio_file(input_path, output_dir="output", max_duration=14):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        audio = AudioSegment.from_file(input_path)
        max_duration_ms = max_duration * 1000

        # Tạo 0.3 giây im lặng
        silence_300ms = AudioSegment.silent(duration=300)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        current_audio = audio
        part_number = 1
        output_paths = []
        durations=[]
        while len(current_audio) > max_duration_ms:
            cut_point = find_optimal_cut_point(current_audio, max_duration_ms)
            first_segment = current_audio[:cut_point]
            remaining_audio = current_audio[cut_point:]

            # Thêm 0.3s im lặng vào cuối segment
            first_segment_with_silence = first_segment + silence_300ms

            # Xuất file với tên có 3 chữ số thập phân
            duration_sec = len(first_segment_with_silence) / 1000
            output_path = os.path.join(output_dir, f"{base_name}_part_{part_number:03d}_{duration_sec:.3f}s.mp3")
            first_segment_with_silence.export(output_path, format="mp3")
            output_paths.append(output_path)
            print(f"Saved: {output_path} (len: {duration_sec:.3f}s)")
            durations.append(duration_sec)
            current_audio = remaining_audio
            part_number += 1

        if len(current_audio) > 0:
            # Thêm 0.3s im lặng vào đoạn cuối
            final_segment = current_audio + silence_300ms
            duration_sec = len(final_segment) / 1000
            output_path = os.path.join(output_dir, f"{base_name}_part_{part_number:03d}_{duration_sec:.3f}s.mp3")
            final_segment.export(output_path, format="mp3")
            output_paths.append(output_path)
            print(f"Saved: {output_path} (len: {duration_sec:.3f}s)")
            durations.append(duration_sec)
        return output_paths,durations, True

    except Exception as e:
        print(f"Lỗi khi xử lý file: {str(e)}")
        return [], False

def find_optimal_cut_point(audio, max_duration_ms):
    start_check = 10 * 1000  
    end_check = 13 * 1000  

    end_check = min(end_check, len(audio))
    check_segment = audio[start_check:end_check]

    silence_ranges = detect_silence(check_segment, min_silence_len=100, silence_thresh=-40)

    if silence_ranges:
        first_silence = silence_ranges[0]
        silence_middle = (first_silence[0] + first_silence[1]) // 2
        optimal_cut = start_check + silence_middle
        print(f"Tìm thấy im lặng tại {optimal_cut/1000:.3f}s")
        return optimal_cut
    else:
        print(f"Không tìm thấy im lặng, cắt tại {start_check/1000:.3f}s")
        return start_check

def process_audio_file(input_path, output_dir="output"):
    return split_audio_file(input_path, output_dir, max_duration=14)