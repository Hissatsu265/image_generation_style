import os
from pydub import AudioSegment
from pydub.silence import detect_silence
import random

def split_audio_file(input_path, output_dir="output", max_duration=7):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        audio = AudioSegment.from_file(input_path)
        max_duration_ms = max_duration * 1000

        silence_300ms = AudioSegment.silent(duration=300)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        current_audio = audio
        part_number = 1
        output_paths = []
        durations = []

        while len(current_audio) > max_duration_ms :
            cut_point = find_optimal_cut_point(current_audio, max_duration_ms)
            first_segment = current_audio[:cut_point]
            remaining_audio = current_audio[cut_point:]

            # Thêm 0.3s im lặng vào cuối segment
            first_segment_with_silence = first_segment + silence_300ms

            # 👇 Chỉ lấy 2 chữ số thập phân
            duration_sec = round(len(first_segment_with_silence) / 1000, 2)
            output_path = os.path.join(output_dir, f"{base_name}_part_{part_number:03d}_{duration_sec:.2f}s.mp3")
            first_segment_with_silence.export(output_path, format="mp3")

            output_paths.append(output_path)
            durations.append(duration_sec)
            print(f"✅ Saved: {output_path} (len: {duration_sec:.2f}s)")

            current_audio = remaining_audio
            part_number += 1

        if len(current_audio) > 0:
            final_segment = current_audio + silence_300ms
            duration_sec = round(len(final_segment) / 1000, 2)
            output_path = os.path.join(output_dir, f"{base_name}_part_{part_number:03d}_{duration_sec:.2f}s.mp3")
            final_segment.export(output_path, format="mp3")

            output_paths.append(output_path)
            durations.append(duration_sec)
            print(f"✅ Saved: {output_path} (len: {duration_sec:.2f}s)")

        return output_paths, durations, True

    except Exception as e:
        print(f"❌ Lỗi khi xử lý file: {str(e)}")
        return [], [], False


def find_optimal_cut_point(audio, max_duration_ms):
    start_check = 4 * 1000  
    end_check = 12 * 1000  

    end_check = min(end_check, len(audio))
    check_segment = audio[start_check:end_check]

    silence_ranges = detect_silence(check_segment, min_silence_len=100, silence_thresh=-40)
    print("==========================", silence_ranges, "==================")

    if silence_ranges:
        chosen_silence = random.choice(silence_ranges)  # chọn ngẫu nhiên một khoảng im lặng
        silence_middle = (chosen_silence[0] + chosen_silence[1]) // 2
        optimal_cut = start_check + silence_middle
        print(f"🔇 Tìm thấy im lặng tại {optimal_cut / 1000:.2f}s (ngẫu nhiên)")
        return optimal_cut
    else:
        optimal_cut = random.randint(start_check, end_check)
        print(f"⚠️ No silence found, randomly cutting at {optimal_cut / 1000:.2f}s")
        return optimal_cut


def process_audio_file(input_path, output_dir="output"):
    return split_audio_file(input_path, output_dir, max_duration=15)

# t,s,y=process_audio_file("/workspace/multitalk_verquant/audio/german41s_add1s.wav", "output_segments")