from pydub import AudioSegment
import numpy as np

def detect_real_peak_voice(audio_path, frame_duration_ms=200, z_threshold=1.2):
    audio = AudioSegment.from_file(audio_path)
    duration_sec = len(audio) / 1000.0
    rms_list, timestamps = [], []

    for i in range(0, len(audio), frame_duration_ms):
        segment = audio[i:i+frame_duration_ms]
        rms = segment.rms
        rms_list.append(rms)
        timestamps.append(i / 1000.0)

    rms_array = np.array(rms_list)
    mean_rms = np.mean(rms_array)
    std_rms = np.std(rms_array)

    peak_times = [timestamps[i] for i, rms in enumerate(rms_array)
                  if rms > mean_rms + z_threshold * std_rms]

    return peak_times, duration_sec

def group_peaks(peaks, max_audio_time, min_gap=0.5, max_gap=3.0, min_start_time=3.5):
    print(peaks)
    groups = []
    n = len(peaks)
    for i in range(n):
        for j in range(i+1, min(i+4, n)):  # tổ hợp 2-3 điểm
            group = peaks[i:j+1]
            if len(group) >= 2 and group[0] >= min_start_time and group[-1] < max_audio_time - 4:
                gaps = [group[k+1] - group[k] for k in range(len(group)-1)]
                if all(min_gap <= g <= max_gap for g in gaps):
                    groups.append(group)
    return groups

def remove_elements_between(arr, a, b):
    return [x for x in arr if (a < x < b)]

def select_peak_segment(audio_path):
    peak_times, audio_duration = detect_real_peak_voice(audio_path)
    print(f"⏱️ Audio duration: {audio_duration:.2f}s")
    
    # Lọc bỏ các peak nằm ngoài khoảng cho phép
    peak_times = remove_elements_between(peak_times, 2, audio_duration - 4)
    peak_times = sorted(peak_times)

    valid_groups = group_peaks(peak_times, max_audio_time=audio_duration, min_start_time=3.5)

    if valid_groups:
        print("✅ Chọn tổ hợp hợp lệ đầu tiên:")
        selected = valid_groups[0]
    else:
        fallback = [t for t in peak_times if 3.5 <= t < audio_duration - 4]
        if fallback:
            print("⚠️ Không có tổ hợp hợp lệ, chọn mốc đầu tiên khả dụng:")
            selected = [fallback[0]]
        else:
            print("❌ Không tìm thấy mốc phù hợp.")
            selected = []

    print("👉 Thời điểm áp dụng zoom:", selected)
    return selected

# video_path = "/workspace/multitalk_verquant/audio/Neuberger_2.wav"
# selected_peaks = select_peak_segment(video_path)
# print(selected_peaks)