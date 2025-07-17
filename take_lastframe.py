import cv2
import os

def save_last_frame(video_path: str, output_path: str = "last_frame.jpg") -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("Video has no frames")

    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()

    if not ret:
        raise IOError("Failed to read the last frame.")

    cv2.imwrite(output_path, frame)
    cap.release()
    return os.path.abspath(output_path)
path = save_last_frame("/workspace/multitalk_verquant/d1.mp4")
print("Saved frame path:", path)