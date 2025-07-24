import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

class ImagePadder:
    def __init__(self):
        self.VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
        
        self.ASPECT_RATIO_627 = {
            '0.26': ([320, 1216], 1),
            '0.38': ([384, 1024], 1),
            '0.50': ([448, 896], 1),
            '0.67': ([512, 768], 1),
            '0.82': ([576, 704], 1),
            '1.00': ([640, 640], 1),
            '1.22': ([704, 576], 1),
            '1.50': ([768, 512], 1),
            '1.86': ([832, 448], 1),
            '2.00': ([896, 448], 1),
            '2.50': ([960, 384], 1),
            '2.83': ([1088, 384], 1),
            '3.60': ([1152, 320], 1),
            '3.80': ([1216, 320], 1),
            '4.00': ([1280, 320], 1)
        }
        
        self.ASPECT_RATIO_960 = {
            '0.22': ([448, 2048], 1),
            '0.29': ([512, 1792], 1),
            '0.36': ([576, 1600], 1),
            '0.45': ([640, 1408], 1),
            '0.55': ([704, 1280], 1),
            '0.63': ([768, 1216], 1),
            '0.76': ([832, 1088], 1),
            '0.88': ([896, 1024], 1),
            '1.00': ([960, 960], 1),
            '1.14': ([1024, 896], 1),
            '1.31': ([1088, 832], 1),
            '1.50': ([1152, 768], 1),
            '1.58': ([1216, 768], 1),
            '1.82': ([1280, 704], 1),
            '1.91': ([1344, 704], 1),
            '2.20': ([1408, 640], 1),
            '2.30': ([1472, 640], 1),
            '2.67': ([1536, 576], 1),
            '2.89': ([1664, 576], 1),
            '3.62': ([1856, 512], 1),
            '3.75': ([1920, 512], 1)
        }
    
    def find_closest_ratio(self, image_ratio: float, ratio_dict: Dict) -> Tuple[str, Tuple[int, int], float]:
        """Tìm tỉ lệ gần nhất trong dictionary"""
        closest_ratio_str = min(ratio_dict.keys(), 
                               key=lambda x: abs(float(x) - image_ratio))
        target_size = ratio_dict[closest_ratio_str][0]
        target_ratio = float(closest_ratio_str)
        return closest_ratio_str, target_size, target_ratio
    
    def pad_image(self, image_path: str, ratio_type: str = "627", 
                  output_path: Optional[str] = None, 
                  info_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Pad ảnh để đạt tỉ lệ mong muốn và lưu thông tin khôi phục
        
        Args:
            image_path: Đường dẫn ảnh đầu vào
            ratio_type: "627" hoặc "960" để chọn bộ tỉ lệ
            output_path: Đường dẫn ảnh đầu ra (nếu None sẽ tự tạo)
            info_path: Đường dẫn file json lưu thông tin (nếu None sẽ tự tạo)
        
        Returns:
            Dictionary chứa thông tin padding
        """
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh từ {image_path}")
        
        original_height, original_width = image.shape[:2]
        original_ratio = original_width / original_height
        
        # Chọn bộ tỉ lệ
        ratio_dict = self.ASPECT_RATIO_627 if ratio_type == "627" else self.ASPECT_RATIO_960
        
        # Tìm tỉ lệ gần nhất
        closest_ratio_str, target_size, target_ratio = self.find_closest_ratio(original_ratio, ratio_dict)
        target_width, target_height = target_size
        
        print(f"Tỉ lệ gốc: {original_ratio:.3f} ({original_width}x{original_height})")
        print(f"Tỉ lệ target: {target_ratio:.3f} ({target_width}x{target_height})")
        
        # Tính toán kích thước sau khi pad để đạt target ratio
        if original_ratio > target_ratio:
            # Ảnh quá rộng so với target -> cần pad trên/dưới
            new_width = original_width
            new_height = int(original_width / target_ratio)
            pad_left = pad_right = 0
            pad_total_height = new_height - original_height
            pad_top = pad_total_height // 2
            pad_bottom = pad_total_height - pad_top
        else:
            # Ảnh quá cao so với target -> cần pad trái/phải
            new_height = original_height
            new_width = int(original_height * target_ratio)
            pad_top = pad_bottom = 0
            pad_total_width = new_width - original_width
            pad_left = pad_total_width // 2
            pad_right = pad_total_width - pad_left
        
        # Tạo ảnh mới với padding đen
        padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        
        # Đặt ảnh gốc vào giữa ảnh padded
        start_y = pad_top
        end_y = start_y + original_height
        start_x = pad_left
        end_x = start_x + original_width
        
        padded_image[start_y:end_y, start_x:end_x] = image
        
        # Kiểm tra tỉ lệ sau khi pad
        final_ratio = new_width / new_height
        print(f"Tỉ lệ sau pad: {final_ratio:.3f} ({new_width}x{new_height})")
        print(f"Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
        
        # Thông tin để khôi phục
        padding_info = {
            "original_size": [original_width, original_height],
            "padded_size": [new_width, new_height],
            "target_ratio_str": closest_ratio_str,
            "target_size": target_size,
            "ratio_type": ratio_type,
            "padding": {
                "top": pad_top,
                "bottom": pad_bottom,
                "left": pad_left,
                "right": pad_right
            },
            "original_ratio": original_ratio,
            "target_ratio": target_ratio,
            "final_ratio": final_ratio
        }
        
        # Tạo tên file output nếu không được cung cấp
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_padded{input_path.suffix}"
        
        if info_path is None:
            input_path = Path(image_path)
            info_path = input_path.parent / f"{input_path.stem}_padding_info.json"
        
        # Lưu ảnh và thông tin
        cv2.imwrite(str(output_path), padded_image)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(padding_info, f, indent=2, ensure_ascii=False)
        
        print(f"Đã lưu ảnh padded: {output_path}")
        print(f"Đã lưu thông tin padding: {info_path}")
        
        return padding_info, output_path
    
    def restore_video_ratio(self, video_path: str, padding_info_path: str, 
                           output_path: Optional[str] = None) -> str:
        """
        Khôi phục tỉ lệ video về tỉ lệ gốc bằng cách crop và resize
        
        Args:
            video_path: Đường dẫn video cần khôi phục (đã được mô hình resize về target_size)
            padding_info_path: Đường dẫn file json chứa thông tin padding
            output_path: Đường dẫn video đầu ra
        
        Returns:
            Đường dẫn video đã khôi phục
        """
        # Đọc thông tin padding
        with open(padding_info_path, 'r', encoding='utf-8') as f:
            padding_info = json.load(f)
        
        # Mở video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video {video_path}")
        
        # Lấy thông tin video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video từ mô hình: {video_width}x{video_height}, FPS: {fps}")
        print(f"Target size từ padding info: {padding_info['target_size']}")
        
        # Video từ mô hình có kích thước = target_size trong padding_info
        target_width, target_height = padding_info["target_size"]
        
        # Kiểm tra xem video có đúng kích thước target không
        if video_width != target_width or video_height != target_height:
            print(f"Cảnh báo: Video size ({video_width}x{video_height}) không khớp với target size ({target_width}x{target_height})")
            print("Sẽ sử dụng kích thước video thực tế để tính toán")
            target_width, target_height = video_width, video_height
        
        # Tính toán vùng crop dựa trên tỉ lệ padding trong ảnh padded gốc
        padded_width, padded_height = padding_info["padded_size"]
        original_width, original_height = padding_info["original_size"]
        
        # Tính tỉ lệ của vùng ảnh gốc trong ảnh padded
        original_region_ratio_x = original_width / padded_width
        original_region_ratio_y = original_height / padded_height
        
        # Tính vị trí của vùng ảnh gốc trong video (đã được resize về target_size)
        crop_width = int(target_width * original_region_ratio_x)
        crop_height = int(target_height * original_region_ratio_y)
        
        # Tính offset để crop từ giữa (vì padding được đặt đều 2 bên)
        crop_left = (target_width - crop_width) // 2
        crop_top = (target_height - crop_height) // 2
        crop_right = crop_left + crop_width
        crop_bottom = crop_top + crop_height
        
        print(f"Padding info:")
        print(f"  - Ảnh gốc: {original_width}x{original_height}")
        print(f"  - Ảnh padded: {padded_width}x{padded_height}")
        print(f"  - Padding: top={padding_info['padding']['top']}, bottom={padding_info['padding']['bottom']}, left={padding_info['padding']['left']}, right={padding_info['padding']['right']}")
        print(f"Crop region: left={crop_left}, top={crop_top}, right={crop_right}, bottom={crop_bottom}")
        print(f"Crop size: {crop_width}x{crop_height}")
        print(f"Tỉ lệ sau crop: {crop_width/crop_height:.3f} (gốc: {padding_info['original_ratio']:.3f})")
        
        # Tạo tên file output
        if output_path is None:
            input_path = Path(video_path)
            output_path = input_path.parent / f"{input_path.stem}_restored{input_path.suffix}"
        
        # Tạo video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_width, crop_height))
        
        # Xử lý từng frame
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop frame để loại bỏ padding
            cropped_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
            out.write(cropped_frame)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Đã xử lý {frame_idx}/{frame_count} frames")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Đã lưu video restored: {output_path}")
        print(f"Video restored có tỉ lệ: {crop_width/crop_height:.3f}")
        return str(output_path)

# # Ví dụ sử dụng
# if __name__ == "__main__":
#     padder = ImagePadder()
    
#     # Pad ảnh
#     try:
#         info,output_path = padder.pad_image(
#             image_path="/content/1.png",
#             ratio_type="627", 
#             output_path="padded_image.jpg",
#             info_path="padding_info.json"
#         )
        
#         # Sau khi tạo video từ ảnh padded, khôi phục tỉ lệ gốc
#         restored_video = padder.restore_video_ratio(
#             video_path="/content/247singleperson.mp4",
#             padding_info_path="padding_info.json",
#             output_path="restored_video.mp4"
#         )
        
#     except Exception as e:
#         print(f"Lỗi: {e}")