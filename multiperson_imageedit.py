from PIL import Image
import os

def convert_bbox_pil_coordinates(xmin, ymin, xmax, ymax, img_width, img_height):
    xmin = max(0, min(xmin, img_height - 1))
    xmax = max(0, min(xmax, img_height - 1))
    ymin = max(0, min(ymin, img_width - 1))
    ymax = max(0, min(ymax, img_width - 1))

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("❌ Tọa độ không hợp lệ: xmax > xmin và ymax > ymin")

    top = img_height - xmax
    bottom = img_height - xmin
    left = ymin
    right = ymax

    return (left, top, right, bottom)

def expand_crop_to_ratio(left, top, right, bottom, target_ratio, img_width, img_height):
    """
    Mở rộng vùng crop để đạt được tỉ lệ mong muốn mà không cần padding
    """
    current_width = right - left
    current_height = bottom - top
    current_ratio = current_width / current_height
    
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    
    if current_ratio < target_ratio:
        # Cần mở rộng chiều rộng
        new_width = current_height * target_ratio
        new_left = center_x - new_width / 2
        new_right = center_x + new_width / 2
        new_top = top
        new_bottom = bottom
    else:
        # Cần mở rộng chiều cao
        new_height = current_width / target_ratio
        new_top = center_y - new_height / 2
        new_bottom = center_y + new_height / 2
        new_left = left
        new_right = right
    
    # Đảm bảo không vượt quá biên ảnh
    new_left = max(0, new_left)
    new_top = max(0, new_top)
    new_right = min(img_width, new_right)
    new_bottom = min(img_height, new_bottom)
    
    # Nếu sau khi giới hạn biên, tỉ lệ bị thay đổi, điều chỉnh lại
    actual_width = new_right - new_left
    actual_height = new_bottom - new_top
    actual_ratio = actual_width / actual_height
    
    # Nếu tỉ lệ vẫn chưa đúng sau khi giới hạn biên, điều chỉnh theo chiều có thể điều chỉnh được
    if abs(actual_ratio - target_ratio) > 0.01:  # tolerance
        if new_left == 0 or new_right == img_width:
            # Không thể mở rộng thêm chiều rộng, điều chỉnh chiều cao
            target_height = actual_width / target_ratio
            center_y = (new_top + new_bottom) / 2
            new_top = max(0, center_y - target_height / 2)
            new_bottom = min(img_height, center_y + target_height / 2)
        
        if new_top == 0 or new_bottom == img_height:
            # Không thể mở rộng thêm chiều cao, điều chỉnh chiều rộng
            target_width = actual_height * target_ratio
            center_x = (new_left + new_right) / 2
            new_left = max(0, center_x - target_width / 2)
            new_right = min(img_width, center_x + target_width / 2)
    
    return (int(new_left), int(new_top), int(new_right), int(new_bottom))

def crop_with_ratio_expansion(image_path, bboxes, output_dir="output_crops"):
    """
    Crop ảnh với việc mở rộng vùng crop để giữ nguyên tỉ lệ ảnh gốc
    """
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path)
    img_width, img_height = image.size
    original_ratio = img_width / img_height

    result_paths = []

    for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
        try:
            left, top, right, bottom = convert_bbox_pil_coordinates(
                xmin, ymin, xmax, ymax, img_width, img_height
            )
            
            # Mở rộng vùng crop để đạt tỉ lệ mong muốn
            expanded_coords = expand_crop_to_ratio(
                left, top, right, bottom, original_ratio, img_width, img_height
            )
            
            print(f"BBox {i+1}:")
            print(f"  Original crop: ({left}, {top}, {right}, {bottom})")
            print(f"  Expanded crop: {expanded_coords}")
            print(f"  Original size: {right-left}x{bottom-top}")
            print(f"  Expanded size: {expanded_coords[2]-expanded_coords[0]}x{expanded_coords[3]-expanded_coords[1]}")
            
        except ValueError as e:
            print(f"BBox {i+1} bị lỗi: {e}")
            continue

        # Crop với tọa độ đã mở rộng
        cropped = image.crop(expanded_coords)
        out_path = os.path.join(output_dir, f"crop_{i+1}.png")
        cropped.save(out_path)
        result_paths.append(out_path)
        
        print(f"  Saved: {out_path}")
        print(f"  Final ratio: {cropped.size[0]/cropped.size[1]:.3f} (target: {original_ratio:.3f})")
        print()

    return result_paths


# if __name__ == "__main__":
#     image_path = "/content/Screenshot 2025-07-23 114649.png"
#     bboxes = [
#         (391, 211, 654, 534), 
#         (291, 611, 554, 834)
#     ]
#     results = crop_with_ratio_expansion(image_path, bboxes)
#     print("Kết quả đã lưu:", results)