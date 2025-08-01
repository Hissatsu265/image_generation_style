import cv2
import os

def crop_and_resize_image(img, target_width, target_height, save_path="output.jpg"):
    """
    Crop ảnh về tỉ lệ khung hình mong muốn và resize về kích thước cụ thể,
    sau đó lưu ảnh và trả về đường dẫn.

    Args:
        img: Ảnh đầu vào (numpy array)
        target_width: Chiều rộng mục tiêu
        target_height: Chiều cao mục tiêu
        save_path: Đường dẫn lưu ảnh

    Returns:
        Đường dẫn ảnh đã lưu
    """
    h, w = img.shape[:2]
    target_ratio = target_width / target_height
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Ảnh quá rộng, cần crop chiều rộng
        new_width = int(h * target_ratio)
        start_x = (w - new_width) // 2
        cropped = img[:, start_x:start_x + new_width]
    else:
        # Ảnh quá cao, cần crop chiều cao
        new_height = int(w / target_ratio)
        start_y = (h - new_height) // 2
        cropped = img[start_y:start_y + new_height, :]

    # Resize về kích thước mục tiêu
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Lưu ảnh
    cv2.imwrite(save_path, resized)

    return save_path
import cv2

img = cv2.imread("/content/coca_sp.jpg")
path = crop_and_resize_image(img, 448, 782, save_path="resized/cropped_image.jpg")
print("Ảnh đã lưu tại:", path)
