from PIL import Image, ImageOps
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

def pad_to_ratio(image, target_ratio):
    width, height = image.size
    current_ratio = width / height

    if abs(current_ratio - target_ratio) < 1e-3:
        return image  # gần đúng rồi, khỏi pad

    if current_ratio > target_ratio:
        # ảnh rộng hơn → pad trên và dưới
        target_height = int(width / target_ratio)
        pad_total = target_height - height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return ImageOps.expand(image, border=(0, pad_top, 0, pad_bottom), fill="black")
    else:
        # ảnh hẹp hơn → pad trái và phải
        target_width = int(height * target_ratio)
        pad_total = target_width - width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return ImageOps.expand(image, border=(pad_left, 0, pad_right, 0), fill="black")

def crop_and_pad_bboxes(image_path, bboxes, output_dir="output_crops"):
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
        except ValueError as e:
            print(f"BBox {i+1} bị lỗi: {e}")
            continue

        cropped = image.crop((left, top, right, bottom))
        padded = pad_to_ratio(cropped, original_ratio)
        out_path = os.path.join(output_dir, f"crop_{i+1}.png")
        padded.save(out_path)
        result_paths.append(out_path)

    return result_paths



# if __name__ == "__main__":
#     image_path = "/content/1586449950729.png"
#     bboxes = [
#         (100, 100, 600, 500), 
#         (100, 550, 600, 900)
#     ]
#     swapped_bboxes = [(ymin, xmin, ymax, xmax) for xmin, ymin, xmax, ymax in bboxes]
#     results = crop_and_pad_image(image_path, swapped_bboxes)
#     print("Saved:", results)
