import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import math

# ========================
# Các hiệu ứng video frame
# ========================

def effect_slide(frame, progress, direction="horizontal"):
    h, w = frame.shape[:2]
    offset = int(progress * (w if direction == "horizontal" else h))
    if direction == "horizontal":
        left = frame[:, :w - offset]
        right = frame[:, w - offset:]
        return np.hstack((right, left))
    else:
        top = frame[:h - offset, :]
        bottom = frame[h - offset:, :]
        return np.vstack((bottom, top))

def effect_rotate(frame, progress, max_angle=180):
    h, w = frame.shape[:2]
    angle = progress * max_angle
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(frame, M, (w, h))

def effect_circle_mask(frame, progress):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = int(progress * np.sqrt(h**2 + w**2) / 2)
    cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return masked

# ========================
# Hiệu ứng mới
# ========================

def effect_fade_in(frame, progress):
    """Fade in từ đen"""
    alpha = progress
    return (frame * alpha).astype(np.uint8)

def effect_fade_out(frame, progress):
    """Fade out về đen"""
    alpha = 1.0 - progress
    return (frame * alpha).astype(np.uint8)

def effect_fadeout_fadein(frame, progress):
    """Nửa đầu fade out, nửa sau fade in"""
    if progress <= 0.5:
        # Nửa đầu: fade out
        alpha = 1.0 - (progress * 2)
    else:
        # Nửa sau: fade in
        alpha = (progress - 0.5) * 2
    return (frame * alpha).astype(np.uint8)

def effect_crossfade(frame1, frame2, progress):
    """Crossfade giữa 2 frame (cần 2 frame input)"""
    alpha = progress
    result = (frame1 * (1 - alpha) + frame2 * alpha).astype(np.uint8)
    return result

def effect_rgb_split(frame, progress, max_offset=20):
    """RGB Split - Glitch effect tách kênh màu"""
    h, w = frame.shape[:2]
    offset = int(progress * max_offset)

    # Tạo frame kết quả
    result = np.zeros_like(frame)

    # Kênh đỏ dịch sang trái
    if offset > 0:
        result[:, :-offset, 2] = frame[:, offset:, 2]  # Red channel
    else:
        result[:, :, 2] = frame[:, :, 2]

    # Kênh xanh lá giữ nguyên
    result[:, :, 1] = frame[:, :, 1]  # Green channel

    # Kênh xanh dương dịch sang phải
    if offset > 0:
        result[:, offset:, 0] = frame[:, :-offset, 0]  # Blue channel
    else:
        result[:, :, 0] = frame[:, :, 0]

    return result

def effect_flip_horizontal(frame, progress):
    """Lật ngang dần dần"""
    h, w = frame.shape[:2]
    # Tạo hiệu ứng lật bằng cách thay đổi tỷ lệ ngang
    scale_x = 1 - 2 * progress if progress <= 0.5 else 2 * progress - 1
    if scale_x < 0:
        scale_x = abs(scale_x)
        frame = cv2.flip(frame, 1)  # Flip horizontal

    M = np.float32([[scale_x, 0, w * (1 - scale_x) / 2],
                    [0, 1, 0]])
    return cv2.warpAffine(frame, M, (w, h))

def effect_flip_vertical(frame, progress):
    """Lật dọc dần dần"""
    h, w = frame.shape[:2]
    scale_y = 1 - 2 * progress if progress <= 0.5 else 2 * progress - 1
    if scale_y < 0:
        scale_y = abs(scale_y)
        frame = cv2.flip(frame, 0)  # Flip vertical

    M = np.float32([[1, 0, 0],
                    [0, scale_y, h * (1 - scale_y) / 2]])
    return cv2.warpAffine(frame, M, (w, h))

def effect_push_blur(frame, progress, direction="left"):
    """Đẩy ngang kèm mờ dần"""
    h, w = frame.shape[:2]

    # Tạo blur
    blur_amount = int(progress * 15) * 2 + 1  # Odd number for kernel
    if blur_amount > 1:
        blurred = cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)
    else:
        blurred = frame.copy()

    # Tạo hiệu ứng push
    result = np.zeros_like(frame)

    if direction == "left":
        offset = int(progress * w)
        if offset > 0 and offset < w:
            # Push từ phải sang trái
            result[:, :w-offset] = blurred[:, offset:]
        elif offset == 0:
            result = blurred.copy()

    elif direction == "right":
        offset = int(progress * w)
        if offset > 0 and offset < w:
            # Push từ trái sang phải
            result[:, offset:] = blurred[:, :w-offset]
        elif offset == 0:
            result = blurred.copy()

    elif direction == "up":
        offset = int(progress * h)
        if offset > 0 and offset < h:
            # Push từ dưới lên trên
            result[:h-offset, :] = blurred[offset:, :]
        elif offset == 0:
            result = blurred.copy()

    elif direction == "down":
        offset = int(progress * h)
        if offset > 0 and offset < h:
            # Push từ trên xuống dưới
            result[offset:, :] = blurred[:h-offset, :]
        elif offset == 0:
            result = blurred.copy()

    return result
def effect_squeeze_horizontal(frame, progress):
    """Màn đen 2 bên ép vào rồi mở dần ra"""
    h, w = frame.shape[:2]

    if progress <= 0.5:
        # Nửa đầu: ép vào
        squeeze = progress * 2
        visible_width = int(w * (1 - squeeze))
        start_x = (w - visible_width) // 2

        result = np.zeros_like(frame)
        if visible_width > 0:
            result[:, start_x:start_x + visible_width] = frame[:, start_x:start_x + visible_width]
    else:
        # Nửa sau: mở ra
        expand = (progress - 0.5) * 2
        visible_width = int(w * expand)
        start_x = (w - visible_width) // 2

        result = np.zeros_like(frame)
        if visible_width > 0:
            result[:, start_x:start_x + visible_width] = frame[:, start_x:start_x + visible_width]

    return result

def effect_wave_distortion(frame, progress, amplitude=20):
    h, w = frame.shape[:2]

    # Tạo map tọa độ cho warp
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            map_x[i, j] = j + amplitude * progress * math.sin(2 * math.pi * i / 30)
            map_y[i, j] = i

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

def effect_pixelate(frame, progress, min_pixel_size=2, max_pixel_size=50):
    """Hiệu ứng pixel hóa"""
    h, w = frame.shape[:2]

    # Tính kích thước pixel
    pixel_size = int(min_pixel_size + (max_pixel_size - min_pixel_size) * progress)
    if pixel_size < 1:
        return frame

    # Thu nhỏ rồi phóng to để tạo hiệu ứng pixel
    small_h = max(1, h // pixel_size)
    small_w = max(1, w // pixel_size)

    small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

    pixelated = cv2.resize(small_frame, (w, h), interpolation=cv2.INTER_NEAREST)

    return pixelated
# ================================================
def effect_shatter(frame, progress, num_pieces=20):
    """Hiệu ứng vỡ tan như kính"""
    h, w = frame.shape[:2]
    result = np.zeros_like(frame)

    # Tạo các mảnh vỡ
    piece_size = min(w, h) // int(math.sqrt(num_pieces))

    for i in range(0, h, piece_size):
        for j in range(0, w, piece_size):
            # Random offset cho từng mảnh
            offset_x = int(np.random.randint(-50, 51) * progress)
            offset_y = int(np.random.randint(-50, 51) * progress)

            # Rotation cho từng mảnh
            angle = np.random.randint(-30, 31) * progress

            # Lấy mảnh
            piece = frame[i:min(i+piece_size, h), j:min(j+piece_size, w)]
            if piece.size == 0:
                continue

            # Xoay mảnh
            piece_h, piece_w = piece.shape[:2]
            if piece_h > 0 and piece_w > 0:
                M = cv2.getRotationMatrix2D((piece_w//2, piece_h//2), angle, 1)
                rotated_piece = cv2.warpAffine(piece, M, (piece_w, piece_h))

                # Đặt mảnh vào vị trí mới
                new_i = max(0, min(h - piece_h, i + offset_y))
                new_j = max(0, min(w - piece_w, j + offset_x))

                # Alpha blending để tạo hiệu ứng mờ dần
                alpha = 1.0 - progress * 0.3
                result[new_i:new_i+piece_h, new_j:new_j+piece_w] = rotated_piece * alpha

    return result.astype(np.uint8)

def effect_kaleidoscope(frame, progress, segments=8):
    """Hiệu ứng vạn hoa tổng"""
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Lấy tam giác từ center
    angle_step = 360 // segments
    mask = np.zeros((h, w), dtype=np.uint8)

    # Tạo mask cho một segment
    points = np.array([
        [center_x, center_y],
        [center_x + int(w * math.cos(math.radians(0))), center_y + int(h * math.sin(math.radians(0)))],
        [center_x + int(w * math.cos(math.radians(angle_step))), center_y + int(h * math.sin(math.radians(angle_step)))]
    ], np.int32)

    cv2.fillPoly(mask, [points], 255)

    result = np.zeros_like(frame)
    base_segment = cv2.bitwise_and(frame, frame, mask=mask)

    # Tạo các segment khác bằng cách xoay
    for i in range(segments):
        angle = i * angle_step + progress * 360  # Thêm rotation theo progress
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
        rotated = cv2.warpAffine(base_segment, M, (w, h))
        result = cv2.add(result, rotated)

    return result

def effect_page_turn(frame, progress, direction="right"):
    """Hiệu ứng lật trang sách"""
    h, w = frame.shape[:2]
    result = frame.copy()

    if direction == "right":
        # Tính toán vị trí gấp
        fold_x = int(w * (1 - progress))

        if fold_x < w:
            # Phần bị gấp
            folded_part = frame[:, fold_x:]
            folded_part = cv2.flip(folded_part, 1)  # Flip horizontal

            # Tạo shadow effect
            shadow_width = min(20, folded_part.shape[1])
            if shadow_width > 0:
                shadow = np.zeros((h, shadow_width, 3), dtype=np.uint8)
                gradient = np.linspace(0.3, 0.7, shadow_width).reshape(1, -1, 1)
                shadow = (folded_part[:, :shadow_width] * gradient).astype(np.uint8)
                folded_part[:, :shadow_width] = shadow

            # Đặt phần gấp lại
            result[:, fold_x:] = folded_part

    elif direction == "left":
        fold_x = int(w * progress)

        if fold_x > 0:
            folded_part = frame[:, :fold_x]
            folded_part = cv2.flip(folded_part, 1)

            shadow_width = min(20, folded_part.shape[1])
            if shadow_width > 0:
                shadow = np.zeros((h, shadow_width, 3), dtype=np.uint8)
                gradient = np.linspace(0.7, 0.3, shadow_width).reshape(1, -1, 1)
                shadow = (folded_part[:, -shadow_width:] * gradient).astype(np.uint8)
                folded_part[:, -shadow_width:] = shadow

            result[:, :fold_x] = folded_part

    return result

def effect_television(frame, progress):
    """Hiệu ứng tivi cũ tắt/mở"""
    h, w = frame.shape[:2]

    if progress <= 0.5:
        # Thu nhỏ về giữa như tivi tắt
        scale = 1.0 - progress * 2
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        # Resize
        resized = cv2.resize(frame, (new_w, new_h))

        # Đặt vào giữa
        result = np.zeros_like(frame)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2

        result[start_y:start_y+new_h, start_x:start_x+new_w] = resized

        # Thêm scanline effect
        if progress > 0.3:
            for i in range(0, h, 2):
                result[i, :] = result[i, :] * 0.7

    else:
        # Mở rộng ra như tivi mở
        scale = (progress - 0.5) * 2
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        if new_h <= h and new_w <= w:
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2

            cropped = frame[start_y:start_y+new_h, start_x:start_x+new_w]
            result = cv2.resize(cropped, (w, h))
        else:
            result = frame.copy()

        # Static noise effect
        if progress < 0.8:
            noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
            result = cv2.add(result, noise)

    return result

def effect_film_burn(frame, progress, intensity=50):
    """Hiệu ứng phim cháy"""
    h, w = frame.shape[:2]
    result = frame.copy()

    # Tạo burn pattern từ dưới lên
    burn_height = int(h * progress)

    if burn_height > 0:
        # Tạo mặt nạ burn với edge không đều
        burn_mask = np.zeros((h, w), dtype=np.uint8)

        for x in range(w):
            # Tạo edge không đều
            noise_offset = int(np.random.randint(-10, 11) * progress)
            burn_y = min(h, burn_height + noise_offset)
            if burn_y > 0:
                burn_mask[h-burn_y:, x] = 255

        # Tạo màu cháy (vàng cam đỏ)
        burn_color = np.zeros_like(frame)
        burn_color[:, :, 0] = 30   # Blue
        burn_color[:, :, 1] = 100  # Green
        burn_color[:, :, 2] = 255  # Red

        # Áp dụng burn effect
        burn_area = cv2.bitwise_and(burn_color, burn_color, mask=burn_mask)
        normal_area = cv2.bitwise_and(result, result, mask=cv2.bitwise_not(burn_mask))

        result = cv2.add(burn_area, normal_area)

        # Thêm flicker effect
        if np.random.random() < progress:
            brightness = np.random.randint(180, 255)
            result = cv2.addWeighted(result, 0.7, burn_color, 0.3, brightness-200)

    return result

def effect_matrix_rain(frame, progress, density=0.1):
    """Hiệu ứng ma trận digital rain"""
    h, w = frame.shape[:2]
    result = frame.copy()

    # Tạo hiệu ứng digital rain overlay
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Số lượng streams
    num_streams = int(w * density)

    for _ in range(num_streams):
        x = np.random.randint(0, w)

        # Chiều dài stream
        stream_length = np.random.randint(20, h//3)
        start_y = int((h + stream_length) * progress) % (h + stream_length)

        for i in range(stream_length):
            y = (start_y - i) % h
            if 0 <= y < h:
                # Độ sáng giảm dần theo chiều dài stream
                brightness = max(50, 255 - (i * 10))

                # Màu xanh lá Matrix
                overlay[y, x, 1] = brightness  # Green channel

                # Thêm một số ký tự sáng hơn
                if i < 3:
                    overlay[y, x, 1] = 255
                    overlay[y, x, 0] = brightness // 3  # Một chút blue

    # Blend với frame gốc
    alpha = progress * 0.7
    result = cv2.addWeighted(result, 1-alpha, overlay, alpha, 0)

    return result

def effect_old_film(frame, progress):
    """Hiệu ứng phim cũ với hạt và scratches"""
    result = frame.copy()
    h, w = frame.shape[:2]

    # Chuyển sang sepia tone
    sepia_kernel = np.array([[0.272, 0.534, 0.131],
                            [0.349, 0.686, 0.168],
                            [0.393, 0.769, 0.189]])

    result = cv2.transform(result, sepia_kernel)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Thêm noise/grain
    noise_intensity = int(30 * progress)
    if noise_intensity > 0:
        noise = np.random.randint(-noise_intensity, noise_intensity, (h, w, 3), dtype=np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Thêm vertical scratches
    num_scratches = int(5 * progress)
    for _ in range(num_scratches):
        x = np.random.randint(0, w)
        scratch_length = np.random.randint(h//4, h)
        start_y = np.random.randint(0, h - scratch_length)

        # Random scratch intensity
        scratch_color = np.random.randint(200, 255)
        result[start_y:start_y+scratch_length, x] = [scratch_color, scratch_color, scratch_color]

    # Vignette effect
    center_x, center_y = w // 2, h // 2
    max_distance = np.sqrt(center_x**2 + center_y**2)

    for i in range(h):
        for j in range(w):
            distance = np.sqrt((j - center_x)**2 + (i - center_y)**2)
            vignette_factor = 1 - (distance / max_distance) * progress * 0.5
            result[i, j] = (result[i, j] * vignette_factor).astype(np.uint8)

    return result
def effect_mosaic_blur(frame, progress, tile_size=20):
    """Hiệu ứng mosaic blur - chia thành các ô vuông và blur"""
    h, w = frame.shape[:2]
    result = frame.copy()

    # Tính tile size dựa trên progress
    current_tile_size = max(1, int(tile_size * progress))

    # Chia frame thành các tile
    for i in range(0, h, current_tile_size):
        for j in range(0, w, current_tile_size):
            # Lấy tile
            end_i = min(i + current_tile_size, h)
            end_j = min(j + current_tile_size, w)
            tile = result[i:end_i, j:end_j]

            if tile.size > 0:
                # Tính màu trung bình của tile
                avg_color = np.mean(tile.reshape(-1, 3), axis=0)
                # Gán màu trung bình cho toàn bộ tile
                result[i:end_i, j:end_j] = avg_color

    return result

def effect_lens_flare(frame, progress, intensity=100):
    """Hiệu ứng lens flare với các vòng tròn sáng"""
    h, w = frame.shape[:2]
    result = frame.copy().astype(np.float32)

    # Vị trí lens flare di chuyển theo đường chéo
    flare_x = int(w * progress)
    flare_y = int(h * (1 - progress))

    # Tạo các vòng tròn sáng với kích thước khác nhau
    flare_radii = [60, 40, 25, 15, 10]
    flare_colors = [
        (255, 255, 200),  # Vàng nhạt
        (255, 200, 150),  # Cam nhạt
        (200, 255, 255),  # Xanh nhạt
        (255, 180, 255),  # Tím nhạt
        (255, 255, 255)   # Trắng
    ]

    overlay = np.zeros_like(result)

    for i, (radius, color) in enumerate(zip(flare_radii, flare_colors)):
        # Offset cho từng vòng tròn
        offset_x = int((i - 2) * 30 * progress)
        offset_y = int((i - 2) * 20 * progress)

        center_x = flare_x + offset_x
        center_y = flare_y + offset_y

        # Vẽ gradient circle
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2

        # Tạo gradient từ center ra ngoài
        gradient = np.exp(-mask / (2 * radius**2))
        gradient = gradient * intensity * progress

        # Áp dụng màu
        for c in range(3):
            overlay[:, :, c] += gradient * (color[c] / 255.0)

    # Blend với frame gốc
    result = result + overlay
    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)

def effect_digital_glitch(frame, progress, max_blocks=20):
    """Hiệu ứng digital glitch với data corruption"""
    result = frame.copy()
    h, w = result.shape[:2]

    # Số lượng glitch blocks
    num_blocks = int(max_blocks * progress)

    for _ in range(num_blocks):
        # Random vị trí và kích thước block
        block_w = np.random.randint(10, w // 4)
        block_h = np.random.randint(5, 30)
        start_x = np.random.randint(0, w - block_w)
        start_y = np.random.randint(0, h - block_h)

        # Random shift cho block
        shift_x = np.random.randint(-20, 21)
        shift_y = np.random.randint(-5, 6)

        # Lấy block gốc
        source_block = result[start_y:start_y+block_h, start_x:start_x+block_w].copy()

        # Tính toán vị trí mới
        new_x = np.clip(start_x + shift_x, 0, w - block_w)
        new_y = np.clip(start_y + shift_y, 0, h - block_h)

        # Random color corruption
        corruption_type = np.random.randint(0, 4)
        if corruption_type == 0:
            # Channel swap
            source_block = source_block[:, :, [1, 2, 0]]
        elif corruption_type == 1:
            # Color invert
            source_block = 255 - source_block
        elif corruption_type == 2:
            # Add noise
            noise = np.random.randint(-50, 51, source_block.shape, dtype=np.int16)
            source_block = np.clip(source_block.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif corruption_type == 3:
            # Posterize
            source_block = (source_block // 64) * 64

        # Đặt block vào vị trí mới
        result[new_y:new_y+block_h, new_x:new_x+block_w] = source_block

    # Thêm horizontal line corruption
    num_lines = int(10 * progress)
    for _ in range(num_lines):
        y = np.random.randint(0, h)
        line_shift = np.random.randint(-30, 31)

        if line_shift > 0:
            result[y, line_shift:] = result[y, :-line_shift]
            result[y, :line_shift] = np.random.randint(0, 255, (line_shift, 3), dtype=np.uint8)
        elif line_shift < 0:
            result[y, :line_shift] = result[y, -line_shift:]
            result[y, line_shift:] = np.random.randint(0, 255, (-line_shift, 3), dtype=np.uint8)

    return result

def effect_waterfall(frame, progress, direction="down"):
    """Hiệu ứng thác nước - dữ liệu chảy xuống"""
    h, w = frame.shape[:2]
    result = np.zeros_like(frame)

    if direction == "down":
        # Thác chảy từ trên xuống
        reveal_height = int(h * progress)

        if reveal_height > 0:
            # Tạo hiệu ứng wave cho edge
            wave_amplitude = 10
            wave_frequency = 0.1

            for x in range(w):
                wave_offset = int(wave_amplitude * math.sin(x * wave_frequency + progress * 10))
                actual_height = min(h, reveal_height + wave_offset)

                if actual_height > 0:
                    result[:actual_height, x] = frame[:actual_height, x]

                    # Thêm hiệu ứng drop/splash tại edge
                    edge_y = actual_height - 1
                    if edge_y > 0 and edge_y < h - 1:
                        # Tăng độ sáng tại edge
                        result[edge_y, x] = np.clip(result[edge_y, x] * 1.3, 0, 255)

    elif direction == "up":
        # Thác chảy từ dưới lên
        reveal_height = int(h * progress)
        start_y = h - reveal_height

        if reveal_height > 0:
            for x in range(w):
                wave_offset = int(10 * math.sin(x * 0.1 + progress * 10))
                actual_start = max(0, start_y + wave_offset)

                result[actual_start:, x] = frame[actual_start:, x]

                # Edge highlight
                if actual_start < h - 1:
                    result[actual_start, x] = np.clip(result[actual_start, x] * 1.3, 0, 255)

    elif direction == "left":
        # Thác chảy từ phải sang trái
        reveal_width = int(w * progress)
        start_x = w - reveal_width

        if reveal_width > 0:
            for y in range(h):
                wave_offset = int(10 * math.sin(y * 0.1 + progress * 10))
                actual_start = max(0, start_x + wave_offset)

                result[y, actual_start:] = frame[y, actual_start:]

                if actual_start < w - 1:
                    result[y, actual_start] = np.clip(result[y, actual_start] * 1.3, 0, 255)

    elif direction == "right":
        # Thác chảy từ trái sang phải
        reveal_width = int(w * progress)

        if reveal_width > 0:
            for y in range(h):
                wave_offset = int(10 * math.sin(y * 0.1 + progress * 10))
                actual_width = min(w, reveal_width + wave_offset)

                if actual_width > 0:
                    result[y, :actual_width] = frame[y, :actual_width]

                    edge_x = actual_width - 1
                    if edge_x > 0:
                        result[y, edge_x] = np.clip(result[y, edge_x] * 1.3, 0, 255)

    return result

def effect_honeycomb(frame, progress, hex_size=30):
    """Hiệu ứng tổ ong - xuất hiện theo pattern lục giác"""
    h, w = frame.shape[:2]
    result = np.zeros_like(frame)

    # Tính toán grid lục giác
    hex_height = int(hex_size * math.sqrt(3) / 2)

    # Tạo mask pattern lục giác
    mask = np.zeros((h, w), dtype=np.uint8)

    # Tính số lượng hexagon cần hiện dựa trên progress
    total_hexagons = 0
    revealed_hexagons = 0

    for row in range(0, h, hex_height):
        for col in range(0, w, hex_size):
            # Offset cho hàng lẻ (tạo pattern lục giác)
            offset = (hex_size // 2) if (row // hex_height) % 2 == 1 else 0
            hex_x = col + offset

            total_hexagons += 1

            # Quyết định hexagon này có hiện không dựa trên progress
            if revealed_hexagons < int(total_hexagons * progress):
                # Vẽ lục giác
                center_x = hex_x + hex_size // 2
                center_y = row + hex_height // 2

                # Tạo các điểm của lục giác
                points = []
                for i in range(6):
                    angle = i * math.pi / 3
                    x = int(center_x + (hex_size // 2) * math.cos(angle))
                    y = int(center_y + (hex_size // 2) * math.sin(angle))
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)

                revealed_hexagons += 1

    # Áp dụng mask
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Thêm border cho các hexagon
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 1)

    return result

# ===================================
# Hàm xử lý chính với MoviePy + OpenCV
# ===================================
def apply_effect(video_path, output_path, start_time, end_time, effect_name, **kwargs):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    temp_video = "temp_no_audio.mp4"
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_idx <= end_frame:
            progress = (frame_idx - start_frame) / (end_frame - start_frame)
            if effect_name == "slide":
                frame = effect_slide(frame, progress, kwargs.get("direction", "horizontal"))
            elif effect_name == "rotate":
                frame = effect_rotate(frame, progress, kwargs.get("max_angle", 180))
            elif effect_name == "circle_mask":
                frame = effect_circle_mask(frame, progress)
            elif effect_name == "fade_in":
                frame = effect_fade_in(frame, progress)
            elif effect_name == "fade_out":
                frame = effect_fade_out(frame, progress)
            elif effect_name == "fadeout_fadein":
                frame = effect_fadeout_fadein(frame, progress)
            elif effect_name == "crossfade" and prev_frame is not None:
                frame = effect_crossfade(prev_frame, frame, progress)
            elif effect_name == "rgb_split":
                frame = effect_rgb_split(frame, progress, kwargs.get("max_offset", 20))
            elif effect_name == "flip_horizontal":
                frame = effect_flip_horizontal(frame, progress)
            elif effect_name == "flip_vertical":
                frame = effect_flip_vertical(frame, progress)
            elif effect_name == "push_blur":
                frame = effect_push_blur(frame, progress, kwargs.get("direction", "left"))
            elif effect_name == "squeeze_horizontal":
                frame = effect_squeeze_horizontal(frame, progress)
            elif effect_name == "wave_distortion":
                frame = effect_wave_distortion(frame, progress, kwargs.get("amplitude", 20))
            elif effect_name == "zoom_blur":
                frame = effect_zoom_blur(frame, progress, kwargs.get("max_zoom", 1.5))
            elif effect_name == "spiral":
                frame = effect_spiral(frame, progress)
            elif effect_name == "pixelate":
                frame = effect_pixelate(frame, progress,
                                      kwargs.get("min_pixel_size", 2),
                                      kwargs.get("max_pixel_size", 50))
            elif effect_name == "shatter":
                frame = effect_shatter(frame, progress, kwargs.get("num_pieces", 20))
            elif effect_name == "kaleidoscope":
                frame = effect_kaleidoscope(frame, progress, kwargs.get("segments", 8))
            elif effect_name == "page_turn":
                frame = effect_page_turn(frame, progress, kwargs.get("direction", "right"))
            elif effect_name == "television":
                frame = effect_television(frame, progress)
            elif effect_name == "film_burn":
                frame = effect_film_burn(frame, progress, kwargs.get("intensity", 50))
            elif effect_name == "matrix_rain":
                frame = effect_matrix_rain(frame, progress, kwargs.get("density", 0.1))
            elif effect_name == "old_film":
                frame = effect_old_film(frame, progress)
            elif effect_name == "mosaic_blur":
                frame = effect_mosaic_blur(frame, progress, kwargs.get("tile_size", 20))
            elif effect_name == "lens_flare":
                frame = effect_lens_flare(frame, progress, kwargs.get("intensity", 100))
            elif effect_name == "digital_glitch":
                frame = effect_digital_glitch(frame, progress, kwargs.get("max_blocks", 20))
            elif effect_name == "waterfall":
                frame = effect_waterfall(frame, progress, kwargs.get("direction", "down"))
            elif effect_name == "honeycomb":
                frame = effect_honeycomb(frame, progress, kwargs.get("hex_size", 30))
            else:
                pass

        prev_frame = frame.copy()
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    try:
        original_clip = VideoFileClip(video_path)
        processed_clip = VideoFileClip(temp_video).set_audio(original_clip.audio)
        processed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        original_clip.close()
        processed_clip.close()
    except Exception as e:
        print(f"Error processing audio: {e}")
        # Nếu có lỗi với audio, chỉ copy video
        import shutil
        shutil.move(temp_video, output_path)

