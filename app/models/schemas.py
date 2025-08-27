from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Resolution(str, Enum):
    HD_720P = "720"
    FHD_1080P = "1080"
    QHD_1440P = "1440"
    UHD_4K = "2160"
    SQUARE_1080 = "1080"  
    VERTICAL_1080 = "1920"
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoCreateRequest(BaseModel):
    image_paths: List[str]
    prompts: List[str]
    audio_path: str
    resolution: Resolution = Resolution.HD_720P

class VideoCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: Optional[int] = None
    video_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[int] = None
    is_processing: Optional[bool] = None
    current_processing_job: Optional[str] = None
# ============================VIDEO EFFECT==========================================================
# class TransitionEffect(str, Enum):
#     FADE = "fade"
#     DISSOLVE = "dissolve"
#     WIPE_LEFT = "wipe_left"
#     WIPE_RIGHT = "wipe_right"
#     SLIDE_UP = "slide_up"
#     SLIDE_DOWN = "slide_down"
#     ZOOM_IN = "zoom_in"
#     ZOOM_OUT = "zoom_out"
#     ROTATE = "rotate"
#     BLUR = "blur"

# class DollyEffectType(str, Enum):
#     ZOOM_GRADUAL = "zoom_gradual"  # Zoom từ từ
#     DOUBLE_ZOOM = "double_zoom"    # Double zoom
#     ZOOM_IN_OUT = "zoom_in_out"    # Zoom vào rồi ra
#     PAN_ZOOM = "pan_zoom"          # Pan kết hợp zoom

# class DollyEffect(BaseModel):
#     scene_index: int = Field(..., description="Cảnh áp dụng (bắt đầu từ 0)")
#     start_time: float = Field(..., description="Thời gian bắt đầu (giây)")
#     duration: float = Field(..., description="Thời gian áp dụng (giây)")
#     zoom_percent: float = Field(..., ge=10, le=500, description="Zoom bao nhiêu % (10-500%)")
#     effect_type: DollyEffectType = Field(..., description="Loại hiệu ứng dolly")
#     x_coordinate: Optional[float] = Field(None, description="Tọa độ X (0-1, tùy chọn)")
#     y_coordinate: Optional[float] = Field(None, description="Tọa độ Y (0-1, tùy chọn)")

# class VideoEffectRequest(BaseModel):
#     video_path: str = Field(..., description="Đường dẫn video input")
#     transition_times: List[float] = Field(..., description="Thời điểm chuyển cảnh (giây)")
#     transition_effects: List[TransitionEffect] = Field(..., description="Hiệu ứng chuyển cảnh")
#     transition_durations: List[float] = Field(..., description="Thời gian duration từng hiệu ứng (giây)")
#     dolly_effects: Optional[List[DollyEffect]] = Field(default=[], description="Danh sách hiệu ứng dolly (tùy chọn)")
class TransitionEffect(str, Enum):
    SLIDE = "slide"
    ROTATE = "rotate"
    CIRCLE_MASK = "circle_mask"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    FADEOUT_FADEIN = "fadeout_fadein"
    CROSSFADE = "crossfade"
    RGB_SPLIT = "rgb_split"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    PUSH_BLUR = "push_blur"
    SQUEEZE_HORIZONTAL = "squeeze_horizontal"
    WAVE_DISTORTION = "wave_distortion"
    ZOOM_BLUR = "zoom_blur"
    SPIRAL = "spiral"
    PIXELATE = "pixelate"
    SHATTER = "shatter"
    KALEIDOSCOPE = "kaleidoscope"
    PAGE_TURN = "page_turn"
    TELEVISION = "television"
    FILM_BURN = "film_burn"
    MATRIX_RAIN = "matrix_rain"
    OLD_FILM = "old_film"
    MOSAIC_BLUR = "mosaic_blur"
    LENS_FLARE = "lens_flare"
    DIGITAL_GLITCH = "digital_glitch"
    WATERFALL = "waterfall"
    HONEYCOMB = "honeycomb"
    NONE="none"

class DollyEffectType(str, Enum):
    AUTO_ZOOM = "auto_zoom"        # Zoom tự động
    MANUAL_ZOOM = "manual_zoom"    # Zoom thủ công (có tọa độ X, Y)
    DOUBLE_ZOOM = "double_zoom"    # Double zoom
    # PAN_ZOOM = "pan_zoom"          # Pan kết hợp zoom

# class DollyEffect(BaseModel):
#     scene_index: int = Field(..., description="Cảnh áp dụng (bắt đầu từ 0)")
#     start_time: float = Field(..., description="Thời gian bắt đầu (giây)")
#     duration: float = Field(..., description="Thời gian áp dụng (giây)")
#     zoom_percent: float = Field(..., ge=10, le=500, description="Zoom bao nhiêu % (10-500%)")
#     effect_type: DollyEffectType = Field(..., description="Loại hiệu ứng dolly")
#     x_coordinate: Optional[float] = Field(None, description="Tọa độ X (0-1, tùy chọn)")
#     y_coordinate: Optional[float] = Field(None, description="Tọa độ Y (0-1, tùy chọn)")
class DollyEndType(str, Enum):
    smooth = "smooth"
    instant = "instant"

class DollyEffect(BaseModel):
    scene_index: int = Field(None, description="Cảnh áp dụng (bắt đầu từ 0)")
    start_time: float = Field(0.0, description="Thời gian bắt đầu (giây)")
    duration: float = Field(0.5, description="Thời gian áp dụng (giây)")
    zoom_percent: float = Field(0, ge=10, le=100, description="Zoom bao nhiêu % (0-100%)")
    effect_type: DollyEffectType = Field(..., description="Loại hiệu ứng dolly")
    x_coordinate: Optional[float] = Field(None, description="Tọa độ X (0-1, tùy chọn)")
    y_coordinate: Optional[float] = Field(None, description="Tọa độ Y (0-1, tùy chọn)")
    
    end_time: Optional[float] = Field(
        None, description="Thời gian kết thúc hiệu ứng (giây, tùy chọn)"
    )
    end_type: DollyEndType = Field(
        DollyEndType.smooth, description="Loại kết thúc hiệu ứng: smooth hoặc instant"
    )
class VideoEffectRequest(BaseModel):
    video_path: str = Field(..., description="Đường dẫn video input")
    transition_times: List[float] = Field(..., description="Thời điểm chuyển cảnh (giây)")
    transition_effects: List[TransitionEffect] = Field(..., description="Hiệu ứng chuyển cảnh")
    transition_durations: List[float] = Field(..., description="Thời gian duration từng hiệu ứng (giây)")
    dolly_effects: Optional[List[DollyEffect]] = Field(default=[], description="Danh sách hiệu ứng dolly (tùy chọn)")

class VideoEffectResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    queue_position: int
    estimated_wait_time: int  # minutes
    available_workers: int
    total_workers: int

class EffectJobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int
    video_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[int] = None
    worker_id: Optional[int] = None  