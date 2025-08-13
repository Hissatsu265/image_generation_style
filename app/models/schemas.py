from pydantic import BaseModel
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
