import asyncio
import json
import uuid
import os
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
import redis.asyncio as redis
from app.models.schemas import JobStatus

class JobService:
    def __init__(self):
        self.redis_client = None
        
        # Existing video creation jobs
        self.job_queue = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.processing = False
        self.cleanup_task = None
        self.video_processing_lock = asyncio.Lock()
        self.current_processing_job = None
        self.worker_task = None
        
        # NEW: Effect jobs system
        self.effect_jobs: Dict[str, Dict[str, Any]] = {}
        self.effect_job_queue = asyncio.Queue()
        self.effect_workers: List[asyncio.Task] = []
        self.effect_processing_locks: List[asyncio.Lock] = []
        self.effect_workers_busy: List[bool] = []
        self.max_effect_workers = self._calculate_effect_workers()
        
        # Initialize effect workers
        self._init_effect_workers()

    def _calculate_effect_workers(self) -> int:
        """Tính số effect workers = số CPU cores / 2, tối thiểu 1"""
        cpu_count = psutil.cpu_count(logical=False) or 2
        workers = max(1, cpu_count // 2)
        print(f"Initializing {workers} effect workers (CPU cores: {cpu_count})")
        return workers

    def _init_effect_workers(self):
        """Khởi tạo effect workers và locks"""
        self.effect_processing_locks = [asyncio.Lock() for _ in range(self.max_effect_workers)]
        self.effect_workers_busy = [False for _ in range(self.max_effect_workers)]

    # === EXISTING METHODS (giữ nguyên) ===
    async def init_redis(self):
        """Initialize Redis connection"""
        try:
            from config import REDIS_URL
            self.redis_client = redis.from_url(REDIS_URL)
            await self.redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_client = None

    # [Giữ nguyên tất cả existing methods: cleanup, create_job, get_job_status, etc.]
    
    # === NEW EFFECT JOBS METHODS ===
    
    async def create_effect_job(self, 
                              video_path: str,
                              transition_times: List[float],
                              transition_effects: List[str],
                              transition_durations: List[float],
                              dolly_effects: List[Dict] = None) -> str:
        """Tạo effect job mới - NON-BLOCKING"""
        
        job_id = str(uuid.uuid4())
        
        job_data = {
            "job_id": job_id,
            "job_type": "effect",  # Phân biệt với video creation job
            "status": JobStatus.PENDING,
            "video_path": video_path,
            "transition_times": transition_times,
            "transition_effects": transition_effects,
            "transition_durations": transition_durations,
            "dolly_effects": dolly_effects or [],
            "progress": 0,
            "output_video_path": None,
            "error_message": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "queue_position": self.effect_job_queue.qsize() + 1,
            "worker_id": None
        }
        
        # Lưu vào memory và Redis
        self.effect_jobs[job_id] = job_data
        if self.redis_client:
            asyncio.create_task(self._save_effect_job_to_redis(job_id, job_data))
        
        # Thêm vào effect queue
        await self.effect_job_queue.put(job_data)
        
        # Đảm bảo effect workers đang chạy
        await self.start_effect_workers()
        
        return job_id

    async def _save_effect_job_to_redis(self, job_id: str, job_data: dict):
        """Save effect job to Redis"""
        try:
            await self.redis_client.set(
                f"effect_job:{job_id}",
                json.dumps(job_data, default=str),
                ex=86400  # 24h
            )
        except Exception as e:
            print(f"Error saving effect job to Redis: {e}")

    async def start_effect_workers(self):
        """Khởi động các effect workers nếu chưa chạy"""
        
        # Clean up completed workers
        self.effect_workers = [w for w in self.effect_workers if not w.done()]
        
        # Start workers if needed
        workers_needed = self.max_effect_workers - len(self.effect_workers)
        
        for i in range(workers_needed):
            worker_id = len(self.effect_workers)
            worker_task = asyncio.create_task(self.process_effect_jobs(worker_id))
            self.effect_workers.append(worker_task)
            print(f"Started effect worker {worker_id}")

    async def process_effect_jobs(self, worker_id: int):
        """Effect job worker - xử lý nhiều jobs đồng thời"""
        print(f"Effect worker {worker_id} started")
        
        while True:
            try:
                # Lấy job từ queue
                job_data = await self.effect_job_queue.get()
                job_id = job_data["job_id"]
                
                print(f"Effect worker {worker_id} processing job: {job_id}")
                
                # Acquire lock cho worker này
                async with self.effect_processing_locks[worker_id]:
                    self.effect_workers_busy[worker_id] = True
                    
                    # Cập nhật job status
                    await self.update_effect_job_status(
                        job_id, 
                        JobStatus.PROCESSING, 
                        progress=10,
                        worker_id=worker_id
                    )
                    
                    try:
                        # Import effect service
                        from app.services.video_effect_service import VideoEffectService
                        effect_service = VideoEffectService()
                        
                        # Validate video duration trước
                        video_duration = await effect_service.get_video_duration(job_data["video_path"])
                        await self.update_effect_job_status(job_id, JobStatus.PROCESSING, progress=20)
                        
                        # Validate effects timing
                        effect_service.validate_effects_timing(
                            job_data["transition_times"],
                            job_data["dolly_effects"],
                            video_duration
                        )
                        await self.update_effect_job_status(job_id, JobStatus.PROCESSING, progress=30)
                        
                        # Process video effects
                        output_path = await effect_service.apply_effects(
                            video_path=job_data["video_path"],
                            transition_times=job_data["transition_times"],
                            transition_effects=job_data["transition_effects"],
                            transition_durations=job_data["transition_durations"],
                            dolly_effects=job_data["dolly_effects"],
                            job_id=job_id
                        )
                        
                        await self.update_effect_job_status(
                            job_id,
                            JobStatus.COMPLETED,
                            progress=100,
                            output_video_path=output_path
                        )
                        
                        print(f"Effect job completed: {job_id} by worker {worker_id}")
                        
                    except Exception as e:
                        print(f"Effect job failed: {job_id}, Error: {e}")
                        await self.update_effect_job_status(
                            job_id,
                            JobStatus.FAILED,
                            error_message=str(e)
                        )
                    
                    finally:
                        self.effect_workers_busy[worker_id] = False
                
                # Mark task done
                self.effect_job_queue.task_done()
                
            except Exception as e:
                print(f"Error in effect worker {worker_id}: {e}")
                self.effect_workers_busy[worker_id] = False
                await asyncio.sleep(1)

    async def get_effect_job_status(self, job_id: str) -> Dict[str, Any]:
        """Lấy trạng thái effect job"""
        
        # Check memory first
        if job_id in self.effect_jobs:
            job_data = self.effect_jobs[job_id].copy()
            
            # Update queue position for pending jobs
            if job_data["status"] == JobStatus.PENDING:
                position = await self.get_effect_queue_position(job_id)
                job_data["queue_position"] = position
            
            return job_data
        
        # Check Redis
        if self.redis_client:
            try:
                job_data = await self.redis_client.get(f"effect_job:{job_id}")
                if job_data:
                    return json.loads(job_data)
            except Exception as e:
                print(f"Error getting effect job from Redis: {e}")
        
        return None

    async def get_effect_queue_position(self, job_id: str) -> int:
        """Lấy vị trí trong effect queue"""
        position = 1
        temp_queue = []
        found = False
        
        try:
            while not self.effect_job_queue.empty():
                item = await asyncio.wait_for(self.effect_job_queue.get(), timeout=0.1)
                temp_queue.append(item)
                if item["job_id"] == job_id:
                    found = True
                    break
                position += 1
            
            # Put items back
            for item in reversed(temp_queue):
                await self.effect_job_queue.put(item)
            
            return position if found else 0
        except:
            for item in reversed(temp_queue):
                await self.effect_job_queue.put(item)
            return 0

    async def update_effect_job_status(self, job_id: str, status: JobStatus, **kwargs):
        """Cập nhật effect job status"""
        
        if job_id in self.effect_jobs:
            self.effect_jobs[job_id]["status"] = status
            for key, value in kwargs.items():
                self.effect_jobs[job_id][key] = value
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                self.effect_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            # Update Redis
            if self.redis_client:
                asyncio.create_task(self._update_effect_job_in_redis(job_id))

    async def _update_effect_job_in_redis(self, job_id: str):
        """Update effect job in Redis"""
        try:
            await self.redis_client.set(
                f"effect_job:{job_id}",
                json.dumps(self.effect_jobs[job_id], default=str),
                ex=86400
            )
        except Exception as e:
            print(f"Error updating effect job in Redis: {e}")

    async def get_effect_workers_info(self):
        """Lấy thông tin về effect workers"""
        
        available_workers = sum(1 for busy in self.effect_workers_busy if not busy)
        busy_workers = sum(1 for busy in self.effect_workers_busy if busy)
        
        return {
            "total_workers": self.max_effect_workers,
            "available_workers": available_workers,
            "busy_workers": busy_workers,
            "pending_jobs": self.effect_job_queue.qsize(),
            "workers_status": [
                {
                    "worker_id": i,
                    "is_busy": self.effect_workers_busy[i],
                    "is_running": i < len(self.effect_workers) and not self.effect_workers[i].done()
                }
                for i in range(self.max_effect_workers)
            ]
        }

    async def cancel_effect_job(self, job_id: str) -> bool:
        """Hủy effect job nếu đang pending"""
        
        if job_id in self.effect_jobs:
            job = self.effect_jobs[job_id]
            if job["status"] == JobStatus.PENDING:
                await self.update_effect_job_status(
                    job_id, 
                    JobStatus.FAILED, 
                    error_message="Job cancelled by user"
                )
                return True
            elif job["status"] == JobStatus.PROCESSING:
                return False  # Cannot cancel processing job
        return False

    # === UPDATED STATS METHOD ===
    async def get_stats(self):
        """Lấy thống kê bao gồm cả effect jobs"""
        from config import OUTPUT_DIR
        
        # Existing stats
        memory_jobs = len(self.jobs)
        status_count = {}
        for job_data in self.jobs.values():
            status = job_data.get("status", "unknown")
            status_count[status] = status_count.get(status, 0) + 1
        
        # Effect jobs stats  
        effect_memory_jobs = len(self.effect_jobs)
        effect_status_count = {}
        for job_data in self.effect_jobs.values():
            status = job_data.get("status", "unknown")
            effect_status_count[status] = effect_status_count.get(status, 0) + 1
        
        # Redis stats
        redis_jobs = 0
        redis_effect_jobs = 0
        if self.redis_client:
            try:
                redis_jobs = len([key async for key in self.redis_client.scan_iter(match="job:*")])
                redis_effect_jobs = len([key async for key in self.redis_client.scan_iter(match="effect_job:*")])
            except:
                redis_jobs = redis_effect_jobs = -1
        
        # Video files stats
        output_path = Path(OUTPUT_DIR)
        video_files = len(list(output_path.glob("*.mp4"))) if output_path.exists() else 0
        total_size = 0
        if output_path.exists():
            for video_file in output_path.glob("*.mp4"):
                try:
                    total_size += video_file.stat().st_size
                except:
                    pass
        
        # Workers info
        effect_workers_info = await self.get_effect_workers_info()
        
        return {
            "video_creation_jobs": {
                "memory": memory_jobs,
                "redis": redis_jobs,
                "status_breakdown": status_count,
                "queue_pending": self.job_queue.qsize(),
                "current_processing": self.current_processing_job,
                "is_processing": self.video_processing_lock.locked()
            },
            "effect_jobs": {
                "memory": effect_memory_jobs,
                "redis": redis_effect_jobs,
                "status_breakdown": effect_status_count,
                "queue_pending": self.effect_job_queue.qsize(),
                "workers": effect_workers_info
            },
            "videos": {
                "count": video_files,
                "total_size_mb": round(total_size / 1024 / 1024, 2)
            },
            "system": {
                "cpu_cores": psutil.cpu_count(logical=False),
                "max_effect_workers": self.max_effect_workers
            }
        }

# Global instance
job_service = JobService()