import asyncio
import json
import uuid
import os
import psutil

from datetime import datetime, timedelta
from typing import Dict, Any
from pathlib import Path
import redis.asyncio as redis
from app.models.schemas import JobStatus
from typing import Dict, Any, List
from app.models.mongodb import mongodb
from app.services.job_repository import job_repository
class JobService:
    def __init__(self):
        self.redis_client = None
        self.job_queue = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.processing = False
        self.cleanup_task = None
        self.video_processing_lock = asyncio.Lock()  # Lock cho video processing
        self.current_processing_job = None  # Track job đang được xử lý
        self.worker_task = None  # Task của worker

        self.job_queue = asyncio.Queue()
        self.effect_job_queue = asyncio.Queue()
        # NEW: Effect jobs system
        self.effect_jobs: Dict[str, Dict[str, Any]] = {}
        self.effect_job_queue = asyncio.Queue()
        self.effect_workers: List[asyncio.Task] = []
        self.effect_processing_locks: List[asyncio.Lock] = []
        self.effect_workers_busy: List[bool] = []
        self.max_effect_workers = self._calculate_effect_workers()
        self._init_effect_workers()
    async def init_mongodb(self):
        """Khởi tạo MongoDB connection"""
        try:
            await mongodb.connect()
            print("MongoDB initialized successfully")
        except Exception as e:
            print(f"MongoDB initialization failed: {e}")
            raise
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
    async def init_redis(self):
        """Initialize Redis connection"""
        try:
            from config import REDIS_URL
            self.redis_client = redis.from_url(REDIS_URL)
            await self.redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_client = None

    async def start_cleanup_task(self):
        """Bắt đầu task cleanup tự động"""
        if not self.cleanup_task or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self.periodic_cleanup())

    async def periodic_cleanup(self):
        """Task cleanup chạy định kỳ"""
        from config import CLEANUP_INTERVAL_MINUTES
        
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL_MINUTES * 60)  # Convert to seconds
                await self.cleanup_old_jobs()
                await self.cleanup_old_videos()
                print(f"Cleanup completed at {datetime.now()}")
            except Exception as e:
                print(f"Cleanup error: {e}")

    async def cleanup_old_jobs(self):
        """Xóa job cũ khỏi memory và Redis"""
        from config import JOB_RETENTION_HOURS
        
        cutoff_time = datetime.now() - timedelta(hours=JOB_RETENTION_HOURS)
        jobs_to_remove = []
        
        # Cleanup memory jobs
        for job_id, job_data in self.jobs.items():
            try:
                created_at = datetime.fromisoformat(job_data["created_at"])
                if created_at < cutoff_time:
                    jobs_to_remove.append(job_id)
            except (KeyError, ValueError):
                # Invalid job data, mark for removal
                jobs_to_remove.append(job_id)
        
        # Remove from memory
        for job_id in jobs_to_remove:
            self.jobs.pop(job_id, None)
            print(f"Removed job from memory: {job_id}")
        
        # Cleanup Redis jobs
        if self.redis_client:
            try:
                # Scan for job keys
                async for key in self.redis_client.scan_iter(match="job:*"):
                    job_data = await self.redis_client.get(key)
                    if job_data:
                        try:
                            data = json.loads(job_data)
                            created_at = datetime.fromisoformat(data["created_at"])
                            if created_at < cutoff_time:
                                await self.redis_client.delete(key)
                                print(f"Removed job from Redis: {key}")
                        except (json.JSONDecodeError, KeyError, ValueError):
                            # Invalid job data, remove it
                            await self.redis_client.delete(key)
                            print(f"Removed invalid job from Redis: {key}")
            except Exception as e:
                print(f"Redis cleanup error: {e}")

    async def cleanup_old_videos(self):
        """Xóa video files cũ"""
        from config import VIDEO_RETENTION_HOURS, OUTPUT_DIR
        
        cutoff_time = datetime.now() - timedelta(hours=VIDEO_RETENTION_HOURS)
        output_path = Path(OUTPUT_DIR)
        
        if not output_path.exists():
            return
        
        files_removed = 0
        for video_file in output_path.glob("*.mp4"):
            try:
                # Lấy thời gian tạo file
                file_time = datetime.fromtimestamp(video_file.stat().st_ctime)
                if file_time < cutoff_time:
                    video_file.unlink()
                    files_removed += 1
                    print(f"Removed old video: {video_file.name}")
            except Exception as e:
                print(f"Error removing video {video_file.name}: {e}")
        
        if files_removed > 0:
            print(f"Cleanup completed: {files_removed} video files removed")



    async def manual_cleanup(self):
        """Cleanup thủ công"""
        await self.cleanup_old_jobs()
        await self.cleanup_old_videos()
        return {"message": "Manual cleanup completed"}

    # async def create_job(self, image_paths: list, prompts: list, audio_path: str, resolution: str = "1920x1080", background: str | None = None) -> str:
    #     """Tạo job mới và thêm vào queue - NON-BLOCKING"""
    #     job_id = str(uuid.uuid4())
        
    #     job_data = {
    #         "job_id": job_id,
    #         "status": JobStatus.PENDING,
    #         "image_paths": image_paths,
    #         "prompts": prompts,
    #         "audio_path": audio_path,
    #         "resolution": resolution,
    #         "background": background,
    #         "progress": 0,
    #         "video_path": None,
    #         "error_message": None,
    #         "created_at": datetime.now().isoformat(),
    #         "completed_at": None,
    #         "queue_position": self.job_queue.qsize() + 1  # Vị trí trong queue
    #     }
        
    #     # Lưu vào memory và Redis (nếu có) - NON-BLOCKING
    #     self.jobs[job_id] = job_data
    #     if self.redis_client:
    #         # Sử dụng create_task để không block
    #         asyncio.create_task(self._save_job_to_redis(job_id, job_data))
        
    #     # Thêm vào queue - NON-BLOCKING
    #     await self.job_queue.put(job_data)
        
    #     # Bắt đầu worker nếu chưa chạy
    #     await self.start_worker()
        
    #     # Bắt đầu cleanup task nếu chưa chạy
    #     asyncio.create_task(self.start_cleanup_task())
        
    #     return job_id
    async def create_job(self, image_paths: list, prompts: list,  style:str,resolution: str = "1920x1080") -> str:
        """Tạo job mới - LƯU VÀO MONGODB"""
        job_id = str(uuid.uuid4())
        
        job_data = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "image_paths": image_paths,
            "prompts": prompts,
            "style": style,
            "resolution": resolution,
            "progress": 0,
            "video_path": None,
            "error_message": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "queue_position": self.job_queue.qsize() + 1
        }
        
        # LƯU VÀO MONGODB thay vì RAM
        await job_repository.insert_job(job_data)
        
        # Thêm vào queue (giữ nguyên)
        await self.job_queue.put(job_data)
        
        # Start worker và cleanup (giữ nguyên)
        await self.start_worker()
        asyncio.create_task(self.start_cleanup_task())
        
        return job_id

    async def _save_job_to_redis(self, job_id: str, job_data: dict):
        """Helper method để save job to Redis không block"""
        try:
            await self.redis_client.set(
                f"job:{job_id}", 
                json.dumps(job_data, default=str),
                ex=86400  
            )
        except Exception as e:
            print(f"Error saving job to Redis: {e}")

    async def start_worker(self):
        """Bắt đầu worker nếu chưa chạy"""
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.process_jobs())
            self.processing = True

    # async def get_job_status(self, job_id: str) -> Dict[str, Any]:
    #     """Lấy trạng thái job - NON-BLOCKING"""
    #     # Kiểm tra trong memory trước
    #     if job_id in self.jobs:
    #         job_data = self.jobs[job_id].copy()
            
    #         # Cập nhật queue position cho job pending
    #         if job_data["status"] == JobStatus.PENDING:
    #             position = await self.get_queue_position(job_id)
    #             job_data["queue_position"] = position
            
    #         return job_data
        
    #     # Kiểm tra trong Redis - sử dụng create_task để không block
    #     if self.redis_client:
    #         try:
    #             job_data = await self.redis_client.get(f"job:{job_id}")
    #             if job_data:
    #                 return json.loads(job_data)
    #         except Exception as e:
    #             print(f"Error getting job from Redis: {e}")
        
    #     return None
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Lấy job status từ MongoDB"""
        
        # LẤY TỪ MONGODB thay vì RAM
        job_data = await job_repository.find_job_by_id(job_id)
        
        if not job_data:
            return None
            
        # Cập nhật queue position cho pending jobs
        if job_data["status"] == JobStatus.PENDING:
            position = await self.get_queue_position(job_id)
            job_data["queue_position"] = position
        
        return job_data

    async def get_queue_position(self, job_id: str) -> int:
        """Lấy vị trí job trong queue"""
        position = 1
        temp_queue = []
        found = False
        
        # Tạm thời lấy items từ queue để tìm position
        try:
            while not self.job_queue.empty():
                item = await asyncio.wait_for(self.job_queue.get(), timeout=0.1)
                temp_queue.append(item)
                if item["job_id"] == job_id:
                    found = True
                    break
                position += 1
            
            # Đưa items trở lại queue
            for item in reversed(temp_queue):
                await self.job_queue.put(item)
            
            return position if found else 0
        except:
            # Nếu có lỗi, đưa items trở lại queue
            for item in reversed(temp_queue):
                await self.job_queue.put(item)
            return 0

    # async def update_job_status(self, job_id: str, status: JobStatus, **kwargs):
    #     if job_id in self.jobs:
    #         self.jobs[job_id]["status"] = status
    #         for key, value in kwargs.items():
    #             self.jobs[job_id][key] = value
            
    #         if status == JobStatus.COMPLETED or status == JobStatus.FAILED:
    #             self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
    #         if self.redis_client:
    #             asyncio.create_task(self._update_job_in_redis(job_id))
    async def update_job_status(self, job_id: str, status: JobStatus, **kwargs):
        """Cập nhật job status trong MongoDB"""
        
        update_data = {"status": status}
        update_data.update(kwargs)
        
        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            update_data["completed_at"] = datetime.now().isoformat()
        
        # CẬP NHẬT MONGODB thay vì RAM
        await job_repository.update_job(job_id, update_data)

    async def _update_job_in_redis(self, job_id: str):
        """Helper method để update job in Redis không block"""
        print("Sử dụng redis trong file JobService")
        try:
            await self.redis_client.set(
                f"job:{job_id}",
                json.dumps(self.jobs[job_id], default=str),
                ex=86400
            )
        except Exception as e:
            print(f"Error updating job in Redis: {e}")

    async def process_jobs(self):
        print("Job worker started")
        
        while True:
            try:
                # Lấy job từ queue
                job_data = await self.job_queue.get()
                job_id = job_data["job_id"]
                
                print(f"Processing job: {job_id}")
                
                async with self.video_processing_lock:
                    self.current_processing_job = job_id
                    
                    # Cập nhật status thành processing
                    await self.update_job_status(job_id, JobStatus.PROCESSING, progress=10)
                    
                    # Import video service
                    from app.services.video_service import VideoService
                    video_service = VideoService()
                    
                    try:
                        print(f"Creating video for job: {job_id}")
                        list_img = await video_service.create_video(
                            image_paths=job_data["image_paths"],
                            prompts=job_data["prompts"],
                            style=job_data["style"],
                            resolution=job_data["resolution"],
                            job_id=job_id
                        )
                        # print("fsfssfsdfsfs: ", list_scene)
                        await self.update_job_status(
                            job_id, 
                            JobStatus.COMPLETED, 
                            progress=100,
                            # video_path=video_path,
                            list_img=list_img
                        )
                        
                        print(f"Job completed: {job_id}")
                        
                    except Exception as e:
                        print(f"Job failed: {job_id}, Error: {e}")
                        await self.update_job_status(
                            job_id,
                            JobStatus.FAILED,
                            error_message=str(e)
                        )
                    
                    finally:
                        self.current_processing_job = None
                
                self.job_queue.task_done()
                
            except Exception as e:
                print(f"Error in job worker: {e}")
                await asyncio.sleep(1)

    async def get_queue_info(self):
        return {
            "pending_jobs": self.job_queue.qsize(),
            "current_processing": self.current_processing_job,
            "is_processing": self.video_processing_lock.locked(),
            "worker_running": self.processing
        }

    async def cancel_job(self, job_id: str) -> bool:
        """Hủy job nếu đang pending - NON-BLOCKING"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if job["status"] == JobStatus.PENDING:
                await self.update_job_status(job_id, JobStatus.FAILED, error_message="Job cancelled by user")
                return True
            elif job["status"] == JobStatus.PROCESSING:
                return False  # Không thể hủy job đang processing
        return False
    # ============================VIDEO EFFECT JOBS SYSTEM==========================================================
    
    async def create_effect_job(self, 
                              video_path: str,
                              transition_times: List[float],
                              transition_effects: List[str],
                              transition_durations: List[float],
                              dolly_effects: List[Dict] = None) -> str:
        
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
                        
                        # Validate video duration và timing - sử dụng executor
                        loop = asyncio.get_event_loop()
                        video_duration = await loop.run_in_executor(
                            None, 
                            effect_service.get_video_duration_sync,
                            job_data["video_path"]
                        )
                        await self.update_effect_job_status(job_id, JobStatus.PROCESSING, progress=20)
                        
                        # Validate effects timing trong executor
                        await loop.run_in_executor(
                            None,
                            effect_service.validate_effects_timing,
                            job_data["transition_times"],
                            job_data["dolly_effects"], 
                            video_duration
                        )
                        await self.update_effect_job_status(job_id, JobStatus.PROCESSING, progress=30)
                        
                        # Process video effects - Use thread pool để không block
                        import concurrent.futures
                        
                        # Tạo wrapper function cho sync processing
                        def process_video_sync():
                            # Gọi sync version của effect processing
                            return effect_service.apply_effects_sync(
                                video_path=job_data["video_path"],
                                transition_times=job_data["transition_times"],
                                transition_effects=job_data["transition_effects"],
                                transition_durations=job_data["transition_durations"],
                                dolly_effects=job_data["dolly_effects"],
                                job_id=job_id
                            )
                        
                        # Chạy trong thread pool để không block event loop
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            output_path = await loop.run_in_executor(executor, process_video_sync)
                        
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