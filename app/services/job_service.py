# import asyncio
# import json
# import uuid
# import os
# from datetime import datetime, timedelta
# from typing import Dict, Any
# from pathlib import Path
# import redis.asyncio as redis
# from app.models.schemas import JobStatus

# class JobService:
#     def __init__(self):
#         self.redis_client = None
#         self.job_queue = asyncio.Queue()
#         self.jobs: Dict[str, Dict[str, Any]] = {}
#         self.processing = False
#         self.cleanup_task = None

#     async def init_redis(self):
#         """Initialize Redis connection"""
#         try:
#             from config import REDIS_URL
#             self.redis_client = redis.from_url(REDIS_URL)
#             await self.redis_client.ping()
#         except Exception as e:
#             print(f"Redis connection failed: {e}")
#             self.redis_client = None

#     async def start_cleanup_task(self):
#         """Bắt đầu task cleanup tự động"""
#         if not self.cleanup_task or self.cleanup_task.done():
#             self.cleanup_task = asyncio.create_task(self.periodic_cleanup())

#     async def periodic_cleanup(self):
#         """Task cleanup chạy định kỳ"""
#         from config import CLEANUP_INTERVAL_MINUTES
        
#         while True:
#             try:
#                 await asyncio.sleep(CLEANUP_INTERVAL_MINUTES * 60)  # Convert to seconds
#                 await self.cleanup_old_jobs()
#                 await self.cleanup_old_videos()
#                 print(f"Cleanup completed at {datetime.now()}")
#             except Exception as e:
#                 print(f"Cleanup error: {e}")

#     async def cleanup_old_jobs(self):
#         """Xóa job cũ khỏi memory và Redis"""
#         from config import JOB_RETENTION_HOURS
        
#         cutoff_time = datetime.now() - timedelta(hours=JOB_RETENTION_HOURS)
#         jobs_to_remove = []
        
#         # Cleanup memory jobs
#         for job_id, job_data in self.jobs.items():
#             try:
#                 created_at = datetime.fromisoformat(job_data["created_at"])
#                 if created_at < cutoff_time:
#                     jobs_to_remove.append(job_id)
#             except (KeyError, ValueError):
#                 # Invalid job data, mark for removal
#                 jobs_to_remove.append(job_id)
        
#         # Remove from memory
#         for job_id in jobs_to_remove:
#             self.jobs.pop(job_id, None)
#             print(f"Removed job from memory: {job_id}")
        
#         # Cleanup Redis jobs
#         if self.redis_client:
#             try:
#                 # Scan for job keys
#                 async for key in self.redis_client.scan_iter(match="job:*"):
#                     job_data = await self.redis_client.get(key)
#                     if job_data:
#                         try:
#                             data = json.loads(job_data)
#                             created_at = datetime.fromisoformat(data["created_at"])
#                             if created_at < cutoff_time:
#                                 await self.redis_client.delete(key)
#                                 print(f"Removed job from Redis: {key}")
#                         except (json.JSONDecodeError, KeyError, ValueError):
#                             # Invalid job data, remove it
#                             await self.redis_client.delete(key)
#                             print(f"Removed invalid job from Redis: {key}")
#             except Exception as e:
#                 print(f"Redis cleanup error: {e}")

#     async def cleanup_old_videos(self):
#         """Xóa video files cũ"""
#         from config import VIDEO_RETENTION_HOURS, OUTPUT_DIR
        
#         cutoff_time = datetime.now() - timedelta(hours=VIDEO_RETENTION_HOURS)
#         output_path = Path(OUTPUT_DIR)
        
#         if not output_path.exists():
#             return
        
#         files_removed = 0
#         for video_file in output_path.glob("*.mp4"):
#             try:
#                 # Lấy thời gian tạo file
#                 file_time = datetime.fromtimestamp(video_file.stat().st_ctime)
#                 if file_time < cutoff_time:
#                     video_file.unlink()
#                     files_removed += 1
#                     print(f"Removed old video: {video_file.name}")
#             except Exception as e:
#                 print(f"Error removing video {video_file.name}: {e}")
        
#         if files_removed > 0:
#             print(f"Cleanup completed: {files_removed} video files removed")

#     async def get_stats(self):
#         """Lấy thống kê jobs và storage"""
#         from config import OUTPUT_DIR
        
#         # Memory jobs stats
#         memory_jobs = len(self.jobs)
#         status_count = {}
#         for job_data in self.jobs.values():
#             status = job_data.get("status", "unknown")
#             status_count[status] = status_count.get(status, 0) + 1
        
#         # Redis jobs stats
#         redis_jobs = 0
#         if self.redis_client:
#             try:
#                 redis_jobs = len([key async for key in self.redis_client.scan_iter(match="job:*")])
#             except:
#                 redis_jobs = -1  # Error
        
#         # Video files stats
#         output_path = Path(OUTPUT_DIR)
#         video_files = len(list(output_path.glob("*.mp4"))) if output_path.exists() else 0
#         total_size = 0
#         if output_path.exists():
#             for video_file in output_path.glob("*.mp4"):
#                 try:
#                     total_size += video_file.stat().st_size
#                 except:
#                     pass
        
#         return {
#             "jobs": {
#                 "memory": memory_jobs,
#                 "redis": redis_jobs,
#                 "status_breakdown": status_count
#             },
#             "videos": {
#                 "count": video_files,
#                 "total_size_mb": round(total_size / 1024 / 1024, 2)
#             },
#             "queue": {
#                 "pending": self.job_queue.qsize()
#             }
#         }

#     async def manual_cleanup(self):
#         """Cleanup thủ công"""
#         await self.cleanup_old_jobs()
#         await self.cleanup_old_videos()
#         return {"message": "Manual cleanup completed"}

#     async def create_job(self, image_paths: list, prompts: list, audio_path: str, resolution: str = "1920x1080") -> str:
#         """Tạo job mới và thêm vào queue"""
#         job_id = str(uuid.uuid4())
        
#         job_data = {
#             "job_id": job_id,
#             "status": JobStatus.PENDING,
#             "image_paths": image_paths,
#             "prompts": prompts,
#             "audio_path": audio_path,
#             "resolution": resolution,  # Thêm resolution
#             "progress": 0,
#             "video_path": None,
#             "error_message": None,
#             "created_at": datetime.now().isoformat(),
#             "completed_at": None
#         }
        
#         # Lưu vào memory và Redis (nếu có)
#         self.jobs[job_id] = job_data
#         if self.redis_client:
#             await self.redis_client.set(
#                 f"job:{job_id}", 
#                 json.dumps(job_data, default=str),
#                 ex=86400  # Expire after 24 hours
#             )
        
#         # Thêm vào queue
#         await self.job_queue.put(job_data)
        
#         # Bắt đầu worker nếu chưa chạy
#         if not self.processing:
#             asyncio.create_task(self.process_jobs())
        
#         # Bắt đầu cleanup task nếu chưa chạy
#         await self.start_cleanup_task()
        
#         return job_id

#     async def get_job_status(self, job_id: str) -> Dict[str, Any]:
#         """Lấy trạng thái job"""
#         # Kiểm tra trong memory trước
#         if job_id in self.jobs:
#             return self.jobs[job_id]
        
#         # Kiểm tra trong Redis
#         if self.redis_client:
#             job_data = await self.redis_client.get(f"job:{job_id}")
#             if job_data:
#                 return json.loads(job_data)
        
#         return None

#     async def update_job_status(self, job_id: str, status: JobStatus, **kwargs):
#         """Cập nhật trạng thái job"""
#         if job_id in self.jobs:
#             self.jobs[job_id]["status"] = status
#             for key, value in kwargs.items():
#                 self.jobs[job_id][key] = value
            
#             if status == JobStatus.COMPLETED or status == JobStatus.FAILED:
#                 self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
#             # Cập nhật Redis
#             if self.redis_client:
#                 await self.redis_client.set(
#                     f"job:{job_id}",
#                     json.dumps(self.jobs[job_id], default=str),
#                     ex=86400
#                 )

#     async def process_jobs(self):
#         """Worker xử lý jobs trong queue"""
#         self.processing = True
        
#         while True:
#             try:
#                 # Lấy job từ queue
#                 job_data = await self.job_queue.get()
#                 job_id = job_data["job_id"]
                
#                 # Cập nhật status thành processing
#                 await self.update_job_status(job_id, JobStatus.PROCESSING, progress=10)
                
#                 # Import video service
#                 from app.services.video_service import VideoService
#                 video_service = VideoService()
                
#                 # Xử lý video
#                 try:
#                     video_path = await video_service.create_video(
#                         image_paths=job_data["image_paths"],
#                         prompts=job_data["prompts"],
#                         audio_path=job_data["audio_path"],
#                         resolution=job_data["resolution"],  # Thêm resolution
#                         job_id=job_id
#                     )
                    
#                     await self.update_job_status(
#                         job_id, 
#                         JobStatus.COMPLETED, 
#                         progress=100,
#                         video_path=video_path
#                     )
                    
#                 except Exception as e:
#                     await self.update_job_status(
#                         job_id,
#                         JobStatus.FAILED,
#                         error_message=str(e)
#                     )
                
#                 # Đánh dấu task hoàn thành
#                 self.job_queue.task_done()
                
#             except Exception as e:
#                 print(f"Error processing job: {e}")
#                 await asyncio.sleep(1)

# job_service = JobService()
import asyncio
import json
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, Any
from pathlib import Path
import redis.asyncio as redis
from app.models.schemas import JobStatus

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

    async def get_stats(self):
        """Lấy thống kê jobs và storage"""
        from config import OUTPUT_DIR
        
        # Memory jobs stats
        memory_jobs = len(self.jobs)
        status_count = {}
        for job_data in self.jobs.values():
            status = job_data.get("status", "unknown")
            status_count[status] = status_count.get(status, 0) + 1
        
        # Redis jobs stats
        redis_jobs = 0
        if self.redis_client:
            try:
                redis_jobs = len([key async for key in self.redis_client.scan_iter(match="job:*")])
            except:
                redis_jobs = -1  # Error
        
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
        
        return {
            "jobs": {
                "memory": memory_jobs,
                "redis": redis_jobs,
                "status_breakdown": status_count
            },
            "videos": {
                "count": video_files,
                "total_size_mb": round(total_size / 1024 / 1024, 2)
            },
            "queue": {
                "pending": self.job_queue.qsize()
            },
            "processing": {
                "current_job": self.current_processing_job,
                "is_processing": not self.video_processing_lock.locked() is False
            }
        }

    async def manual_cleanup(self):
        """Cleanup thủ công"""
        await self.cleanup_old_jobs()
        await self.cleanup_old_videos()
        return {"message": "Manual cleanup completed"}

    async def create_job(self, image_paths: list, prompts: list, audio_path: str, resolution: str = "1920x1080") -> str:
        """Tạo job mới và thêm vào queue - NON-BLOCKING"""
        job_id = str(uuid.uuid4())
        
        job_data = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "image_paths": image_paths,
            "prompts": prompts,
            "audio_path": audio_path,
            "resolution": resolution,
            "progress": 0,
            "video_path": None,
            "error_message": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "queue_position": self.job_queue.qsize() + 1  # Vị trí trong queue
        }
        
        # Lưu vào memory và Redis (nếu có) - NON-BLOCKING
        self.jobs[job_id] = job_data
        if self.redis_client:
            # Sử dụng create_task để không block
            asyncio.create_task(self._save_job_to_redis(job_id, job_data))
        
        # Thêm vào queue - NON-BLOCKING
        await self.job_queue.put(job_data)
        
        # Bắt đầu worker nếu chưa chạy
        await self.start_worker()
        
        # Bắt đầu cleanup task nếu chưa chạy
        asyncio.create_task(self.start_cleanup_task())
        
        return job_id

    async def _save_job_to_redis(self, job_id: str, job_data: dict):
        """Helper method để save job to Redis không block"""
        try:
            await self.redis_client.set(
                f"job:{job_id}", 
                json.dumps(job_data, default=str),
                ex=86400  # Expire after 24 hours
            )
        except Exception as e:
            print(f"Error saving job to Redis: {e}")

    async def start_worker(self):
        """Bắt đầu worker nếu chưa chạy"""
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.process_jobs())
            self.processing = True

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Lấy trạng thái job - NON-BLOCKING"""
        # Kiểm tra trong memory trước
        if job_id in self.jobs:
            job_data = self.jobs[job_id].copy()
            
            # Cập nhật queue position cho job pending
            if job_data["status"] == JobStatus.PENDING:
                position = await self.get_queue_position(job_id)
                job_data["queue_position"] = position
            
            return job_data
        
        # Kiểm tra trong Redis - sử dụng create_task để không block
        if self.redis_client:
            try:
                job_data = await self.redis_client.get(f"job:{job_id}")
                if job_data:
                    return json.loads(job_data)
            except Exception as e:
                print(f"Error getting job from Redis: {e}")
        
        return None

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

    async def update_job_status(self, job_id: str, status: JobStatus, **kwargs):
        """Cập nhật trạng thái job - NON-BLOCKING"""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            for key, value in kwargs.items():
                self.jobs[job_id][key] = value
            
            if status == JobStatus.COMPLETED or status == JobStatus.FAILED:
                self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            # Cập nhật Redis không block
            if self.redis_client:
                asyncio.create_task(self._update_job_in_redis(job_id))

    async def _update_job_in_redis(self, job_id: str):
        """Helper method để update job in Redis không block"""
        try:
            await self.redis_client.set(
                f"job:{job_id}",
                json.dumps(self.jobs[job_id], default=str),
                ex=86400
            )
        except Exception as e:
            print(f"Error updating job in Redis: {e}")

    async def process_jobs(self):
        """Worker xử lý jobs trong queue - chạy trong background"""
        print("Job worker started")
        
        while True:
            try:
                # Lấy job từ queue
                job_data = await self.job_queue.get()
                job_id = job_data["job_id"]
                
                print(f"Processing job: {job_id}")
                
                # Acquire lock để đảm bảo chỉ 1 video được tạo tại 1 thời điểm
                async with self.video_processing_lock:
                    self.current_processing_job = job_id
                    
                    # Cập nhật status thành processing
                    await self.update_job_status(job_id, JobStatus.PROCESSING, progress=10)
                    
                    # Import video service
                    from app.services.video_service import VideoService
                    video_service = VideoService()
                    
                    # Xử lý video - đây là phần blocking
                    try:
                        print(f"Creating video for job: {job_id}")
                        video_path = await video_service.create_video(
                            image_paths=job_data["image_paths"],
                            prompts=job_data["prompts"],
                            audio_path=job_data["audio_path"],
                            resolution=job_data["resolution"],
                            job_id=job_id
                        )
                        
                        await self.update_job_status(
                            job_id, 
                            JobStatus.COMPLETED, 
                            progress=100,
                            video_path=video_path
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
                
                # Đánh dấu task hoàn thành
                self.job_queue.task_done()
                
            except Exception as e:
                print(f"Error in job worker: {e}")
                await asyncio.sleep(1)

    async def get_queue_info(self):
        """Lấy thông tin queue - NON-BLOCKING"""
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

# Global instance
job_service = JobService()