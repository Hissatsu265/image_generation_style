from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
# from app.models.schemas import VideoCreateRequest, VideoCreateResponse, JobStatusResponse, JobStatus
from app.services.job_service import job_service
import os
import asyncio
from typing import Optional
from app.models.schemas import (
    VideoCreateRequest, VideoCreateResponse, JobStatusResponse, JobStatus,
    VideoEffectRequest, VideoEffectResponse, EffectJobStatusResponse
)
from directus.function_downloadfile import download_image, download_audio

router = APIRouter(prefix="/api/v1", tags=["video"])

# @router.post("/videos/create", response_model=VideoCreateResponse)
# async def create_video(request: VideoCreateRequest):
    
#     if len(request.image_paths) != len(request.prompts):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Number of images must match number of prompts"
#         )
    
#     # Kiểm tra file tồn tại - sử dụng asyncio để không block
#     async def check_file_exists(file_path: str, file_type: str):
#         try:
#             # Sử dụng thread pool để check file không block event loop
#             loop = asyncio.get_event_loop()
#             exists = await loop.run_in_executor(None, os.path.exists, file_path)
#             if not exists:
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"{file_type} file not found: {file_path}"
#                 )
#         except HTTPException:
#             raise
#         except Exception as e:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Error checking {file_type} file: {str(e)}"
#             )
    
#     # Check files concurrently
#     check_tasks = []
#     for img_path in request.image_paths:
#         check_tasks.append(check_file_exists(img_path, "Image"))
#     check_tasks.append(check_file_exists(request.audio_path, "Audio"))
    
#     try:
#         await asyncio.gather(*check_tasks)
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to validate files: {str(e)}"
#         )
    
#     try:
#         job_id = await job_service.create_job(
#             image_paths=request.image_paths,
#             prompts=request.prompts,
#             audio_path=request.audio_path,
#             resolution=request.resolution,
#             background=request.background  
#         )
        
#         # Lấy thông tin queue để trả về cho user
#         queue_info = await job_service.get_queue_info()
        
#         return VideoCreateResponse(
#             job_id=job_id,
#             status=JobStatus.PENDING,
#             message=f"Job created successfully. Position in queue: {queue_info['pending_jobs']}",
#             queue_position=queue_info['pending_jobs'],
#             estimated_wait_time=queue_info['pending_jobs'] * 5  
#         )
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to create job: {str(e)}"
#         )
# ===============================================================
@router.post("/images_gen/create", response_model=VideoCreateResponse)
async def create_video(request: VideoCreateRequest):
    # if len(request.image_paths) != len(request.prompts):
    request.image_paths=["hehe"]

    # try:
    #     request.image_paths = [
    #         download_image(img_url) for img_url in request.image_paths
    #     ]
    #     # request.audio_path = download_audio(request.audio_path)
    #     print("Downloaded images:", request.image_paths)
    #     # print("s====================s")
    #     # print("Downloaded audio:", request.audio_path)
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail=f"Failed to download files: {str(e)}"
    #     )


    try:
        job_id = await job_service.create_job(
            image_paths=request.image_paths,
            prompts=request.prompts,
            style=request.style,
            resolution=request.resolution  
        )

        queue_info = await job_service.get_queue_info()

        return VideoCreateResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message=f"Job created successfully. Position in queue: {queue_info['pending_jobs']}",
            queue_position=queue_info['pending_jobs'],
            estimated_wait_time=queue_info['pending_jobs'] * 5
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )
# =============================================================================
@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    
    job_data = await job_service.get_job_status(job_id)
    print("Job data:", job_data)
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job_data["status"] == JobStatus.PENDING:
        queue_info = await job_service.get_queue_info()
        job_data["queue_position"] = job_data.get("queue_position", 0)
        job_data["estimated_wait_time"] = job_data["queue_position"] * 5  # 5 phút/job
        job_data["is_processing"] = queue_info["is_processing"]
        job_data["current_processing_job"] = queue_info["current_processing"]
    
    return JobStatusResponse(**job_data)

@router.get("/jobs/{job_id}/download")
async def download_video(job_id: str):
    """Download video đã tạo - Non-blocking"""
    
    job_data = await job_service.get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job_data["status"] != JobStatus.COMPLETED:
        # Trả về thông tin status thay vì chỉ error
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "message": "Video is not ready yet",
                "status": job_data["status"],
                "progress": job_data.get("progress", 0),
                "queue_position": job_data.get("queue_position", 0) if job_data["status"] == JobStatus.PENDING else None
            }
        )
    
    video_path = job_data["video_path"]
    
    # Check file exists không block
    loop = asyncio.get_event_loop()
    file_exists = await loop.run_in_executor(None, os.path.exists, video_path)
    
    if not file_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file not found on server"
        )
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"video_{job_id}.mp4",
        headers={"Content-Disposition": f"attachment; filename=video_{job_id}.mp4"}
    )

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Hủy job nếu đang pending"""
    
    cancelled = await job_service.cancel_job(job_id)
    
    if not cancelled:
        job_data = await job_service.get_job_status(job_id)
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        elif job_data["status"] == JobStatus.PROCESSING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot cancel job that is currently processing"
            )
        elif job_data["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job that is already {job_data['status']}"
            )
    
    return {"message": "Job cancelled successfully", "job_id": job_id}

@router.get("/queue/info")
async def get_queue_info():
    """Lấy thông tin queue hiện tại"""
    queue_info = await job_service.get_queue_info()
    return {
        "pending_jobs": queue_info["pending_jobs"],
        "is_processing": queue_info["is_processing"],
        "current_processing_job": queue_info["current_processing"],
        "worker_status": "running" if queue_info["worker_running"] else "stopped",
        "estimated_total_wait_time": queue_info["pending_jobs"] * 5  # phút
    }

@router.get("/jobs")
async def list_jobs(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List jobs với filter và pagination"""
    
    try:
        # Lấy stats từ job service
        stats = await job_service.get_stats()
        
        # Lọc jobs theo status nếu có
        all_jobs = []
        for job_id, job_data in job_service.jobs.items():
            if status_filter and job_data.get("status") != status_filter:
                continue
            
            job_summary = {
                "job_id": job_id,
                "status": job_data.get("status"),
                "created_at": job_data.get("created_at"),
                "completed_at": job_data.get("completed_at"),
                "progress": job_data.get("progress", 0),
                "resolution": job_data.get("resolution"),
                "error_message": job_data.get("error_message")
            }
            all_jobs.append(job_summary)
        
        # Sort by created_at desc
        all_jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Pagination
        paginated_jobs = all_jobs[offset:offset + limit]
        
        return {
            "jobs": paginated_jobs,
            "total": len(all_jobs),
            "limit": limit,
            "offset": offset,
            "has_more": len(all_jobs) > offset + limit,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )

# === HEALTH CHECK ===

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        queue_info = await job_service.get_queue_info()
        stats = await job_service.get_stats()
        
        # Check Redis connection
        redis_status = "not_configured"
        if job_service.redis_client:
            try:
                await job_service.redis_client.ping()
                redis_status = "healthy"
            except:
                redis_status = "error"
        
        return {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "queue": queue_info,
            "stats": stats,
            "redis": redis_status,
            "worker_running": queue_info["worker_running"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
        )

# === ADMIN/DEBUG ENDPOINTS ===

@router.get("/admin/stats")
async def get_system_stats():
    """Lấy thống kê hệ thống (jobs, videos, storage)"""
    return await job_service.get_stats()

@router.post("/admin/cleanup")
async def manual_cleanup(background_tasks: BackgroundTasks):
    """Cleanup thủ công jobs và videos cũ - Non-blocking"""
    
    # Chạy cleanup trong background để không block response
    background_tasks.add_task(job_service.manual_cleanup)
    
    return {
        "message": "Cleanup started in background",
        "timestamp": asyncio.get_event_loop().time()
    }

@router.get("/admin/redis")
async def test_redis():
    """Test kết nối Redis - Non-blocking"""
    if job_service.redis_client:
        try:
            # Test ping với timeout
            await asyncio.wait_for(job_service.redis_client.ping(), timeout=5.0)
            return {"redis": "connected", "status": "healthy"}
        except asyncio.TimeoutError:
            return {"redis": "timeout", "status": "slow_response"}
        except Exception as e:
            return {"redis": "error", "message": str(e)}
    return {"redis": "not_configured", "status": "using_memory_only"}

@router.post("/admin/worker/restart")
async def restart_worker():
    """Restart job worker"""
    try:
        # Cancel current worker task if exists
        if job_service.worker_task and not job_service.worker_task.done():
            job_service.worker_task.cancel()
            try:
                await job_service.worker_task
            except asyncio.CancelledError:
                pass
        
        # Start new worker
        await job_service.start_worker()
        
        return {
            "message": "Worker restarted successfully",
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart worker: {str(e)}"
        )

@router.get("/admin/queue/clear")
async def clear_queue():
    """Clear pending jobs from queue (DANGER!)"""
    try:
        cleared_count = 0
        while not job_service.job_queue.empty():
            try:
                await asyncio.wait_for(job_service.job_queue.get(), timeout=0.1)
                job_service.job_queue.task_done()
                cleared_count += 1
            except asyncio.TimeoutError:
                break
        
        return {
            "message": f"Queue cleared: {cleared_count} jobs removed",
            "cleared_jobs": cleared_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear queue: {str(e)}"
        )
# ========================
# ========================
# ========================




# === NEW VIDEO EFFECTS ENDPOINTS ===

@router.post("/videos/effects", response_model=VideoEffectResponse)
async def create_video_effects(request: VideoEffectRequest):
    """Tạo video với hiệu ứng - Multiple workers processing"""
    
    # Validate input
    if len(request.transition_times) != len(request.transition_effects):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Number of transition_times must match transition_effects"
        )
    
    if len(request.transition_effects) != len(request.transition_durations):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Number of transition_effects must match transition_durations"
        )
    
    # Kiểm tra video file tồn tại
    try:
        loop = asyncio.get_event_loop()
        exists = await loop.run_in_executor(None, os.path.exists, request.video_path)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video file not found: {request.video_path}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking video file: {str(e)}"
        )
    
    # Validate dolly effects timing
    for dolly in request.dolly_effects or []:
        if dolly.start_time < 0 or dolly.duration <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dolly effect timing: start_time={dolly.start_time}, duration={dolly.duration}"
            )
    
    try:
        # Tạo effect job
        job_id = await job_service.create_effect_job(
            video_path=request.video_path,
            transition_times=request.transition_times,
            transition_effects=[effect.value for effect in request.transition_effects],  # Convert enum to string
            transition_durations=request.transition_durations,
            dolly_effects=[dolly.dict() for dolly in request.dolly_effects] if request.dolly_effects else []
        )
        
        # Lấy thông tin workers
        workers_info = await job_service.get_effect_workers_info()
        
        return VideoEffectResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message=f"Effect job created successfully. Queue position: {workers_info['pending_jobs']}",
            queue_position=workers_info['pending_jobs'],
            estimated_wait_time=max(1, workers_info['pending_jobs'] // workers_info['total_workers']) * 3,  # Ước tính 3 phút/batch
            available_workers=workers_info['available_workers'],
            total_workers=workers_info['total_workers']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create effect job: {str(e)}"
        )

@router.get("/effects/{job_id}/status", response_model=EffectJobStatusResponse)
async def get_effect_job_status(job_id: str):
    """Lấy trạng thái effect job"""
    
    job_data = await job_service.get_effect_job_status(job_id)
    
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Effect job not found"
        )
    
    # Thêm thông tin workers nếu job đang pending
    if job_data["status"] == JobStatus.PENDING:
        workers_info = await job_service.get_effect_workers_info()
        job_data["queue_position"] = job_data.get("queue_position", 0)
        job_data["estimated_wait_time"] = max(1, job_data["queue_position"] // workers_info['total_workers']) * 3
    
    return EffectJobStatusResponse(**job_data)

@router.get("/effects/{job_id}/download")
async def download_effect_video(job_id: str):
    """Download video đã xử lý hiệu ứng"""
    
    job_data = await job_service.get_effect_job_status(job_id)
    
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Effect job not found"
        )
    
    if job_data["status"] != JobStatus.COMPLETED:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "message": "Effect video is not ready yet",
                "status": job_data["status"],
                "progress": job_data.get("progress", 0),
                "queue_position": job_data.get("queue_position", 0) if job_data["status"] == JobStatus.PENDING else None,
                "worker_id": job_data.get("worker_id")
            }
        )
    
    output_video_path = job_data["output_video_path"]
    
    if not output_video_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Output video path not found"
        )
    
    # Check file exists
    loop = asyncio.get_event_loop()
    file_exists = await loop.run_in_executor(None, os.path.exists, output_video_path)
    
    if not file_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Effect video file not found on server"
        )
    
    return FileResponse(
        path=output_video_path,
        media_type="video/mp4",
        filename=f"effect_video_{job_id}.mp4",
        headers={"Content-Disposition": f"attachment; filename=effect_video_{job_id}.mp4"}
    )

@router.delete("/effects/{job_id}")
async def cancel_effect_job(job_id: str):
    """Hủy effect job nếu đang pending"""
    
    cancelled = await job_service.cancel_effect_job(job_id)
    
    if not cancelled:
        job_data = await job_service.get_effect_job_status(job_id)
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Effect job not found"
            )
        elif job_data["status"] == JobStatus.PROCESSING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot cancel effect job that is currently processing"
            )
        elif job_data["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel effect job that is already {job_data['status']}"
            )
    
    return {"message": "Effect job cancelled successfully", "job_id": job_id}

@router.get("/effects/workers/info")
async def get_effect_workers_info():
    """Lấy thông tin về effect workers"""
    
    workers_info = await job_service.get_effect_workers_info()
    return {
        "total_workers": workers_info["total_workers"],
        "available_workers": workers_info["available_workers"], 
        "busy_workers": workers_info["busy_workers"],
        "pending_jobs": workers_info["pending_jobs"],
        "workers_status": workers_info["workers_status"],
        "estimated_processing_time": f"{max(1, workers_info['pending_jobs'] // workers_info['total_workers']) * 3} minutes"
    }

# === EXISTING ENDPOINTS (giữ nguyên) ===
# delete jobs, queue info, list jobs, health check, admin endpoints...

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Hủy job nếu đang pending"""
    
    cancelled = await job_service.cancel_job(job_id)
    
    if not cancelled:
        job_data = await job_service.get_job_status(job_id)
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        elif job_data["status"] == JobStatus.PROCESSING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot cancel job that is currently processing"
            )
        elif job_data["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job that is already {job_data['status']}"
            )
    
    return {"message": "Job cancelled successfully", "job_id": job_id}

@router.get("/queue/info")
async def get_queue_info():
    """Lấy thông tin queue hiện tại"""
    queue_info = await job_service.get_queue_info()
    return {
        "pending_jobs": queue_info["pending_jobs"],
        "is_processing": queue_info["is_processing"],
        "current_processing_job": queue_info["current_processing"],
        "worker_status": "running" if queue_info["worker_running"] else "stopped",
        "estimated_total_wait_time": queue_info["pending_jobs"] * 5  # phút
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        queue_info = await job_service.get_queue_info()
        stats = await job_service.get_stats()
        
        # Check Redis connection
        redis_status = "not_configured"
        if job_service.redis_client:
            try:
                await job_service.redis_client.ping()
                redis_status = "healthy"
            except:
                redis_status = "error"
        
        return {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "queue": queue_info,
            "stats": stats,
            "redis": redis_status,
            "worker_running": queue_info["worker_running"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
        )

@router.get("/admin/stats")
async def get_system_stats():
    """Lấy thống kê hệ thống (jobs, videos, storage)"""
    return await job_service.get_stats()