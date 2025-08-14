
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from app.models.schemas import VideoCreateRequest, VideoCreateResponse, JobStatusResponse, JobStatus
from app.services.job_service import job_service
import os
import asyncio
from typing import Optional

router = APIRouter(prefix="/api/v1", tags=["video"])

@router.post("/videos/create", response_model=VideoCreateResponse)
async def create_video(request: VideoCreateRequest):
    """Tạo video từ ảnh, prompt và audio - Non-blocking"""
    
    # Validate input
    if len(request.image_paths) != len(request.prompts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Number of images must match number of prompts"
        )
    
    # Kiểm tra file tồn tại - sử dụng asyncio để không block
    async def check_file_exists(file_path: str, file_type: str):
        try:
            # Sử dụng thread pool để check file không block event loop
            loop = asyncio.get_event_loop()
            exists = await loop.run_in_executor(None, os.path.exists, file_path)
            if not exists:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{file_type} file not found: {file_path}"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error checking {file_type} file: {str(e)}"
            )
    
    # Check files concurrently
    check_tasks = []
    for img_path in request.image_paths:
        check_tasks.append(check_file_exists(img_path, "Image"))
    check_tasks.append(check_file_exists(request.audio_path, "Audio"))
    
    try:
        await asyncio.gather(*check_tasks)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate files: {str(e)}"
        )
    
    try:
        job_id = await job_service.create_job(
            image_paths=request.image_paths,
            prompts=request.prompts,
            audio_path=request.audio_path,
            resolution=request.resolution  
        )
        
        # Lấy thông tin queue để trả về cho user
        queue_info = await job_service.get_queue_info()
        
        return VideoCreateResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message=f"Job created successfully. Position in queue: {queue_info['pending_jobs']}",
            queue_position=queue_info['pending_jobs'],
            estimated_wait_time=queue_info['pending_jobs'] * 5  # Ước tính 5 phút/job
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )

@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Lấy trạng thái job - Non-blocking"""
    
    job_data = await job_service.get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Thêm thông tin queue nếu job đang pending
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