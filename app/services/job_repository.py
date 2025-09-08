from datetime import datetime
from typing import Dict, Any, List, Optional
from app.models.mongodb import  mongodb
from app.models.schemas import JobStatus
import pymongo

class JobRepository:
    def __init__(self):
        self.mongodb = mongodb
    
    # ===== VIDEO CREATION JOBS =====
    
    async def insert_job(self, job_data: Dict[str, Any]) -> str:
        """Thêm job mới vào MongoDB"""
        try:
            result = await self.mongodb.jobs_col.insert_one(job_data)
            return job_data["job_id"]
        except Exception as e:
            raise Exception(f"Error inserting job: {e}")
    
    async def find_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Tìm job theo ID"""
        try:
            job = await self.mongodb.jobs_col.find_one({"job_id": job_id})
            return job
        except Exception as e:
            print(f"Error finding job {job_id}: {e}")
            return None
    
    async def update_job(self, job_id: str, update_data: Dict[str, Any]) -> bool:
        """Cập nhật job"""
        try:
            result = await self.mongodb.jobs_col.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating job {job_id}: {e}")
            return False
    
    async def find_jobs_by_status(self, status: JobStatus) -> List[Dict[str, Any]]:
        """Tìm jobs theo status"""
        try:
            cursor = self.mongodb.jobs_col.find({"status": status})
            return await cursor.to_list(length=None)
        except Exception as e:
            print(f"Error finding jobs by status {status}: {e}")
            return []
    
    async def count_jobs_by_status(self, status: JobStatus) -> int:
        """Đếm số jobs theo status"""
        try:
            return await self.mongodb.jobs_col.count_documents({"status": status})
        except Exception as e:
            print(f"Error counting jobs by status {status}: {e}")
            return 0
    
    async def list_jobs(self, 
                       status_filter: Optional[str] = None,
                       limit: int = 50, 
                       offset: int = 0) -> Dict[str, Any]:
        """List jobs với pagination"""
        try:
            # Build query
            query = {}
            if status_filter:
                query["status"] = status_filter
            
            # Get total count
            total = await self.mongodb.jobs_col.count_documents(query)
            
            # Get jobs with pagination
            cursor = self.mongodb.jobs_col.find(query).sort("created_at", pymongo.DESCENDING).skip(offset).limit(limit)
            jobs = await cursor.to_list(length=limit)
            
            return {
                "jobs": jobs,
                "total": total,
                "has_more": total > offset + limit
            }
        except Exception as e:
            print(f"Error listing jobs: {e}")
            return {"jobs": [], "total": 0, "has_more": False}
    
    async def delete_job(self, job_id: str) -> bool:
        """Xóa job"""
        try:
            result = await self.mongodb.jobs_col.delete_one({"job_id": job_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting job {job_id}: {e}")
            return False
    
    async def cleanup_old_jobs(self, cutoff_time: datetime) -> int:
        """Xóa jobs cũ"""
        try:
            result = await self.mongodb.jobs_col.delete_many({
                "created_at": {"$lt": cutoff_time.isoformat()}
            })
            return result.deleted_count
        except Exception as e:
            print(f"Error cleaning up old jobs: {e}")
            return 0
    
    # ===== EFFECT JOBS =====
    
    async def insert_effect_job(self, job_data: Dict[str, Any]) -> str:
        """Thêm effect job mới"""
        try:
            result = await self.mongodb.effect_jobs_col.insert_one(job_data)
            return job_data["job_id"]
        except Exception as e:
            raise Exception(f"Error inserting effect job: {e}")
    
    async def find_effect_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Tìm effect job theo ID"""
        try:
            job = await self.mongodb.effect_jobs_col.find_one({"job_id": job_id})
            return job
        except Exception as e:
            print(f"Error finding effect job {job_id}: {e}")
            return None
    
    async def update_effect_job(self, job_id: str, update_data: Dict[str, Any]) -> bool:
        """Cập nhật effect job"""
        try:
            result = await self.mongodb.effect_jobs_col.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating effect job {job_id}: {e}")
            return False
    
    async def find_effect_jobs_by_status(self, status: JobStatus) -> List[Dict[str, Any]]:
        """Tìm effect jobs theo status"""
        try:
            cursor = self.mongodb.effect_jobs_col.find({"status": status})
            return await cursor.to_list(length=None)
        except Exception as e:
            print(f"Error finding effect jobs by status {status}: {e}")
            return []
    
    async def count_effect_jobs_by_status(self, status: JobStatus) -> int:
        """Đếm effect jobs theo status"""
        try:
            return await self.mongodb.effect_jobs_col.count_documents({"status": status})
        except Exception as e:
            print(f"Error counting effect jobs by status {status}: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê jobs từ MongoDB"""
        try:
            # Video creation jobs stats
            video_stats = {}
            for status in JobStatus:
                video_stats[status] = await self.count_jobs_by_status(status)
            
            # Effect jobs stats  
            effect_stats = {}
            for status in JobStatus:
                effect_stats[status] = await self.count_effect_jobs_by_status(status)
            
            return {
                "video_creation_jobs": {
                    "status_breakdown": video_stats,
                    "total": sum(video_stats.values())
                },
                "effect_jobs": {
                    "status_breakdown": effect_stats,
                    "total": sum(effect_stats.values())
                }
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                "video_creation_jobs": {"status_breakdown": {}, "total": 0},
                "effect_jobs": {"status_breakdown": {}, "total": 0}
            }

# Global instance
job_repository = JobRepository()