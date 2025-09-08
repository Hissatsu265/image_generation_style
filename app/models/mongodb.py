from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from datetime import datetime
import asyncio

class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None
        self.jobs_col = None
        self.effect_jobs_col = None
        
    async def connect(self):
        """Kết nối async MongoDB"""
        try:
            # Sử dụng connection string từ config
            from config import MONGODB_URI, MONGODB_DATABASE
            
            self.client = AsyncIOMotorClient(MONGODB_URI)
            self.db = self.client[MONGODB_DATABASE]
            
            # Tạo collections
            self.jobs_col = self.db["jobs"]
            self.effect_jobs_col = self.db["effect_jobs"]
            
            # Test connection
            await self.client.admin.command('ping')
            print("MongoDB connected successfully")
            
            # Tạo indexes
            await self.create_indexes()
            
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            raise
    
    async def create_indexes(self):
        """Tạo indexes cho performance"""
        try:
            # Jobs collection indexes
            await self.jobs_col.create_index("job_id", unique=True)
            await self.jobs_col.create_index("status")
            await self.jobs_col.create_index("created_at")
            
            # Effect jobs collection indexes  
            await self.effect_jobs_col.create_index("job_id", unique=True)
            await self.effect_jobs_col.create_index("status")
            await self.effect_jobs_col.create_index("created_at")
            await self.effect_jobs_col.create_index("worker_id")
            
            print("MongoDB indexes created")
        except Exception as e:
            print(f"Error creating indexes: {e}")
    
    async def close(self):
        """Đóng connection"""
        if self.client:
            self.client.close()

# Global instance
mongodb = MongoDB()