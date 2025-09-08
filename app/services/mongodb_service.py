from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional

class MongoDBService:
    def __init__(self, uri: str, db_name: str):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        
    async def connect(self):
        """Kết nối tới MongoDB"""
        try:
            loop = asyncio.get_event_loop()
            self.client = await loop.run_in_executor(
                self.executor, 
                lambda: MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            )
            self.db = self.client[self.db_name]
            
            # Test connection
            await loop.run_in_executor(
                self.executor,
                lambda: self.client.admin.command('ping')
            )
            print(f"Connected to MongoDB: {self.db_name}")
            return True
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Đóng kết nối MongoDB"""
        if self.client:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.client.close)
            self.executor.shutdown(wait=True)
    
    async def insert_one(self, collection_name: str, document: dict) -> str:
        """Insert một document"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.db[collection_name].insert_one(document)
            )
            return str(result.inserted_id)
        except Exception as e:
            print(f"MongoDB insert error: {e}")
            raise
    
    async def find_one(self, collection_name: str, filter_dict: dict) -> Optional[dict]:
        """Tìm một document"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.db[collection_name].find_one(filter_dict)
            )
            if result:
                # Convert ObjectId to string
                result['_id'] = str(result['_id'])
            return result
        except Exception as e:
            print(f"MongoDB find error: {e}")
            return None
    
    async def update_one(self, collection_name: str, filter_dict: dict, update_dict: dict) -> bool:
        """Update một document"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.db[collection_name].update_one(filter_dict, {"$set": update_dict})
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"MongoDB update error: {e}")
            return False
    
    async def delete_one(self, collection_name: str, filter_dict: dict) -> bool:
        """Xóa một document"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.db[collection_name].delete_one(filter_dict)
            )
            return result.deleted_count > 0
        except Exception as e:
            print(f"MongoDB delete error: {e}")
            return False
    
    async def find_many(self, collection_name: str, filter_dict: dict = {}, 
                       limit: int = 50, skip: int = 0, sort_field: str = None, 
                       sort_desc: bool = True) -> List[dict]:
        """Tìm nhiều documents"""
        try:
            loop = asyncio.get_event_loop()
            
            def query():
                cursor = self.db[collection_name].find(filter_dict)
                if sort_field:
                    cursor = cursor.sort(sort_field, -1 if sort_desc else 1)
                return list(cursor.skip(skip).limit(limit))
            
            results = await loop.run_in_executor(self.executor, query)
            
            # Convert ObjectIds to strings
            for result in results:
                result['_id'] = str(result['_id'])
            
            return results
        except Exception as e:
            print(f"MongoDB find many error: {e}")
            return []
    
    async def count_documents(self, collection_name: str, filter_dict: dict = {}) -> int:
        """Đếm số documents"""
        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(
                self.executor,
                lambda: self.db[collection_name].count_documents(filter_dict)
            )
            return count
        except Exception as e:
            print(f"MongoDB count error: {e}")
            return 0
    
    async def delete_many(self, collection_name: str, filter_dict: dict) -> int:
        """Xóa nhiều documents"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: self.db[collection_name].delete_many(filter_dict)
            )
            return result.deleted_count
        except Exception as e:
            print(f"MongoDB delete many error: {e}")
            return 0