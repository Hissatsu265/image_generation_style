from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.services.job_service import job_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await job_service.init_redis()
    yield
    # Shutdown
    if job_service.redis_client:
        await job_service.redis_client.close()

app = FastAPI(
    title="Video Generator API",
    description="API để tạo video từ ảnh, prompt và audio",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Video Generator API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}