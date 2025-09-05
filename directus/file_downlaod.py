import requests
import os 
import aiohttp
import tempfile
import aiofiles
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mimetypes
from pathlib import Path
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="Media Downloader API", version="1.0.0")

# Pydantic models for request bodies
class VideoUrlRequest(BaseModel):
    videoUrl: str

class AudioUrlRequest(BaseModel):
    audioUrl: str

class ImageUrlRequest(BaseModel):
    imageUrl: str

# Create temporary directories
# TEMP_DIR = os.path.join(tempfile.gettempdir(), "media_downloader")
TEMP_DIR = "/home/toan/marketing-video-ai/directus/RESOURCE"
os.makedirs(TEMP_DIR, exist_ok=True)

def get_file_extension_from_url(url: str, content_type: Optional[str] = None) -> str:
    """
    Get appropriate file extension based on URL or content type
    """
    if content_type:
        extension = mimetypes.guess_extension(content_type)
        if extension:
            return extension
    
    # Fallback to URL-based detection
    path = Path(url.split('?')[0])  # Remove query parameters
    if path.suffix:
        return path.suffix
    
    # Default extensions based on content type
    if content_type:
        if 'video' in content_type:
            return '.mp4'
        elif 'audio' in content_type:
            return '.mp3'
        elif 'image' in content_type:
            return '.jpg'
    
    return '.bin'  # Generic binary file

async def download_media_from_url(media_url: str, temp_dir: str, media_type: str = "media") -> str:
    """
    Download media (video/audio/image) from URL asynchronously and save to temporary file
    """
    try:
        # Download media using aiohttp for async support
        async with aiohttp.ClientSession() as session:
            print(f"Downloading {media_type} from: {media_url}")
            async with session.get(media_url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Failed to download {media_type}. Status: {response.status}"
                    )
                
                # Get content type and determine file extension
                content_type = response.headers.get('content-type', '')
                file_extension = get_file_extension_from_url(media_url, content_type)
                
                # Create temporary file with appropriate extension
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    dir=temp_dir,
                    suffix=file_extension
                )
                temp_file_path = temp_file.name
                temp_file.close()  # Close the file handle
                
                # Write media content to file asynchronously
                async with aiofiles.open(temp_file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
        
        file_size = os.path.getsize(temp_file_path)
        print(f"{media_type.title()} downloaded successfully to: {temp_file_path} ({file_size} bytes)")
        return temp_file_path
        
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=400, detail=f"Error downloading {media_type}: {str(e)}")
    except Exception as e:
        # Clean up the temp file if it was created
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Unexpected error downloading {media_type}: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Media Downloader API", "endpoints": ["/download-video", "/download-audio", "/download-image"]}

@app.post("/download-video")
async def download_video_from_url(request: VideoUrlRequest):
    """
    Download video from URL and return file path
    """
    try:
        temp_video_file_path = await download_media_from_url(request.videoUrl, TEMP_DIR, "video")
        
        return {
            "success": True,
            "message": "Video downloaded successfully",
            "file_path": temp_video_file_path,
            "file_size": os.path.getsize(temp_video_file_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/download-audio")
async def download_audio_from_url(request: AudioUrlRequest):
    """
    Download audio from URL and return file path
    """
    try:
        temp_audio_file_path = await download_media_from_url(request.audioUrl, TEMP_DIR, "audio")
        
        return {
            "success": True,
            "message": "Audio downloaded successfully",
            "file_path": temp_audio_file_path,
            "file_size": os.path.getsize(temp_audio_file_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/download-image")
async def download_image_from_url(request: ImageUrlRequest):
    """
    Download image from URL and return file path
    """
    try:
        temp_image_file_path = await download_media_from_url(request.imageUrl, TEMP_DIR, "image")
        
        return {
            "success": True,
            "message": "Image downloaded successfully",
            "file_path": temp_image_file_path,
            "file_size": os.path.getsize(temp_image_file_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.delete("/cleanup")
async def cleanup_temp_files():
    """
    Clean up all temporary files in the temp directory
    """
    try:
        deleted_count = 0
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                deleted_count += 1
        
        return {
            "success": True,
            "message": f"Cleaned up {deleted_count} temporary files"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "temp_dir": TEMP_DIR,
        "temp_dir_exists": os.path.exists(TEMP_DIR)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)

