# Create multi-style images - API Testing Guide

This guide provides step-by-step instructions for setting up and testing the Marketing Video AI API with multiple image generation styles.

## Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git
## How to Download Required Models

To use this project properly, please download the following model files and place them into the specified folders.

1. **T5 XXL Text Encoder**  
Download from: [T5-v1_1-XXL Encoder](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q8_0.gguf)  
Place it in:  image_generation_style/ComfyUI/models/text_encoders/
2. **FLUX.1 Krea-dev Diffusion Model**  
Download from: [FLUX.1-Krea-dev GGUF](https://huggingface.co/QuantStack/FLUX.1-Krea-dev-GGUF/resolve/main/flux1-krea-dev-Q8_0.gguf)  
Place it in:  image_generation_style/ComfyUI/models/diffusion_models/
3. **FLUX.1 VAE**  
Download from: [FLUX.1-dev VAE](https://huggingface.co/frankjoshua/FLUX.1-dev/resolve/main/ae.safetensors)  
Place it in:  image_generation_style/ComfyUI/models/vae/

4. **Clip L Text Encoder**  
Download from: [Clip L](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors)  
Place it in:  image_generation_style/ComfyUI/models/text_encoders/


## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Hissatsu265/image_generation_style.git
cd image_generation_style
```

### 2. Install Dependencies
Execute the following commands in order:

```bash
# Uninstall conflicting packages
pip uninstall -y tensorflow jax jaxlib
pip uninstall -y torch torchvision torchaudio xformers flash-attn

# Install PyTorch with CUDA support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install xformers and flash-attn
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.6.1 --no-build-isolation

# Install ML libraries
pip install transformers==4.49.0 peft
pip install accelerate

# Install project requirements
pip install -r requirements.txt

# Install system dependencies
apt-get update && apt-get install -y ffmpeg

# Install additional Python packages
pip install ninja psutil packaging

# Fix numpy version
pip uninstall numpy
pip install numpy==1.26.4

# Install web framework dependencies
pip install --upgrade pip
pip install fastapi==0.115.0
pip install uvicorn[standard]==0.32.0  
pip install pydantic==2.11.7
pip install redis==5.2.1
pip install aiofiles==24.1.0
pip install python-multipart==0.0.12
pip install protobuf --upgrade

# Install media processing libraries
pip install moviepy==1.0.3
pip install mediapipe
pip install mutagen
pip install redis
```

## Running the Server

The application requires two servers to be running simultaneously:

### Step 1: Start ComfyUI Server
```bash
cd ComfyUI
python main.py
```
Keep this terminal running. ComfyUI will start on its default port.

### Step 2: Start Main API Server
Open a new terminal and run:
```bash
python run.py
```
The main API server will start on `http://localhost:8000`

## API Usage

### Image Generation Endpoint

**Endpoint:** `POST /api/v1/images_gen/create`

**Request Format:**
```bash
curl -X POST "http://localhost:8000/api/v1/images_gen/create" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["a cat and a dog"],
    "style": "realistic",
    "resolution": "16:9"
  }'
```

### Available Parameters

#### Styles (5 options):
- `realistic` - Photorealistic style
- `anime` - Japanese animation style  
- `cartoon` - Western cartoon style
- `vintage` - Retro/vintage aesthetic
- `minimal` - Minimalist design
- `artistic` - Creative artistic interpretation
- `cyberpunk` - Cyberpunk style

#### Resolutions (3 options):
- `16:9` - Landscape format (1920x1080)
- `9:16` - Portrait format (1080x1920)  
- `1:1` - Square format (1080x1080)

### Job Status Tracking

After submitting an image generation request, you'll receive a job ID. Use this ID to check the status:

```bash
curl "http://localhost:8000/api/v1/jobs/<job_id>/status"
```

Replace `<job_id>` with the actual ID returned from the creation request.

### Response Examples

**Creation Response:**
```json
{
  "job_id": "abc123def456",
  "status": "queued",
  "message": "Job created successfully"
}
```

**Status Check Response:**
```json
{
  "job_id": "abc123def456",
  "status": "completed",
  "progress": 100,
  "result": {
    "images": [
      "http://localhost:8000/static/images/generated_image_1.png"
    ]
  }
}
```

**Status Options:**
- `queued` - Job is waiting in queue
- `processing` - Image is being generated
- `completed` - Generation finished successfully
- `failed` - Generation failed with error

## Troubleshooting

### Common Issues

#### Missing Custom Node Dependencies
If you encounter errors during API testing, it's likely due to missing libraries for custom nodes:

1. Navigate to the custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Install requirements for each node:
   ```bash
   # For each custom node folder, run:
   pip install -r <node_folder>/requirements.txt
   ```

3. Alternatively, install all requirements at once:
   ```bash
   find ComfyUI/custom_nodes/ -name "requirements.txt" -exec pip install -r {} \;
   ```

#### Server Connection Issues
- Ensure both ComfyUI and the main API server are running
- Check that ports are not blocked by firewall
- Verify CUDA installation if using GPU acceleration

#### Memory Issues
- Reduce batch size in requests
- Close unnecessary applications
- Consider using CPU mode if GPU memory is insufficient

## Example Usage Scripts

### Python Example
```python
import requests
import time

# Create image generation job
response = requests.post(
    "http://localhost:8000/api/v1/images_gen/create",
    json={
        "prompts": ["a beautiful sunset over mountains"],
        "style": "artistic",
        "resolution": "16:9"
    }
)

job_data = response.json()
job_id = job_data["job_id"]

# Poll for completion
while True:
    status_response = requests.get(f"http://localhost:8000/api/v1/jobs/{job_id}/status")
    status_data = status_response.json()
    
    print(f"Status: {status_data['status']}")
    
    if status_data["status"] == "completed":
        print(f"Images: {status_data['result']['images']}")
        break
    elif status_data["status"] == "failed":
        print(f"Error: {status_data.get('error', 'Unknown error')}")
        break
    
    time.sleep(2)
```

### Multiple Styles Example
```bash
# Generate the same prompt in different styles
curl -X POST "http://localhost:8000/api/v1/images_gen/create" -H "Content-Type: application/json" -d '{"prompts": ["a majestic dragon"], "style": "realistic", "resolution": "16:9"}'

curl -X POST "http://localhost:8000/api/v1/images_gen/create" -H "Content-Type: application/json" -d '{"prompts": ["a majestic dragon"], "style": "anime", "resolution": "16:9"}'

curl -X POST "http://localhost:8000/api/v1/images_gen/create" -H "Content-Type: application/json" -d '{"prompts": ["a majestic dragon"], "style": "cartoon", "resolution": "16:9"}'
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review ComfyUI logs for detailed error messages
3. Ensure all dependencies are correctly installed
4. Verify GPU drivers and CUDA installation
