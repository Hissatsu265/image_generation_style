# Marketing Video AI - API Testing Guide

This guide provides step-by-step instructions for setting up and testing the Marketing Video AI API.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shohanursobuj/marketing-video-ai.git
cd marketing-video-ai
git checkout MinhToan1
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

To start the API server, simply run:

```bash
python run.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### 1. Create Video

Creates a new video from images and audio.

**Endpoint:** `POST /api/v1/videos/create`

**Request Structure:**
```bash
curl -X POST "http://localhost:8000/api/v1/videos/create" \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["/home/toan/marketing-video-ai/assets/justin-bieber-cái.JPG","/home/toan/marketing-video-ai/assets/justin-bieber-cái.JPG"],
    "prompts": ["",""],
    "audio_path": "/home/toan/marketing-video-ai/audio/oskar_1s.wav",
    "resolution": "1280x720"
  }'
```

**Parameters:**
- `image_paths`: Array of image file paths
- `prompts`: Array of prompts (can be empty strings)
- `audio_path`: Path to audio file
- `resolution`: Video resolution (see supported resolutions below)

**Supported Resolutions:**
- `1280x720` (HD)
- `854x480` (SD)
- `720x1280` (Vertical HD)
- `480x854` (Vertical SD)
- And more...

**Response:**
Returns a job ID for tracking the video creation progress.

### 2. Check Job Status

Check the status of a video creation job.

**Endpoint:** `GET /api/v1/jobs/{job_id}/status`

**Request Structure:**
```bash
curl "http://localhost:8000/api/v1/jobs/<job_id>/status"
```

Replace `<job_id>` with the actual job ID returned from the create video endpoint.

### 3. Add Video Effects

Apply transition and dolly effects to an existing video.

**Endpoint:** `POST /api/v1/videos/effects`

**Request Structure:**
```bash
curl -X POST "http://localhost:8000/api/v1/videos/effects" \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/root/marketing-video-ai/55c95f56_clip_0_cut_11.49s.mp4",
    "transition_times": [2.5, 5.0],
    "transition_effects": ["slide", "fade_in"],
    "transition_durations": [0.5, 1.0],
    "dolly_effects": [
      {
        "scene_index": 0,
        "start_time": 1.5,
        "duration": 1.0,
        "zoom_percent": 50,
        "effect_type": "auto_zoom",
        "end_time": 5.0
      },
      {
        "scene_index": 1,
        "start_time": 3.0,
        "duration": 1.5,
        "zoom_percent": 50,
        "effect_type": "manual_zoom",
        "x_coordinate": 100,
        "y_coordinate": 100,
        "end_time": 6.0,
        "end_type": "smooth"
      }
    ]
  }'
```

**Parameters:**

#### Transition Effects
- `transition_times`: Array of times when transitions occur
- `transition_effects`: Array of transition effect names
- `transition_durations`: Array of transition durations

#### Dolly Effects
- `scene_index`: Index of the scene to apply effect
- `start_time`: When the effect starts
- `duration`: Duration of the effect
- `zoom_percent`: Zoom percentage
- `effect_type`: Type of zoom effect (`auto_zoom` or `manual_zoom`)
- `end_time`: When the effect ends
- `x_coordinate`, `y_coordinate`: Target coordinates (for manual zoom)
- `end_type`: End transition type (`instant` or `smooth`)

## Available Transition Effects

The following transition effects are supported:

- `slide` - Slide transition
- `rotate` - Rotation effect
- `circle_mask` - Circular mask transition
- `fade_in` - Fade in effect
- `fade_out` - Fade out effect
- `fadeout_fadein` - Combined fade out and fade in
- `crossfade` - Cross fade between scenes
- `rgb_split` - RGB color split effect
- `flip_horizontal` - Horizontal flip
- `flip_vertical` - Vertical flip
- `push_blur` - Push with blur effect
- `squeeze_horizontal` - Horizontal squeeze
- `wave_distortion` - Wave distortion effect
- `zoom_blur` - Zoom with blur
- `spiral` - Spiral transition
- `pixelate` - Pixelation effect
- `shatter` - Shatter effect
- `kaleidoscope` - Kaleidoscope effect
- `page_turn` - Page turning effect
- `television` - TV static effect
- `film_burn` - Film burn effect
- `matrix_rain` - Matrix rain effect
- `old_film` - Old film effect
- `mosaic_blur` - Mosaic blur effect
- `lens_flare` - Lens flare effect
- `digital_glitch` - Digital glitch effect
- `waterfall` - Waterfall effect
- `honeycomb` - Honeycomb pattern effect
- `none` - No transition effect

## End Types

For dolly effects, the following end types are available:
- `instant` - Immediate transition
- `smooth` - Gradual transition

## Troubleshooting

1. **CUDA Issues**: Ensure you have a CUDA-compatible GPU and the correct CUDA version installed
2. **Memory Issues**: Close other applications if you encounter out-of-memory errors
3. **File Paths**: Use absolute paths for image and audio files
4. **Dependencies**: If you encounter import errors, try reinstalling the specific package

## Notes

- Make sure all file paths in your requests point to existing files
- The server needs to be running before making API calls
- Video processing may take some time depending on the complexity and your hardware
- Monitor the job status endpoint to track progress