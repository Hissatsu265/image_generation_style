# MultiTalk Workflow Description

## Current Features

This workflow creates videos with audio and human portraits, featuring automatic scene transitions using different images.

## Disabled Features

The repository includes code for additional effects that are currently disabled to keep the workflow simple:
- Transition effects
- Product zoom-in effects

These can be manually enabled after basic video generation.

## Audio Segmentation

**Default Behavior**: Audio files longer than 5 seconds are automatically split for scene transitions.

**To Change the 5-second Limit**:

1. **File: `divide_audio.py`** - Edit lines 61, 62, and 82
2. **File: `app2.py`** - Edit lines 800 and 1136

Change `> 5` to your preferred duration (e.g., `> 10` for 10 seconds).

This controls when automatic scene transitions begin based on audio length.

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install Hugging Face CLI

```bash
pip install huggingface_hub[cli]
```

## Model Download

Download all required models using the Hugging Face CLI:

```bash
# Download Wan2.1 I2V model (Image-to-Video)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P

# Download Chinese Wav2Vec2 base model
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base

# Download specific model.safetensors from PR branch
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base

# Download Kokoro voice model
huggingface-cli download hexgrad/Kokoro-82M --local-dir ./weights/Kokoro-82M

# Download MeiGen MultiTalk model
huggingface-cli download MeiGen-AI/MeiGen-MultiTalk --local-dir ./weights/MeiGen-MultiTalk
```

## Model Setup

After downloading, reorganize the model files:

```bash
# Backup original index file
mv weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model.safetensors.index.json_old

# Copy MultiTalk configuration files
cp weights/MeiGen-MultiTalk/diffusion_pytorch_model.safetensors.index.json weights/Wan2.1-I2V-14B-480P/
cp weights/MeiGen-MultiTalk/multitalk.safetensors weights/Wan2.1-I2V-14B-480P/
```

## Dependencies Installation

### 1. Remove Conflicting Packages

```bash
# Remove potentially conflicting packages
pip uninstall -y tensorflow jax jaxlib
pip uninstall -y torch torchvision torchaudio xformers flash-attn
```

### 2. Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.4.1 with CUDA 12.1 support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Additional CUDA Libraries

```bash
# Install xformers for efficient attention
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# Install flash attention for optimized performance
pip install flash-attn==2.6.1 --no-build-isolation

pip install --upgrade pip
pip install fastapi==0.115.0
pip install uvicorn[standard]==0.32.0  
pip install pydantic==2.11.7
pip install redis==5.2.1
pip install aiofiles==24.1.0
pip install python-multipart==0.0.12
```

### 4. Install Core Dependencies

```bash
# Install transformers and related packages
pip install transformers==4.49.0 peft
pip install accelerate

# Install project requirements
pip install -r requirements.txt
```

### 5. System Dependencies

```bash
# Install system packages
apt-get update && apt-get install -y ffmpeg

# Install additional Python packages
pip install ninja psutil packaging
pip install soundfile librosa
pip install misaki[en]
pip install mediapipe
pip install moviepy==1.0.3
```

### 6. Fix NumPy Version

```bash
# Ensure correct NumPy version
pip uninstall numpy
pip install numpy==1.26.4
```

## Usage

### Basic Command

Run the MultiTalk workflow with the following command:

```bash
python app2.py \
    --quant int8 \
    --quant_dir weights/MeiGen-MultiTalk \
    --lora_dir weights/MeiGen-MultiTalk/quant_models/quant_model_int8_FusionX.safetensors \
    --sample_shift 2
```

### Low VRAM Environment

If your system has limited VRAM, add the following parameter:

```bash
python app2.py \
    --quant int8 \
    --quant_dir weights/MeiGen-MultiTalk \
    --lora_dir weights/MeiGen-MultiTalk/quant_models/quant_model_int8_FusionX.safetensors \
    --sample_shift 2 \
    --num_persistent_param_in_dit 0
```

## Command Line Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--quant` | Quantization method (int8, fp16) | int8 |
| `--quant_dir` | Directory containing quantized models | - |
| `--lora_dir` | Path to LoRA model file | - |
| `--sample_shift` | Sampling shift parameter | 2 |
| `--num_persistent_param_in_dit` | Memory optimization for low VRAM | 0 (for low VRAM) |


### Recommended Requirements
- **GPU**: NVIDIA RTX 3090/4090 or better (24GB+ VRAM)
- **RAM**: 32GB system memory
- **Storage**: 200GB

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Add `--num_persistent_param_in_dit 0` to the command
   - Reduce batch size if applicable
   - Close other GPU-intensive applications
