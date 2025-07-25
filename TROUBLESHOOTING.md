# Troubleshooting Guide

Common issues and solutions for AI Photo Editor.

## Runtime Issues

### ❌ MPS (Apple Silicon) Unsupported Operation Error

**Error:**
```
NotImplementedError: The operator 'aten::upsample_linear1d.out' is not currently implemented for the MPS device
```

**Solution:**
This is **automatically fixed** in the code. On Apple Silicon (M1/M2/M3), the SAM segmentation model uses CPU to avoid MPS compatibility issues, while the SDXL inpainting model uses MPS for better performance.

**Performance:**
- **SAM Segmentation**: Runs on CPU (~2-3 seconds on M-series chips)
- **SDXL Inpainting**: Runs on MPS GPU (~60-90 seconds)

This hybrid approach provides the best compatibility while still leveraging GPU acceleration where it works.

**If you want to force all models to CPU:**
```bash
# Set in .env
FORCE_CPU=True
```

---

## Installation Issues

### ❌ NumPy Compatibility Error

**Error:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**Solution:**
```bash
# Fix 1: Downgrade NumPy
pip install "numpy<2.0.0" --force-reinstall

# Fix 2: Reinstall all dependencies
pip uninstall -y numpy
pip install -r requirements.txt
```

**Why this happens:**
PyTorch was compiled against NumPy 1.x. NumPy 2.0+ has breaking changes that make it incompatible with older PyTorch builds.

**Prevention:**
The `requirements.txt` now specifies `numpy>=1.24.0,<2.0.0` to prevent this issue.

---

### ❌ Out of Memory (OOM) Error

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Enable CPU offloading in .env
ENABLE_CPU_OFFLOAD=True

# Or reduce image size
IMAGE_SIZE=512  # Instead of 1024
```

**Alternative:**
Reduce inference steps in the UI or code:
```python
result = inpaint_model.inpaint(
    ...,
    num_inference_steps=30  # Instead of 50
)
```

---

### ❌ Model Download Fails

**Error:**
```
OSError: Can't load weights for 'facebook/sam-vit-base'
```

**Solution:**
```bash
# Set HuggingFace cache directory with more space
export HF_HOME=/path/to/large/disk/cache

# Or download manually
python -c "from transformers import SamModel, SamProcessor; SamModel.from_pretrained('facebook/sam-vit-base')"
```

**Check disk space:**
```bash
df -h  # Ensure 20GB+ free space
```

---

### ❌ Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Option 1: Change port in .env
PORT=8000

# Option 2: Kill process using port 5000
lsof -ti:5000 | xargs kill -9

# Option 3: Use different port when running
PORT=8000 python app.py
```

---

## Runtime Issues

### ⚠️ Slow Inference (CPU)

**Symptom:**
Segmentation takes 5+ seconds, inpainting takes 10+ minutes.

**Solution:**
```bash
# Check if GPU is detected
python -c "import torch; print(torch.cuda.is_available())"

# If False on NVIDIA GPU:
# 1. Install CUDA Toolkit
# 2. Reinstall PyTorch with CUDA support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For Apple Silicon (M1/M2/M3):**
```bash
# MPS should be automatically detected
python -c "import torch; print(torch.backends.mps.is_available())"
```

---

### ❌ Poor Segmentation Quality

**Symptom:**
Mask doesn't accurately capture the object.

**Solution:**
1. **Add more points** - Click 3-4 times on different parts of the object
2. **Place points strategically** - Avoid edges, click in the center
3. **Use contrasting backgrounds** - Better separation helps SAM

**Example:**
```python
# Instead of 1-2 points:
points = [[256, 256]]

# Use 3-4 well-distributed points:
points = [
    [200, 200],  # Top-left of object
    [300, 300],  # Center
    [250, 350],  # Bottom
]
```

---

### ❌ Poor Inpainting Results

**Symptom:**
Generated background looks unrealistic or doesn't match prompt.

**Solution:**

1. **Improve prompt specificity:**
```python
# Weak prompt:
prompt = "beach"

# Better prompt:
prompt = "tropical beach at golden hour, turquoise water, palm trees, soft lighting, professional photography"
```

2. **Add negative prompts:**
```python
negative_prompt = "people, crowds, text, watermark, low quality, artifacts, distortion"
```

3. **Adjust guidance scale:**
```python
# Lower (5.0-7.0): More creative, less prompt adherence
# Higher (8.0-12.0): More prompt adherence, less creative
guidance_scale = 9.0
```

4. **Increase inference steps:**
```python
num_inference_steps = 75  # Or 100 for best quality
```

---

## Docker Issues

### ❌ Docker Build Fails

**Error:**
```
ERROR: failed to solve: failed to compute cache key
```

**Solution:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

---

### ❌ GPU Not Detected in Docker

**Solution:**
```bash
# Check nvidia-docker installation
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Update docker-compose.yml (uncomment GPU section):
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## Development Issues

### ❌ Import Errors After Code Changes

**Solution:**
```bash
# Restart Python to reload modules
# Or clear __pycache__
find . -type d -name __pycache__ -exec rm -rf {} +
```

---

### ❌ Tests Failing

**Solution:**
```bash
# Ensure models are downloaded first
python -c "from models import SAMSegmentationModel; SAMSegmentationModel()"

# Run tests with verbose output
python tests/test_pipeline.py -v

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Performance Optimization

### Speed Up Inference

1. **Use GPU** (10x faster than CPU)
2. **Reduce steps**: `num_inference_steps=30`
3. **Smaller images**: `IMAGE_SIZE=512`
4. **Enable xformers** (if available):
   ```bash
   pip install xformers
   ```

### Reduce Memory Usage

1. **Enable CPU offloading**: `ENABLE_CPU_OFFLOAD=True`
2. **Use FP16**: Automatically enabled on GPU
3. **Clear cache between runs**:
   ```python
   from utils import clear_cache
   clear_cache(device)
   ```

---

## Getting Help

If you're still experiencing issues:

1. **Check logs**: Look for detailed error messages
2. **Verify setup**: Run `python setup_check.py`
3. **Search issues**: [GitHub Issues](https://github.com/apathi/ai-photo-editor/issues)
4. **Report bug**: Include:
   - Operating system
   - Python version
   - GPU/CPU specs
   - Full error traceback
   - Steps to reproduce

---

## Quick Diagnostic Commands

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "torch|numpy|transformers|diffusers"

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"

# Check disk space
df -h

# Check memory
free -h  # Linux
vm_stat  # macOS

# Test imports
python -c "from models import SAMSegmentationModel, SDXLInpaintingModel; print('OK')"
```

---

**Last Updated:** 2025-01-13
