# 🚀 Quick Start Guide

Get up and running with AI Photo Editor in under 5 minutes!

## Option 1: Local Installation (Recommended for Development)

### Prerequisites
- Python 3.9+
- pip
- 8GB+ RAM

### Steps

1. **Clone and navigate**
   ```bash
   git clone https://github.com/apathi/ai-photo-editor.git
   cd ai-photo-editor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If you encounter a NumPy error:
   ```bash
   pip install "numpy<2.0.0" --force-reinstall
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. **Open browser**
   ```
   http://localhost:5000
   ```

**That's it!** The models will download automatically (~10GB) on first run.

---

## Option 2: Docker (Recommended for Production)

### Prerequisites
- Docker
- Docker Compose

### Steps

1. **Clone repository**
   ```bash
   git clone https://github.com/apathi/ai-photo-editor.git
   cd ai-photo-editor
   ```

2. **Build and run**
   ```bash
   docker-compose up -d
   ```

3. **Check logs**
   ```bash
   docker-compose logs -f
   ```

4. **Open browser**
   ```
   http://localhost:5000
   ```

**GPU Support (Optional)**
```bash
# Uncomment GPU sections in docker-compose.yml
# Requires nvidia-docker
docker-compose up -d
```

---

## First Time Usage

### Web Interface

1. **Upload Image**
   - Click "Choose Image"
   - Select a photo with a clear foreground object
   - JPG or PNG, any size

2. **Select Foreground**
   - Click 2-4 times on the object you want to keep
   - Points should be inside the object boundary

3. **Generate Mask**
   - Click "Generate Mask"
   - Wait ~1-2 seconds
   - Review the green mask overlay

4. **Enter Prompt**
   - Type a description of your desired background
   - Example: "sunset beach with palm trees"
   - Optional: Add negative prompt (things to avoid)

5. **Generate Result**
   - Click "Generate Inpainted Image"
   - Wait 30-60 seconds (first time may be slower)
   - View and download your result!

### Programmatic Usage

```python
from PIL import Image
from models import SAMSegmentationModel, SDXLInpaintingModel
from utils import resize_to_square

# Initialize (one-time setup)
seg_model = SAMSegmentationModel()
inpaint_model = SDXLInpaintingModel()

# Load image
image = Image.open("photo.jpg")
image = resize_to_square(image, 512)

# Generate mask
points = [[250, 300], [280, 320]]
mask = seg_model.segment(image, points)

# Inpaint
result = inpaint_model.inpaint(
    image=image,
    mask=mask,
    prompt="tropical beach at sunset",
    guidance_scale=7.5,
    seed=42
)

result.save("result.png")
```

---

## Testing the Pipeline

Run the demo script with sample image:

```bash
python examples/demo.py
```

This generates multiple variations and saves to `examples/output/`.

Run comprehensive tests:

```bash
python tests/test_pipeline.py
```

This evaluates:
- Segmentation accuracy (IoU scores)
- Inpainting quality (PSNR, SSIM)
- Performance benchmarks
- Edge cases

---

## Troubleshooting

### Models not downloading
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache
python app.py
```

### Out of memory
```bash
# Enable CPU offloading (in .env)
ENABLE_CPU_OFFLOAD=True

# Or reduce image size
IMAGE_SIZE=512
```

### Slow inference
- **Use CUDA GPU** for 10x speedup
- **Reduce steps**: `num_inference_steps=30` (faster, lower quality)
- **Enable offloading**: Trades speed for memory

### Port already in use
```bash
# Change port in .env
PORT=8000
```

---

## Performance Tips

### For Best Quality
- Use `num_inference_steps=100`
- Set `guidance_scale=9.0`
- Use 1024x1024 resolution (if VRAM allows)

### For Speed
- Use `num_inference_steps=30`
- Keep `guidance_scale=7.0`
- Use 512x512 resolution
- Enable CPU offloading

### For Memory Efficiency
- Enable `ENABLE_CPU_OFFLOAD=True`
- Use smaller batch sizes
- Clear cache between runs

---

## Next Steps

1. **Read the full [README.md](README.md)** for detailed documentation
2. **Explore [examples/](examples/)** for usage patterns
3. **Check [tests/](tests/)** for evaluation metrics
4. **Customize prompts** for different backgrounds
5. **Try different images** to test robustness

---

## Quick Reference

### Environment Variables
```bash
PORT=5000                    # Server port
DEBUG=False                  # Debug mode
IMAGE_SIZE=512              # Processing size
ENABLE_CPU_OFFLOAD=True     # Memory optimization
```

### API Endpoints
```bash
GET  /api/health            # Health check
POST /api/segment           # Generate mask
POST /api/inpaint           # Generate inpainted image
```

### Model Parameters

**Segmentation:**
- Input: Image + points
- Output: Binary mask
- Time: ~0.5s (GPU)

**Inpainting:**
- Steps: 20-100 (default: 50)
- Guidance: 1.0-15.0 (default: 7.5)
- Time: ~30s for 50 steps (GPU)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/apathi/ai-photo-editor/issues)
- **Documentation**: [README.md](README.md)
- **Examples**: [examples/README.md](examples/README.md)

---

**Ready to create amazing AI-edited photos!** 🎨✨
