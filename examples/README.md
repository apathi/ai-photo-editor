# Examples

This directory contains sample images and demo scripts for the AI Photo Editor.

## Quick Demo

Run the demo script to see the pipeline in action:

```bash
python examples/demo.py
```

This will:
1. Load a sample image from `sample_images/`
2. Initialize the SAM and SDXL models
3. Generate segmentation masks
4. Create multiple inpainted versions with different prompts
5. Save results to `examples/output/`

## Sample Images

Add your own images to `sample_images/` to test the pipeline:

```bash
cp your_photo.jpg examples/sample_images/
python examples/demo.py
```

Recommended image characteristics:
- **Size**: Any size (will be resized to 512x512)
- **Format**: JPG, PNG
- **Subject**: Clear foreground object on distinguishable background
- **Quality**: High resolution for best results

## Output Examples

After running the demo, check `examples/output/` for:
- `original.png` - Input image (resized)
- `mask.png` - Generated segmentation mask
- `result_1.png`, `result_2.png`, etc. - Inpainted images with different backgrounds

## Custom Usage

You can modify `demo.py` to:
- Change the point coordinates for segmentation
- Use different prompts for background generation
- Adjust inference parameters (steps, guidance scale)
- Process multiple images in batch

Example modifications:

```python
# Custom points (click different areas)
points = [[100, 150], [200, 250], [300, 350]]

# Custom prompt
prompt = "futuristic cyberpunk city at night with neon lights"

# Higher quality settings
result = inpaint_model.inpaint(
    image=image,
    mask=mask,
    prompt=prompt,
    num_inference_steps=100,  # More steps = better quality
    guidance_scale=9.0,        # Higher = more prompt adherence
    seed=12345                 # Reproducible results
)
```

## Tips for Best Results

### Segmentation
- Click on the center of the foreground object
- Add 2-4 points for complex shapes
- Points should be clearly inside the object boundary
- Avoid clicking near edges

### Inpainting Prompts
- Be specific and descriptive
- Mention lighting, style, and atmosphere
- Use negative prompts to avoid unwanted elements
- Example: "professional studio lighting, white background, soft shadows"

### Performance
- First run downloads models (~10GB)
- Subsequent runs are much faster (models cached)
- Use CUDA GPU for 10x faster inference
- Reduce `num_inference_steps` for faster results (trade-off with quality)
