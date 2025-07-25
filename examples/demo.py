"""
Simple demo script for AI Photo Editor.
Shows basic usage of the segmentation and inpainting models.
"""

import os
import sys
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SAMSegmentationModel, SDXLInpaintingModel
from utils import resize_to_square


def main():
    """Run demo with sample image."""
    print("=" * 60)
    print("AI PHOTO EDITOR - DEMO")
    print("=" * 60)

    # Check for sample image
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_images')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        print(f"\nPlease add a sample image to: {sample_dir}")
        print("Then run this script again.")
        return

    # Find first image in samples
    image_files = [f for f in os.listdir(sample_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"\nNo images found in: {sample_dir}")
        print("Please add a sample image and try again.")
        return

    image_path = os.path.join(sample_dir, image_files[0])
    print(f"\nUsing sample image: {image_files[0]}")

    # Load image
    print("\nLoading image...")
    image = Image.open(image_path)
    image = resize_to_square(image, 512)
    print(f"Image resized to: {image.size}")

    # Initialize models
    print("\nInitializing SAM segmentation model...")
    print("(This may take a few minutes on first run)")
    seg_model = SAMSegmentationModel()

    print("\nInitializing SDXL inpainting model...")
    print("(This may take several minutes on first run)")
    inpaint_model = SDXLInpaintingModel()

    # Define points (center and slightly offset)
    points = [
        [256, 256],  # Center
        [300, 300]   # Slightly bottom-right
    ]
    print(f"\nUsing points: {points}")
    print("(In a real app, user would click these on the image)")

    # Generate mask
    print("\nGenerating segmentation mask...")
    mask = seg_model.segment(image, points)

    # Get quality metrics
    iou_scores = seg_model.get_iou_scores(image, points)
    print(f"IoU scores: {iou_scores}")
    print(f"Best IoU: {iou_scores.max():.4f}")

    # Define prompts
    prompts = [
        "beautiful sunset beach with palm trees",
        "snowy mountain landscape with blue sky",
        "modern office interior with large windows"
    ]

    # Generate inpainted versions
    print("\n" + "=" * 60)
    print("GENERATING INPAINTED IMAGES")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Save original
    original_path = os.path.join(output_dir, 'original.png')
    image.save(original_path)
    print(f"\nSaved original: {original_path}")

    # Save mask visualization
    from utils import mask_to_rgba
    mask_rgba = mask_to_rgba(mask, color=(0, 255, 0))
    mask_image = Image.fromarray(mask_rgba)
    mask_path = os.path.join(output_dir, 'mask.png')
    mask_image.save(mask_path)
    print(f"Saved mask: {mask_path}")

    # Generate for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: '{prompt}'")
        print("Generating... (this will take 30-60 seconds)")

        result = inpaint_model.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt="artifacts, low quality, distortion",
            guidance_scale=7.5,
            num_inference_steps=50,
            seed=42 + i  # Different seed for each
        )

        # Save result
        output_path = os.path.join(output_dir, f'result_{i}.png')
        result.save(output_path)
        print(f"Saved: {output_path}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated images:")
    print(f"  • original.png - Original image")
    print(f"  • mask.png - Segmentation mask")
    for i in range(len(prompts)):
        print(f"  • result_{i+1}.png - {prompts[i]}")

    print("\nTo use the web interface, run: python app.py")


if __name__ == '__main__':
    main()
