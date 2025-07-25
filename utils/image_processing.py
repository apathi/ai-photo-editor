"""
Image preprocessing and postprocessing utilities.
Handles image transformations, mask generation, and format conversions.
"""

import numpy as np
from PIL import Image
from typing import Tuple


def resize_to_square(image: Image.Image, size: int = 512) -> Image.Image:
    """
    Resize image to square dimensions while preserving aspect ratio.

    Args:
        image: Input PIL Image
        size: Target size (default 512x512)

    Returns:
        PIL Image resized to square
    """
    return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def mask_to_grayscale(mask: np.ndarray) -> Image.Image:
    """
    Convert binary mask to grayscale PIL Image.

    Args:
        mask: Binary numpy array (1 = region to inpaint, 0 = preserve)

    Returns:
        PIL Image in 'L' mode (grayscale)
    """
    mask_uint8 = (mask.astype(np.uint8) * 255)
    return Image.fromarray(mask_uint8, mode='L')


def mask_to_rgba(mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Convert binary mask to RGBA visualization with transparent background.

    Args:
        mask: Binary numpy array
        color: RGB color for masked region (default green)

    Returns:
        RGBA numpy array for visualization
    """
    height, width = mask.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Apply color with semi-transparency to masked region
    rgba[mask == 1] = [*color, 127]  # 50% transparency

    return rgba


def normalize_points(points: list, image_size: Tuple[int, int]) -> list:
    """
    Normalize point coordinates to ensure they're within image bounds.

    Args:
        points: List of [x, y] coordinates
        image_size: Tuple of (width, height)

    Returns:
        Normalized points within bounds
    """
    width, height = image_size
    normalized = []

    for x, y in points:
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))
        normalized.append([x, y])

    return normalized


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string for web transfer.

    Args:
        image: PIL Image

    Returns:
        Base64 encoded string
    """
    import io
    import base64

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"


def base64_to_image(base64_str: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.

    Args:
        base64_str: Base64 encoded image string

    Returns:
        PIL Image
    """
    import io
    import base64

    # Remove data URL prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))
