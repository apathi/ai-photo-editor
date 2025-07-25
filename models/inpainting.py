"""
SDXL Inpainting pipeline for generative background replacement.
Uses Stable Diffusion XL with classifier-free guidance for high-quality results.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional
from diffusers import AutoPipelineForInpainting
import logging

from utils.device_utils import get_optimal_device, get_torch_dtype, clear_cache
from utils.image_processing import mask_to_grayscale

logger = logging.getLogger(__name__)


class SDXLInpaintingModel:
    """
    Wrapper for Stable Diffusion XL inpainting pipeline.

    Generates new backgrounds based on text prompts while preserving masked regions.
    """

    def __init__(
        self,
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        enable_offload: bool = True
    ):
        """
        Initialize SDXL inpainting pipeline.

        Args:
            model_id: HuggingFace model identifier
            enable_offload: Enable CPU offloading for memory efficiency
        """
        self.device = get_optimal_device()
        self.dtype = get_torch_dtype(self.device)
        self.model_id = model_id

        logger.info(f"Loading SDXL inpainting pipeline: {model_id}")
        logger.info(f"Device: {self.device}, dtype: {self.dtype}")

        # Load pipeline with appropriate dtype
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None
        )

        # Enable CPU offloading for memory efficiency
        if enable_offload:
            self.pipeline.enable_model_cpu_offload()
            logger.info("CPU offloading enabled for memory efficiency")
        else:
            self.pipeline = self.pipeline.to(self.device)

        logger.info("SDXL inpainting pipeline loaded successfully")

    def inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate inpainted image with new background.

        Args:
            image: Original PIL Image
            mask: Binary numpy array (1 = inpaint, 0 = preserve)
            prompt: Text description of desired background
            negative_prompt: Things to avoid in generation
            num_inference_steps: Number of denoising steps (higher = better quality, slower)
            guidance_scale: Classifier-free guidance scale (higher = more prompt adherence)
            seed: Random seed for reproducibility

        Returns:
            PIL Image with inpainted background
        """
        # Convert mask to PIL Image
        mask_image = mask_to_grayscale(mask).resize(image.size)

        # Create generator for reproducibility
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            except RuntimeError:
                # Fallback for devices that don't support generators (e.g., MPS)
                generator = torch.Generator().manual_seed(seed)
                logger.warning(f"Generator on {self.device} not supported, using CPU generator")

        logger.info(f"Inpainting with prompt: '{prompt}'")
        logger.info(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed}")

        # Run inpainting pipeline
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        # Clear cache to free memory
        clear_cache(self.device)

        logger.info("Inpainting complete")

        return result

    def batch_inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompts: list,
        **kwargs
    ) -> list:
        """
        Generate multiple inpainting results with different prompts.

        Args:
            image: Original PIL Image
            mask: Binary numpy array
            prompts: List of text prompts
            **kwargs: Additional arguments for inpaint()

        Returns:
            List of PIL Images
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.inpaint(image, mask, prompt, **kwargs)
            results.append(result)

        return results
