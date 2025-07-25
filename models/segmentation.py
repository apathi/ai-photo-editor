"""
SAM (Segment Anything Model) implementation for object segmentation.
Uses Meta's facebook/sam-vit-base model for interactive segmentation.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
from transformers import SamModel, SamProcessor
import logging

from utils.device_utils import get_optimal_device, get_torch_dtype, clear_cache

logger = logging.getLogger(__name__)


class SAMSegmentationModel:
    """
    Wrapper for Meta's Segment Anything Model (SAM).

    Provides interactive segmentation based on user-provided points.
    """

    def __init__(self, model_id: str = "facebook/sam-vit-base"):
        """
        Initialize SAM model and processor.

        Args:
            model_id: HuggingFace model identifier
        """
        # SAM has compatibility issues with MPS, use CPU on Apple Silicon
        detected_device = get_optimal_device()
        if detected_device == "mps":
            self.device = "cpu"
            logger.info("SAM model using CPU (MPS not fully supported for SAM operations)")
        else:
            self.device = detected_device

        self.dtype = get_torch_dtype(self.device)
        self.model_id = model_id

        logger.info(f"Loading SAM model: {model_id}")
        logger.info(f"Device: {self.device}, dtype: {self.dtype}")

        # Load model and processor
        self.model = SamModel.from_pretrained(model_id).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_id)

        # Convert model weights to appropriate dtype (only for GPU)
        if self.dtype == torch.float16 and self.device != "cpu":
            self.model = self.model.half()

        logger.info("SAM model loaded successfully")

    def segment(
        self,
        image: Image.Image,
        points: List[List[int]]
    ) -> np.ndarray:
        """
        Generate segmentation mask from input image and user points.

        Args:
            image: PIL Image to segment
            points: List of [x, y] coordinates indicating foreground object

        Returns:
            Binary numpy array (1 = background to inpaint, 0 = foreground to preserve)
        """
        if not points or len(points) == 0:
            raise ValueError("At least one point is required for segmentation")

        # Prepare inputs using the processor
        # SAM expects points in nested list format: [[[x1, y1], [x2, y2]]]
        input_points = [points]

        processor_outputs = self.processor(
            image,
            input_points=input_points,
            return_tensors="pt"
        )

        # Move inputs to device and convert dtypes
        inputs = {}
        for key, value in processor_outputs.items():
            if hasattr(value, "to"):
                tensor = value
                # Convert float64 to appropriate dtype
                if hasattr(tensor, "dtype") and tensor.dtype == torch.float64:
                    tensor = tensor.to(self.dtype)
                elif hasattr(tensor, "dtype") and tensor.dtype.is_floating_point:
                    tensor = tensor.to(self.dtype)
                inputs[key] = tensor.to(self.device)
            else:
                inputs[key] = value

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        # Select best mask based on IoU scores
        best_mask_idx = outputs.iou_scores.argmax()
        best_mask = masks[0][0][best_mask_idx]

        # Invert mask: 0 = foreground (preserve), 1 = background (inpaint)
        # This makes it compatible with inpainting pipelines
        inverted_mask = ~best_mask.cpu().numpy()

        # Clear cache to free memory
        clear_cache(self.device)

        logger.info(f"Segmentation complete. Mask shape: {inverted_mask.shape}")

        return inverted_mask

    def get_iou_scores(
        self,
        image: Image.Image,
        points: List[List[int]]
    ) -> np.ndarray:
        """
        Get IoU (Intersection over Union) scores for all predicted masks.

        Useful for evaluating segmentation quality.

        Args:
            image: PIL Image
            points: List of point coordinates

        Returns:
            Array of IoU scores
        """
        input_points = [points]

        processor_outputs = self.processor(
            image,
            input_points=input_points,
            return_tensors="pt"
        )

        inputs = {}
        for key, value in processor_outputs.items():
            if hasattr(value, "to"):
                tensor = value
                if hasattr(tensor, "dtype") and tensor.dtype == torch.float64:
                    tensor = tensor.to(self.dtype)
                elif hasattr(tensor, "dtype") and tensor.dtype.is_floating_point:
                    tensor = tensor.to(self.dtype)
                inputs[key] = tensor.to(self.device)
            else:
                inputs[key] = value

        with torch.no_grad():
            outputs = self.model(**inputs)

        clear_cache(self.device)

        return outputs.iou_scores.cpu().numpy()
