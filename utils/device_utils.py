"""
Device detection and configuration utilities for optimized model execution.
Automatically selects CUDA > MPS > CPU based on availability.

Note: SAM model automatically uses CPU on Apple Silicon (MPS) due to
compatibility issues with certain operations. SDXL inpainting uses MPS.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_optimal_device() -> str:
    """
    Detect and return the optimal device for model execution.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        str: Device identifier ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {device_name}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Metal Performance Shaders (MPS)")
        logger.info("Note: SAM will use CPU, SDXL will use MPS for optimal compatibility")
    else:
        device = "cpu"
        logger.warning("Using CPU - performance will be significantly slower")

    return device


def get_torch_dtype(device: str) -> torch.dtype:
    """
    Get the optimal torch dtype for the given device.

    Args:
        device: Device identifier ('cuda', 'mps', or 'cpu')

    Returns:
        torch.dtype: float16 for CUDA/MPS, float32 for CPU
    """
    if device in ["cuda", "mps"]:
        return torch.float16
    return torch.float32


def clear_cache(device: str) -> None:
    """
    Clear GPU cache to free memory.

    Args:
        device: Device identifier
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def get_device_info() -> dict:
    """
    Get detailed information about the current device.

    Returns:
        dict: Device information including type, memory, and capabilities
    """
    device = get_optimal_device()
    info = {
        "device": device,
        "dtype": str(get_torch_dtype(device))
    }

    if device == "cuda":
        info["device_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated_gb"] = torch.cuda.memory_allocated(0) / 1e9
        info["memory_reserved_gb"] = torch.cuda.memory_reserved(0) / 1e9

    return info
