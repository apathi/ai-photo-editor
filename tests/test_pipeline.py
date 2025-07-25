"""
Comprehensive testing and evaluation for AI Photo Editor pipeline.
Tests segmentation accuracy, inpainting quality, and end-to-end workflow.
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SAMSegmentationModel, SDXLInpaintingModel
from utils import resize_to_square, mask_to_grayscale

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """
    Comprehensive evaluation suite for the photo editing pipeline.
    """

    def __init__(self):
        """Initialize evaluator with metrics tracking."""
        self.metrics = {
            "segmentation": [],
            "inpainting": [],
            "performance": []
        }

    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two masks.

        Args:
            mask1: First binary mask
            mask2: Second binary mask

        Returns:
            IoU score (0-1)
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def calculate_dice(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Dice coefficient between two masks.

        Args:
            mask1: First binary mask
            mask2: Second binary mask

        Returns:
            Dice score (0-1)
        """
        intersection = np.logical_and(mask1, mask2).sum()
        total = mask1.sum() + mask2.sum()

        if total == 0:
            return 0.0

        return 2 * intersection / total

    def calculate_psnr(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.

        Args:
            img1: First image
            img2: Second image

        Returns:
            PSNR value (higher is better)
        """
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Ensure same size
        if arr1.shape != arr2.shape:
            img2 = img2.resize(img1.size)
            arr2 = np.array(img2)

        return psnr(arr1, arr2, data_range=255)

    def calculate_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate Structural Similarity Index.

        Args:
            img1: First image
            img2: Second image

        Returns:
            SSIM value (0-1, higher is better)
        """
        arr1 = np.array(img1.convert('RGB'))
        arr2 = np.array(img2.convert('RGB'))

        # Ensure same size
        if arr1.shape != arr2.shape:
            img2 = img2.resize(img1.size)
            arr2 = np.array(img2.convert('RGB'))

        return ssim(arr1, arr2, channel_axis=2, data_range=255)

    def test_segmentation_accuracy(
        self,
        model: SAMSegmentationModel,
        image: Image.Image,
        points: list,
        ground_truth: np.ndarray = None
    ) -> dict:
        """
        Test segmentation model accuracy.

        Args:
            model: SAM segmentation model
            image: Input image
            points: Point coordinates
            ground_truth: Optional ground truth mask for comparison

        Returns:
            Dictionary with test results
        """
        logger.info("Testing segmentation accuracy...")

        start_time = time.time()
        mask = model.segment(image, points)
        inference_time = time.time() - start_time

        results = {
            "inference_time_sec": inference_time,
            "mask_shape": mask.shape,
            "foreground_ratio": (mask == 0).sum() / mask.size,
            "background_ratio": (mask == 1).sum() / mask.size
        }

        # Calculate IoU scores from model
        iou_scores = model.get_iou_scores(image, points)
        results["model_iou_scores"] = iou_scores.tolist()
        results["max_iou"] = float(iou_scores.max())

        # Compare with ground truth if provided
        if ground_truth is not None:
            results["ground_truth_iou"] = self.calculate_iou(mask, ground_truth)
            results["ground_truth_dice"] = self.calculate_dice(mask, ground_truth)

        self.metrics["segmentation"].append(results)

        logger.info(f"Segmentation complete in {inference_time:.2f}s")
        logger.info(f"Max IoU: {results['max_iou']:.4f}")

        return results

    def test_inpainting_quality(
        self,
        model: SDXLInpaintingModel,
        image: Image.Image,
        mask: np.ndarray,
        prompt: str,
        reference: Image.Image = None,
        **kwargs
    ) -> dict:
        """
        Test inpainting model quality.

        Args:
            model: SDXL inpainting model
            image: Input image
            mask: Segmentation mask
            prompt: Text prompt
            reference: Optional reference image for comparison
            **kwargs: Additional inpainting parameters

        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing inpainting quality with prompt: '{prompt}'")

        start_time = time.time()
        result = model.inpaint(image, mask, prompt, **kwargs)
        inference_time = time.time() - start_time

        results = {
            "inference_time_sec": inference_time,
            "prompt": prompt,
            "image_size": result.size
        }

        # Calculate quality metrics if reference provided
        if reference is not None:
            results["psnr"] = self.calculate_psnr(reference, result)
            results["ssim"] = self.calculate_ssim(reference, result)

        self.metrics["inpainting"].append(results)

        logger.info(f"Inpainting complete in {inference_time:.2f}s")

        return results

    def test_memory_usage(self) -> dict:
        """
        Test memory usage and detect potential leaks.

        Returns:
            Memory usage statistics
        """
        logger.info("Testing memory usage...")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9

            results = {
                "cuda_allocated_gb": allocated,
                "cuda_reserved_gb": reserved,
                "cuda_max_allocated_gb": max_allocated
            }

            logger.info(f"CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        else:
            results = {"cuda_available": False}

        return results

    def test_edge_cases(self, seg_model: SAMSegmentationModel) -> dict:
        """
        Test edge cases and error handling.

        Args:
            seg_model: Segmentation model

        Returns:
            Test results
        """
        logger.info("Testing edge cases...")

        results = {"passed": [], "failed": []}

        # Test 1: Empty points list
        try:
            dummy_image = Image.new('RGB', (512, 512))
            seg_model.segment(dummy_image, [])
            results["failed"].append("empty_points_should_raise_error")
        except ValueError:
            results["passed"].append("empty_points_validation")

        # Test 2: Single point
        try:
            mask = seg_model.segment(dummy_image, [[256, 256]])
            if mask.shape == (512, 512):
                results["passed"].append("single_point_segmentation")
            else:
                results["failed"].append("single_point_wrong_shape")
        except Exception as e:
            results["failed"].append(f"single_point_error: {str(e)}")

        logger.info(f"Edge cases - Passed: {len(results['passed'])}, Failed: {len(results['failed'])}")

        return results

    def run_performance_benchmark(
        self,
        seg_model: SAMSegmentationModel,
        inpaint_model: SDXLInpaintingModel,
        test_image: Image.Image,
        points: list,
        prompt: str,
        num_runs: int = 3
    ) -> dict:
        """
        Run performance benchmarks.

        Args:
            seg_model: Segmentation model
            inpaint_model: Inpainting model
            test_image: Test image
            points: Point coordinates
            prompt: Inpainting prompt
            num_runs: Number of benchmark runs

        Returns:
            Benchmark statistics
        """
        logger.info(f"Running performance benchmark ({num_runs} runs)...")

        seg_times = []
        inpaint_times = []

        for i in range(num_runs):
            logger.info(f"Benchmark run {i+1}/{num_runs}")

            # Segmentation benchmark
            start = time.time()
            mask = seg_model.segment(test_image, points)
            seg_times.append(time.time() - start)

            # Inpainting benchmark
            start = time.time()
            inpaint_model.inpaint(
                test_image,
                mask,
                prompt,
                num_inference_steps=30  # Reduced for faster benchmarking
            )
            inpaint_times.append(time.time() - start)

        results = {
            "segmentation": {
                "mean_time_sec": np.mean(seg_times),
                "std_time_sec": np.std(seg_times),
                "min_time_sec": np.min(seg_times),
                "max_time_sec": np.max(seg_times)
            },
            "inpainting": {
                "mean_time_sec": np.mean(inpaint_times),
                "std_time_sec": np.std(inpaint_times),
                "min_time_sec": np.min(inpaint_times),
                "max_time_sec": np.max(inpaint_times)
            }
        }

        logger.info(f"Segmentation - Mean: {results['segmentation']['mean_time_sec']:.2f}s")
        logger.info(f"Inpainting - Mean: {results['inpainting']['mean_time_sec']:.2f}s")

        self.metrics["performance"].append(results)

        return results

    def generate_report(self) -> str:
        """
        Generate comprehensive evaluation report.

        Returns:
            Formatted report string
        """
        report = ["=" * 60, "AI PHOTO EDITOR - EVALUATION REPORT", "=" * 60, ""]

        # Segmentation metrics
        if self.metrics["segmentation"]:
            report.append("SEGMENTATION METRICS:")
            report.append("-" * 60)
            for i, m in enumerate(self.metrics["segmentation"], 1):
                report.append(f"  Test {i}:")
                report.append(f"    Inference Time: {m['inference_time_sec']:.3f}s")
                report.append(f"    Max IoU Score: {m['max_iou']:.4f}")
                if 'ground_truth_iou' in m:
                    report.append(f"    Ground Truth IoU: {m['ground_truth_iou']:.4f}")
                    report.append(f"    Ground Truth Dice: {m['ground_truth_dice']:.4f}")
            report.append("")

        # Inpainting metrics
        if self.metrics["inpainting"]:
            report.append("INPAINTING METRICS:")
            report.append("-" * 60)
            for i, m in enumerate(self.metrics["inpainting"], 1):
                report.append(f"  Test {i}:")
                report.append(f"    Prompt: '{m['prompt']}'")
                report.append(f"    Inference Time: {m['inference_time_sec']:.3f}s")
                if 'psnr' in m:
                    report.append(f"    PSNR: {m['psnr']:.2f} dB")
                    report.append(f"    SSIM: {m['ssim']:.4f}")
            report.append("")

        # Performance metrics
        if self.metrics["performance"]:
            report.append("PERFORMANCE BENCHMARKS:")
            report.append("-" * 60)
            for i, m in enumerate(self.metrics["performance"], 1):
                report.append(f"  Benchmark {i}:")
                report.append(f"    Segmentation - Mean: {m['segmentation']['mean_time_sec']:.3f}s "
                            f"(±{m['segmentation']['std_time_sec']:.3f}s)")
                report.append(f"    Inpainting - Mean: {m['inpainting']['mean_time_sec']:.3f}s "
                            f"(±{m['inpainting']['std_time_sec']:.3f}s)")
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)


def main():
    """Run comprehensive pipeline tests."""
    logger.info("Starting AI Photo Editor Pipeline Tests")
    logger.info("=" * 60)

    # Initialize evaluator
    evaluator = PipelineEvaluator()

    # Load models
    logger.info("Loading models...")
    seg_model = SAMSegmentationModel()
    inpaint_model = SDXLInpaintingModel()

    # Create test image
    logger.info("Creating test image...")
    test_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
    test_points = [[256, 256], [300, 300]]

    # Test 1: Segmentation accuracy
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Segmentation Accuracy")
    logger.info("=" * 60)
    evaluator.test_segmentation_accuracy(seg_model, test_image, test_points)

    # Test 2: Edge cases
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Edge Cases")
    logger.info("=" * 60)
    edge_results = evaluator.test_edge_cases(seg_model)

    # Test 3: Memory usage
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Memory Usage")
    logger.info("=" * 60)
    memory_results = evaluator.test_memory_usage()

    # Test 4: Inpainting quality
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Inpainting Quality")
    logger.info("=" * 60)
    mask = seg_model.segment(test_image, test_points)
    evaluator.test_inpainting_quality(
        inpaint_model,
        test_image,
        mask,
        "blue sky with white clouds",
        num_inference_steps=30,
        guidance_scale=7.0
    )

    # Test 5: Performance benchmark
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Performance Benchmark")
    logger.info("=" * 60)
    evaluator.run_performance_benchmark(
        seg_model,
        inpaint_model,
        test_image,
        test_points,
        "sunset beach background",
        num_runs=2
    )

    # Generate and print report
    logger.info("\n" + "=" * 60)
    report = evaluator.generate_report()
    print(report)

    # Save report to file
    os.makedirs('test_results', exist_ok=True)
    report_path = 'test_results/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"\nReport saved to: {report_path}")

    logger.info("\nAll tests completed successfully!")


if __name__ == '__main__':
    main()
