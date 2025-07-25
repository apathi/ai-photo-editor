"""
Flask web application for AI-powered photo editing with inpainting.
Provides REST API endpoints for segmentation and background replacement.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from dotenv import load_dotenv

from models import SAMSegmentationModel, SDXLInpaintingModel
from utils import (
    resize_to_square,
    image_to_base64,
    base64_to_image,
    mask_to_rgba,
    get_device_info
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "512"))
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

# Global model instances (loaded lazily)
segmentation_model = None
inpainting_model = None


def get_segmentation_model():
    """Lazy load SAM segmentation model."""
    global segmentation_model
    if segmentation_model is None:
        logger.info("Initializing SAM segmentation model...")
        segmentation_model = SAMSegmentationModel()
    return segmentation_model


def get_inpainting_model():
    """Lazy load SDXL inpainting model."""
    global inpainting_model
    if inpainting_model is None:
        logger.info("Initializing SDXL inpainting model...")
        # Read CPU offloading setting from environment
        enable_offload = os.getenv('ENABLE_CPU_OFFLOAD', 'True').lower() == 'true'
        logger.info(f"Environment ENABLE_CPU_OFFLOAD: {os.getenv('ENABLE_CPU_OFFLOAD')}")
        logger.info(f"Parsed enable_offload value: {enable_offload}")
        inpainting_model = SDXLInpaintingModel(enable_offload=enable_offload)
    return inpainting_model


@app.route('/')
def index():
    """Render main application page."""
    screenshot_mode = request.args.get('mode') == 'screenshot'
    return render_template('index.html', screenshot_mode=screenshot_mode)


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    Returns device information and model status.
    """
    device_info = get_device_info()
    status = {
        "status": "healthy",
        "device": device_info,
        "models": {
            "segmentation": segmentation_model is not None,
            "inpainting": inpainting_model is not None
        }
    }
    return jsonify(status)


@app.route('/api/segment', methods=['POST'])
def segment():
    """
    Generate segmentation mask from user points.

    Request JSON:
        {
            "image": "base64_encoded_image",
            "points": [[x1, y1], [x2, y2], ...]
        }

    Response JSON:
        {
            "mask": "base64_encoded_mask_visualization",
            "iou_scores": [float, ...]
        }
    """
    try:
        data = request.get_json()

        if not data or 'image' not in data or 'points' not in data:
            return jsonify({"error": "Missing 'image' or 'points' in request"}), 400

        # Parse input
        image_b64 = data['image']
        points = data['points']

        if not points or len(points) == 0:
            return jsonify({"error": "At least one point is required"}), 400

        # Convert base64 to PIL Image
        image = base64_to_image(image_b64)

        # Resize to standard size
        image = resize_to_square(image, IMAGE_SIZE)

        # Load model and perform segmentation
        model = get_segmentation_model()
        mask = model.segment(image, points)

        # Get IoU scores for quality assessment
        iou_scores = model.get_iou_scores(image, points)

        # Convert mask to RGBA visualization
        mask_rgba = mask_to_rgba(mask, color=(0, 255, 0))
        mask_image = Image.fromarray(mask_rgba)

        # Convert to base64 for response
        mask_b64 = image_to_base64(mask_image)

        logger.info(f"Segmentation complete. Points: {len(points)}, IoU: {iou_scores.max():.4f}")

        return jsonify({
            "mask": mask_b64,
            "iou_scores": iou_scores.tolist(),
            "mask_data": mask.tolist()  # Include raw mask for inpainting
        })

    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/inpaint', methods=['POST'])
def inpaint():
    """
    Generate inpainted image with new background.

    Request JSON:
        {
            "image": "base64_encoded_image",
            "mask": [[0, 1, ...], ...],  // 2D array
            "prompt": "description of new background",
            "negative_prompt": "things to avoid (optional)",
            "guidance_scale": 7.0,
            "num_steps": 50,
            "seed": 12345
        }

    Response JSON:
        {
            "result": "base64_encoded_inpainted_image"
        }
    """
    try:
        data = request.get_json()

        required_fields = ['image', 'mask', 'prompt']
        if not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing required fields: {required_fields}"}), 400

        # Parse input
        image_b64 = data['image']
        mask_data = data['mask']
        prompt = data['prompt'].strip()
        negative_prompt = data.get('negative_prompt', '').strip() or None
        guidance_scale = float(data.get('guidance_scale', 7.0))
        num_steps = int(data.get('num_steps', 50))
        seed = data.get('seed')

        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400

        # Convert inputs
        image = base64_to_image(image_b64)
        image = resize_to_square(image, IMAGE_SIZE)
        mask = np.array(mask_data, dtype=bool)

        # Load model and perform inpainting
        model = get_inpainting_model()
        result = model.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            seed=int(seed) if seed is not None else None
        )

        # Convert result to base64
        result_b64 = image_to_base64(result)

        logger.info(f"Inpainting complete. Prompt: '{prompt[:50]}...', Steps: {num_steps}")

        return jsonify({
            "result": result_b64
        })

    except Exception as e:
        logger.error(f"Inpainting error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded."""
    return jsonify({"error": "File too large. Maximum size is 10MB"}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting AI Photo Editor on port {port}")
    logger.info(f"Debug mode: {debug}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
