/**
 * AI Photo Editor - Frontend Application
 * Handles user interactions, canvas manipulation, and API calls
 */

// State Management
const state = {
    originalImage: null,
    currentPoints: [],
    maskData: null,
    canvasSize: 512,
};

// DOM Elements
const elements = {
    imageUpload: document.getElementById('imageUpload'),
    originalCanvas: document.getElementById('originalCanvas'),
    maskCanvas: document.getElementById('maskCanvas'),
    resultCanvas: document.getElementById('resultCanvas'),
    pointsOverlay: document.getElementById('pointsOverlay'),
    pointsInfo: document.getElementById('pointsInfo'),
    maskInfo: document.getElementById('maskInfo'),
    resultInfo: document.getElementById('resultInfo'),
    clearPoints: document.getElementById('clearPoints'),
    generateMask: document.getElementById('generateMask'),
    inpaint: document.getElementById('inpaint'),
    prompt: document.getElementById('prompt'),
    negativePrompt: document.getElementById('negativePrompt'),
    guidanceScale: document.getElementById('guidanceScale'),
    guidanceValue: document.getElementById('guidanceValue'),
    numSteps: document.getElementById('numSteps'),
    stepsValue: document.getElementById('stepsValue'),
    seed: document.getElementById('seed'),
    progressBar: document.getElementById('progressBar'),
    statusMessage: document.getElementById('statusMessage'),
    deviceInfo: document.getElementById('deviceInfo'),
};

// Canvas Contexts
const ctx = {
    original: elements.originalCanvas.getContext('2d'),
    mask: elements.maskCanvas.getContext('2d'),
    result: elements.resultCanvas.getContext('2d'),
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    fetchDeviceInfo();
    updateSliderValues();
});

/**
 * Setup Event Listeners
 */
function setupEventListeners() {
    // Image upload
    elements.imageUpload.addEventListener('change', handleImageUpload);

    // Canvas click for point selection
    elements.originalCanvas.addEventListener('click', handleCanvasClick);

    // Buttons
    elements.clearPoints.addEventListener('click', clearPoints);
    elements.generateMask.addEventListener('click', generateMask);
    elements.inpaint.addEventListener('click', runInpainting);

    // Slider updates
    elements.guidanceScale.addEventListener('input', updateSliderValues);
    elements.numSteps.addEventListener('input', updateSliderValues);
}

/**
 * Handle image upload
 */
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showStatus('Please upload a valid image file', 'error');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showStatus('File size must be less than 10MB', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            // Store original image
            state.originalImage = img;

            // Reset state
            clearPoints();
            clearCanvas(ctx.mask, elements.maskCanvas);
            clearCanvas(ctx.result, elements.resultCanvas);

            // Draw image on canvas
            drawImageToCanvas(img, ctx.original, elements.originalCanvas);

            // Update UI
            elements.pointsInfo.textContent = 'Click on the foreground object';
            elements.maskInfo.textContent = 'Add points and generate mask';
            elements.resultInfo.textContent = 'Generate mask and enter prompt';
            elements.generateMask.disabled = true;
            elements.clearPoints.disabled = true;

            showStatus('Image loaded successfully. Click to add points.', 'success');
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

/**
 * Handle canvas click to add points
 */
function handleCanvasClick(event) {
    if (!state.originalImage) return;

    const rect = elements.originalCanvas.getBoundingClientRect();
    const scaleX = state.canvasSize / rect.width;
    const scaleY = state.canvasSize / rect.height;

    // Calculate click position relative to canvas
    const x = Math.round((event.clientX - rect.left) * scaleX);
    const y = Math.round((event.clientY - rect.top) * scaleY);

    // Add point to state
    state.currentPoints.push([x, y]);

    // Update visual indicator
    addPointMarker(
        (event.clientX - rect.left) / rect.width * 100,
        (event.clientY - rect.top) / rect.height * 100
    );

    // Update UI
    updatePointsInfo();
    elements.generateMask.disabled = false;
    elements.clearPoints.disabled = false;
}

/**
 * Add visual point marker
 */
function addPointMarker(xPercent, yPercent) {
    const marker = document.createElement('div');
    marker.className = 'point-marker';
    marker.style.left = `${xPercent}%`;
    marker.style.top = `${yPercent}%`;
    elements.pointsOverlay.appendChild(marker);
}

/**
 * Update points info text
 */
function updatePointsInfo() {
    const count = state.currentPoints.length;
    elements.pointsInfo.textContent = `${count} point${count !== 1 ? 's' : ''} selected. Click to add more or generate mask.`;
}

/**
 * Clear all points
 */
function clearPoints() {
    state.currentPoints = [];
    state.maskData = null;
    elements.pointsOverlay.innerHTML = '';
    elements.pointsInfo.textContent = 'Click on the foreground object';
    elements.generateMask.disabled = true;
    elements.clearPoints.disabled = true;
    elements.inpaint.disabled = true;
    clearCanvas(ctx.mask, elements.maskCanvas);
    clearCanvas(ctx.result, elements.resultCanvas);
}

/**
 * Generate segmentation mask
 */
async function generateMask() {
    if (!state.originalImage || state.currentPoints.length === 0) {
        showStatus('Add at least one point before generating mask', 'error');
        return;
    }

    showProgress('Generating segmentation mask...');
    elements.generateMask.disabled = true;

    try {
        // Convert canvas to base64
        const imageData = elements.originalCanvas.toDataURL('image/png');

        // Call API
        const response = await fetch('/api/segment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: imageData,
                points: state.currentPoints
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        // Store mask data
        state.maskData = data.mask_data;

        // Display mask
        const maskImg = new Image();
        maskImg.onload = () => {
            drawImageToCanvas(maskImg, ctx.mask, elements.maskCanvas);

            const maxIou = Math.max(...data.iou_scores);
            elements.maskInfo.textContent = `Mask generated (IoU: ${maxIou.toFixed(3)})`;
            elements.inpaint.disabled = false;

            hideProgress();
            showStatus('Mask generated successfully! Now enter a prompt.', 'success');
        };
        maskImg.src = data.mask;

    } catch (error) {
        console.error('Segmentation error:', error);
        hideProgress();
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        elements.generateMask.disabled = false;
    }
}

/**
 * Run inpainting
 */
async function runInpainting() {
    const prompt = elements.prompt.value.trim();

    if (!prompt) {
        showStatus('Please enter a prompt describing the background', 'error');
        elements.prompt.focus();
        return;
    }

    if (!state.maskData) {
        showStatus('Please generate a mask first', 'error');
        return;
    }

    showProgress('Generating inpainted image... (this may take 30-60 seconds)');
    elements.inpaint.disabled = true;

    try {
        const imageData = elements.originalCanvas.toDataURL('image/png');
        const negativePrompt = elements.negativePrompt.value.trim();
        const guidanceScale = parseFloat(elements.guidanceScale.value);
        const numSteps = parseInt(elements.numSteps.value);
        const seed = elements.seed.value ? parseInt(elements.seed.value) : null;

        // Call API
        const response = await fetch('/api/inpaint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: imageData,
                mask: state.maskData,
                prompt: prompt,
                negative_prompt: negativePrompt || null,
                guidance_scale: guidanceScale,
                num_steps: numSteps,
                seed: seed
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        // Display result
        const resultImg = new Image();
        resultImg.onload = () => {
            drawImageToCanvas(resultImg, ctx.result, elements.resultCanvas);
            elements.resultInfo.textContent = 'Inpainting complete!';

            hideProgress();
            showStatus('Inpainting complete! Right-click to save image.', 'success');
        };
        resultImg.src = data.result;

    } catch (error) {
        console.error('Inpainting error:', error);
        hideProgress();
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        elements.inpaint.disabled = false;
    }
}

/**
 * Fetch device information
 */
async function fetchDeviceInfo() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        const device = data.device.device.toUpperCase();
        const dtype = data.device.dtype;
        elements.deviceInfo.textContent = `Device: ${device} | Precision: ${dtype}`;
    } catch (error) {
        console.error('Failed to fetch device info:', error);
        elements.deviceInfo.textContent = 'Device info unavailable';
    }
}

/**
 * Update slider value displays
 */
function updateSliderValues() {
    elements.guidanceValue.textContent = elements.guidanceScale.value;
    elements.stepsValue.textContent = elements.numSteps.value;
}

/**
 * Draw image to canvas
 */
function drawImageToCanvas(img, context, canvas) {
    canvas.width = state.canvasSize;
    canvas.height = state.canvasSize;

    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image centered and scaled
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const x = (canvas.width - img.width * scale) / 2;
    const y = (canvas.height - img.height * scale) / 2;

    context.drawImage(img, x, y, img.width * scale, img.height * scale);
}

/**
 * Clear canvas
 */
function clearCanvas(context, canvas) {
    context.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Show progress indicator
 */
function showProgress(message) {
    elements.progressBar.classList.remove('hidden');
    elements.progressBar.querySelector('.progress-text').textContent = message;
}

/**
 * Hide progress indicator
 */
function hideProgress() {
    elements.progressBar.classList.add('hidden');
}

/**
 * Show status message
 */
function showStatus(message, type = 'info') {
    elements.statusMessage.textContent = message;
    elements.statusMessage.className = `status-message show ${type}`;

    // Auto-hide after 5 seconds
    setTimeout(() => {
        elements.statusMessage.classList.remove('show');
    }, 5000);
}
