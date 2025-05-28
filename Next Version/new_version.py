"""
Enhanced Blueprint Object Detection API with Drag & Drop Interface

This FastAPI application provides an endpoint for detecting doors and windows in
architectural blueprints using YOLOv8. The API accepts PNG/JPG images and returns
detection results in a standardized JSON format.

Key Improvements:
- Better model loading with fallback options
- Enhanced error handling and logging
- More detailed health check endpoint
- Additional documentation and metadata
- Better configuration management
- Improved response formats
- New interactive drag and drop interface
- Result visualization interface

Author: Dhiwin Samrich
Date: 28th May 2025
Version: 1.2.0
"""

import os
import io
import time
import base64
import logging
import platform
import sys
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with enhanced metadata
app = FastAPI(
    title="Blueprint Detection API",
    version="1.2.0",
    description="""Detect doors and windows from architectural blueprints using YOLOv8.
    Provides both detection results and visualization capabilities.""",
    contact={
        "name": "Dhiwin Samrich",
        "email": "dhiwin@example.com",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve static files (for the interactive interface)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
class Config:
    MODEL_PATHS = [
        os.path.join("models", "blueprint_detector.pt"),  # Primary model path
        os.path.join("runs", "detect", "train", "weights", "blueprint_detector.pt"),  # Common training output path
        "blueprint_detector.pt"  # Fallback name
    ]
    CLASS_NAMES = ['door', 'window']
    DEFAULT_CONFIDENCE = 0.25
    DEFAULT_IOU = 0.45

model: Optional[YOLO] = None
model_path: Optional[str] = None

def find_model() -> Optional[str]:
    """Search for model in possible locations"""
    for path in Config.MODEL_PATHS:
        if os.path.exists(path):
            logger.info(f"Found model at: {path}")
            return path
    logger.warning("No model found in any of the searched locations")
    return None

def load_model(path: str) -> Optional[YOLO]:
    """Load YOLO model with error handling"""
    try:
        torch.set_num_threads(1)  # Optimize for CPU usage
        logger.info(f"Loading model from {path}...")
        yolo_model = YOLO(path)
        yolo_model.to('cpu')
        logger.info(f"✅ Model loaded successfully from {path}")
        return yolo_model
    except Exception as e:
        logger.error(f"❌ Failed to load model from {path}: {str(e)}")
        return None

@app.on_event("startup")
async def startup_event():
    """Load the YOLO model on startup with fallback logic"""
    global model, model_path
    
    logger.info("Starting API initialization...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Try to find and load custom model
    model_path = find_model()
    if model_path:
        model = load_model(model_path)
    
    # Fallback to pretrained model if custom model not found
    if model is None:
        logger.warning("No custom model found, falling back to pretrained YOLOv8n")
        try:
            model = YOLO("yolov8n.pt")
            model.to('cpu')
            model_path = "yolov8n.pt"
            logger.info("✅ Pretrained YOLOv8n model loaded as fallback")
        except Exception as e:
            logger.error(f"❌ Failed to load fallback model: {str(e)}")
            raise RuntimeError("Could not load any model")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Blueprint Detection API",
        "version": app.version,
        "status": "running",
        "model_loaded": bool(model),
        "model_path": model_path
    }

@app.get("/interactive", tags=["UI"], response_class=HTMLResponse)
async def interactive_interface():
    """Interactive drag and drop interface for blueprint detection"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Blueprint Detection - Interactive</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f7fa;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }}
            .container {{
                display: flex;
                flex-direction: column;
                gap: 30px;
            }}
            .upload-section {{
                background-color: white;
                border-radius: 8px;
                padding: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
            }}
            .drop-area {{
                border: 3px dashed #3498db;
                border-radius: 5px;
                padding: 40px;
                margin: 20px 0;
                cursor: pointer;
                transition: all 0.3s;
                background-color: #f8fafc;
            }}
            .drop-area:hover {{
                background-color: #e8f4fc;
                border-color: #2980b9;
            }}
            .drop-area.highlight {{
                background-color: #e8f4fc;
                border-color: #2980b9;
            }}
            .file-input {{
                display: none;
            }}
            .btn {{
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }}
            .btn:hover {{
                background-color: #2980b9;
            }}
            .btn:disabled {{
                background-color: #95a5a6;
                cursor: not-allowed;
            }}
            .results-section {{
                display: none;
                background-color: white;
                border-radius: 8px;
                padding: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .image-container {{
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }}
            .image-box {{
                flex: 1;
                min-width: 300px;
                text-align: center;
            }}
            .image-box img {{
                max-width: 100%;
                max-height: 500px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .detections-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            .detections-table th, .detections-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            .detections-table th {{
                background-color: #3498db;
                color: white;
            }}
            .detections-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .detections-table tr:hover {{
                background-color: #e6f7ff;
            }}
            .loading {{
                display: none;
                text-align: center;
                margin: 20px 0;
            }}
            .spinner {{
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .status {{
                margin-top: 10px;
                font-weight: bold;
            }}
            .error {{
                color: #e74c3c;
                margin-top: 20px;
            }}
            .success {{
                color: #27ae60;
            }}
            .controls {{
                display: flex;
                justify-content: center;
                gap: 15px;
                margin-top: 20px;
            }}
            .slider-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .slider-container label {{
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
            }}
            .slider {{
                width: 300px;
                max-width: 100%;
            }}
        </style>
    </head>
    <body>
        <h1>Blueprint Object Detection</h1>
        
        <div class="container">
            <div class="upload-section">
                <h2>Upload Blueprint Image</h2>
                <p>Drag & drop your blueprint image file here or click to select</p>
                
                <div id="dropArea" class="drop-area">
                    <p>PNG, JPG, or JPEG files only</p>
                    <input type="file" id="fileInput" class="file-input" accept="image/png, image/jpeg">
                    <button class="btn" id="selectBtn">Select File</button>
                </div>
                
                <div class="slider-container">
                    <label for="confidenceSlider">Confidence Threshold: <span id="confidenceValue">0.25</span></label>
                    <input type="range" id="confidenceSlider" class="slider" min="0.01" max="0.99" step="0.01" value="0.25">
                </div>
                
                <div class="slider-container">
                    <label for="iouSlider">IOU Threshold: <span id="iouValue">0.45</span></label>
                    <input type="range" id="iouSlider" class="slider" min="0.01" max="0.99" step="0.01" value="0.45">
                </div>
                
                <div class="controls">
                    <button class="btn" id="processBtn" disabled>Process Image</button>
                    <button class="btn" id="resetBtn">Reset</button>
                </div>
                
                <div id="status" class="status"></div>
                <div id="error" class="error"></div>
                
                <div id="loading" class="loading">
                    <div class="spinner"></div>
                    <p>Processing image...</p>
                </div>
            </div>
            
            <div id="resultsSection" class="results-section">
                <h2>Detection Results</h2>
                <div class="image-container">
                    <div class="image-box">
                        <h3>Original Image</h3>
                        <img id="originalImage" src="" alt="Original image">
                    </div>
                    <div class="image-box">
                        <h3>Detections</h3>
                        <img id="resultImage" src="" alt="Result with detections">
                    </div>
                </div>
                
                <h3>Detection Details</h3>
                <table class="detections-table">
                    <thead>
                        <tr>
                            <th>Label</th>
                            <th>Confidence</th>
                            <th>Bounding Box (px)</th>
                            <th>Normalized BBox</th>
                        </tr>
                    </thead>
                    <tbody id="detectionsBody">
                        <!-- Detections will be inserted here -->
                    </tbody>
                </table>
                
                <div class="controls">
                    <button class="btn" id="downloadBtn">Download Results</button>
                    <button class="btn" id="newImageBtn">Process New Image</button>
                </div>
            </div>
        </div>
        
        <script>
            // DOM elements
            const dropArea = document.getElementById('dropArea');
            const fileInput = document.getElementById('fileInput');
            const selectBtn = document.getElementById('selectBtn');
            const processBtn = document.getElementById('processBtn');
            const resetBtn = document.getElementById('resetBtn');
            const downloadBtn = document.getElementById('downloadBtn');
            const newImageBtn = document.getElementById('newImageBtn');
            const loadingDiv = document.getElementById('loading');
            const statusDiv = document.getElementById('status');
            const errorDiv = document.getElementById('error');
            const resultsSection = document.getElementById('resultsSection');
            const originalImage = document.getElementById('originalImage');
            const resultImage = document.getElementById('resultImage');
            const detectionsBody = document.getElementById('detectionsBody');
            const confidenceSlider = document.getElementById('confidenceSlider');
            const iouSlider = document.getElementById('iouSlider');
            const confidenceValue = document.getElementById('confidenceValue');
            const iouValue = document.getElementById('iouValue');
            
            // Variables
            let selectedFile = null;
            let imagePreviewUrl = null;
            
            // Event listeners
            selectBtn.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', handleFileSelect);
            dropArea.addEventListener('dragover', handleDragOver);
            dropArea.addEventListener('dragleave', handleDragLeave);
            dropArea.addEventListener('drop', handleDrop);
            processBtn.addEventListener('click', processImage);
            resetBtn.addEventListener('click', resetForm);
            downloadBtn.addEventListener('click', downloadResults);
            newImageBtn.addEventListener('click', resetForm);
            confidenceSlider.addEventListener('input', updateSliderValue);
            iouSlider.addEventListener('input', updateSliderValue);
            
            // Update slider value displays
            function updateSliderValue() {{
                confidenceValue.textContent = confidenceSlider.value;
                iouValue.textContent = iouSlider.value;
            }}
            
            // Handle file selection
            function handleFileSelect(e) {{
                const file = e.target.files[0];
                if (file) validateAndPreviewFile(file);
            }}
            
            // Handle drag over
            function handleDragOver(e) {{
                e.preventDefault();
                dropArea.classList.add('highlight');
            }}
            
            // Handle drag leave
            function handleDragLeave() {{
                dropArea.classList.remove('highlight');
            }}
            
            // Handle drop
            function handleDrop(e) {{
                e.preventDefault();
                dropArea.classList.remove('highlight');
                
                const file = e.dataTransfer.files[0];
                if (file) validateAndPreviewFile(file);
            }}
            
            // Validate and preview file
            function validateAndPreviewFile(file) {{
                // Check file type
                const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
                if (!validTypes.includes(file.type)) {{
                    showError('Please upload a JPEG or PNG image file.');
                    return;
                }}
                
                // Check file size (max 10MB)
                if (file.size > 10 * 1024 * 1024) {{
                    showError('File size too large. Maximum 10MB allowed.');
                    return;
                }}
                
                // Clear any previous errors
                clearError();
                
                // Store the file
                selectedFile = file;
                
                // Create preview
                const reader = new FileReader();
                reader.onload = (e) => {{
                    imagePreviewUrl = e.target.result;
                    statusDiv.textContent = `Selected: ${{file.name}} (${{(file.size / 1024 / 1024).toFixed(2)}} MB)`;
                    statusDiv.className = 'status success';
                    processBtn.disabled = false;
                }};
                reader.readAsDataURL(file);
            }}
            
            // Process the image
            async function processImage() {{
                if (!selectedFile) return;
                
                // Show loading state
                loadingDiv.style.display = 'block';
                processBtn.disabled = true;
                clearError();
                
                try {{
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    formData.append('conf', confidenceSlider.value);
                    formData.append('iou', iouSlider.value);
                    formData.append('return_image', 'true');
                    
                    const response = await fetch('/detect', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    if (!response.ok) {{
                        const error = await response.json();
                        throw new Error(error.detail || 'Failed to process image');
                    }}
                    
                    const result = await response.json();
                    
                    // Display results
                    displayResults(result);
                    
                }} catch (error) {{
                    showError(error.message);
                }} finally {{
                    loadingDiv.style.display = 'none';
                }}
            }}
            
            // Display results
            function displayResults(result) {{
                // Show original and result images
                originalImage.src = imagePreviewUrl;
                resultImage.src = result.visualization;
                
                // Populate detections table
                detectionsBody.innerHTML = '';
                result.detections.forEach(det => {{
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${{det.label}}</td>
                        <td>${{(det.confidence * 100).toFixed(1)}}%</td>
                        <td>[x:${{det.bbox[0].toFixed(0)}}, y:${{det.bbox[1].toFixed(0)}}, 
                            w:${{det.bbox[2].toFixed(0)}}, h:${{det.bbox[3].toFixed(0)}}]</td>
                        <td>[x:${{det.bbox_normalized[0].toFixed(4)}}, y:${{det.bbox_normalized[1].toFixed(4)}}, 
                            w:${{det.bbox_normalized[2].toFixed(4)}}, h:${{det.bbox_normalized[3].toFixed(4)}}]</td>
                    `;
                    detectionsBody.appendChild(row);
                }});
                
                // Update status
                statusDiv.textContent = `Processed successfully - ${{result.detections.length}} detections found`;
                statusDiv.className = 'status success';
                
                // Show results section
                resultsSection.style.display = 'block';
                
                // Scroll to results
                resultsSection.scrollIntoView({{ behavior: 'smooth' }});
            }}
            
            // Download results
            function downloadResults() {{
                if (!resultImage.src) return;
                
                const link = document.createElement('a');
                link.href = resultImage.src;
                link.download = 'detections_' + selectedFile.name;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }}
            
            // Reset form
            function resetForm() {{
                selectedFile = null;
                imagePreviewUrl = null;
                fileInput.value = '';
                statusDiv.textContent = '';
                errorDiv.textContent = '';
                processBtn.disabled = true;
                resultsSection.style.display = 'none';
                detectionsBody.innerHTML = '';
            }}
            
            // Show error
            function showError(message) {{
                errorDiv.textContent = message;
                statusDiv.textContent = '';
                processBtn.disabled = true;
            }}
            
            // Clear error
            function clearError() {{
                errorDiv.textContent = '';
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", tags=["System"])
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "ok" if model else "error",
        "model_loaded": bool(model),
        "model_path": model_path,
        "model_type": "custom" if model_path and "yolov8n.pt" not in model_path else "pretrained",
        "classes": Config.CLASS_NAMES,
        "system": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "device": "cpu",
            "cwd": os.getcwd()
        },
        "timestamp": time.time(),
        "response_time": time.process_time()
    }

@app.post("/detect", tags=["Detection"])
async def detect(
    file: UploadFile = File(..., description="Blueprint image file (PNG/JPG)"),
    conf: float = Query(Config.DEFAULT_CONFIDENCE, ge=0.01, le=0.99, 
                       description="Confidence threshold for detections"),
    iou: float = Query(Config.DEFAULT_IOU, ge=0.01, le=0.99,
                      description="Intersection over Union threshold for NMS"),
    return_image: bool = Query(False, description="Return base64 image with detections")
) -> Dict:
    """
    Detect doors and windows in architectural blueprints.
    
    Processes an uploaded image and returns:
    - List of detections with bounding boxes and confidence scores
    - Optional visualization image with detections
    
    Returns:
        Dictionary containing detection results and metadata
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Only image files are allowed (PNG/JPG/JPEG)."
        )
    
    try:
        start_time = time.time()
        
        # Read and validate image
        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Perform inference
        results = model.predict(
            image,
            conf=conf,
            iou=iou,
            verbose=False  # Disable verbose logging for API
        )[0]
        
        # Process detections
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            width, height = x2 - x1, y2 - y1
            
            # Get class name with fallback
            label = (
                Config.CLASS_NAMES[class_id] 
                if class_id < len(Config.CLASS_NAMES) 
                else f"class_{class_id}"
            )
            
            detections.append({
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(width, 2), round(height, 2)],
                "bbox_normalized": [
                    round(x1/image.width, 4), 
                    round(y1/image.height, 4),
                    round(width/image.width, 4),
                    round(height/image.height, 4)
                ]
            })
        
        # Build response
        response = {
            "success": True,
            "detections": detections,
            "image_info": {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode
            },
            "metrics": {
                "response_time_seconds": round(time.time() - start_time, 4),
                "inference_time_seconds": round(results.speed['inference'] / 1000, 4),
                "detection_count": len(detections)
            },
            "model": {
                "type": "custom" if model_path and "yolov8n.pt" not in model_path else "pretrained",
                "path": model_path,
                "confidence_threshold": conf,
                "iou_threshold": iou
            }
        }
        
        # Add visualization if requested
        if return_image:
            img_with_boxes = results.plot()
            result_img = Image.fromarray(img_with_boxes)
            buffered = io.BytesIO()
            result_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            response["visualization"] = f"data:image/png;base64,{img_str}"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during detection: {str(e)}"
        )

@app.post("/visualize", tags=["Visualization"])
async def visualize(
    file: UploadFile = File(..., description="Blueprint image file (PNG/JPG)"),
    conf: float = Query(Config.DEFAULT_CONFIDENCE, ge=0.01, le=0.99,
                       description="Confidence threshold for detections")
) -> StreamingResponse:
    """Returns an image with bounding boxes drawn around detected objects"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        results = model.predict(image, conf=conf)[0]
        img_with_boxes = results.plot()
        
        img = Image.fromarray(img_with_boxes)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "X-Detection-Count": str(len(results.boxes)),
                "X-Inference-Time": f"{results.speed['inference']}ms"
            }
        )
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during visualization: {str(e)}"
        )

def main():
    """Run the application using Uvicorn"""
    import uvicorn
    
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()