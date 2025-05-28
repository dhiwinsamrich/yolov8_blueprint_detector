"""
Enhanced Blueprint Object Detection API

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

Author: Dhiwin Samrich
Date: 28th May 2025
Version: 1.1.0
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
    version="1.1.0",
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

@app.get("/docs-html", tags=["Documentation"], include_in_schema=False)
async def custom_docs():
    """Custom HTML documentation page"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{app.title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            .endpoint {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .method {{ font-weight: bold; color: #27ae60; }}
            code {{ background-color: #eee; padding: 2px 5px; border-radius: 3px; }}
            .try-it {{ display: inline-block; margin-top: 10px; padding: 8px 15px; background-color: #3498db; color: white; text-decoration: none; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h1>{app.title}</h1>
        <p>Version: {app.version}</p>
        <p>{app.description}</p>
        
        <h2>Endpoints</h2>
        
        <div class="endpoint">
            <p><span class="method">GET</span> <code>/</code> - Root endpoint</p>
            <p>Returns basic API information and status.</p>
            <a href="/" class="try-it">Try it</a>
        </div>
        
        <div class="endpoint">
            <p><span class="method">POST</span> <code>/detect</code> - Object detection</p>
            <p>Detect doors and windows in architectural blueprints.</p>
            <a href="/docs#/default/detect_detect_post" class="try-it">Try it in Swagger UI</a>
        </div>
        
        <div class="endpoint">
            <p><span class="method">POST</span> <code>/visualize</code> - Detection visualization</p>
            <p>Get an image with bounding boxes drawn around detected objects.</p>
            <a href="/docs#/default/visualize_visualize_post" class="try-it">Try it in Swagger UI</a>
        </div>
        
        <div class="endpoint">
            <p><span class="method">GET</span> <code>/health</code> - Health check</p>
            <p>Check system and model status.</p>
            <a href="/health" class="try-it">Try it</a>
        </div>
        
        <h2>Documentation</h2>
        <p><a href="/docs">Interactive Swagger UI</a></p>
        <p><a href="/redoc">ReDoc Documentation</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def main():
    """Run the application using Uvicorn"""
    import uvicorn
    
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