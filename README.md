# Blueprint Detection Project

This project uses YOLOv8 to detect doors and windows in blueprint images. It provides a FastAPI backend.

## Docker Setup

### Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- A trained model file (e.g., `blueprint_detector.pt` or `best.pt`) in the `models/` directory. If not found, it will attempt to use a pretrained `yolov8n.pt`.

### Building and Running with Docker

1. Place your trained model (e.g., `blueprint_detector.pt` or `best.pt`) in the `models/` directory. If no model is found in the specified paths, a pretrained `yolov8n.pt` will be used as a fallback.

2. Build and start the container using Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. The API will be available at: http://localhost:8000
   You can access the API documentation at http://localhost:8000/docs or http://localhost:8000/redoc.

### API Usage

Send a POST request to `/detect` with an image file:

```bash
curl -X POST -F "file=@path/to/your/image.jpg" "http://localhost:8000/detect?conf=0.25&iou=0.45&return_image=false"
```

The response will be a JSON object containing detected doors and windows with their coordinates and confidence scores.

### Stopping the Container

```bash
docker-compose down
```

## Development

If you want to run the application without Docker:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application using Uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   The `--reload` flag is useful for development as it automatically reloads the server when code changes are detected.

## Project Structure

- `app.py`: FastAPI application that serves the detection API
- `train.py`: Script for training the YOLOv8 model
- `models/blueprint_detector.pt` or `models/best.pt`: Trained YOLOv8 model (primary). Falls back to `yolov8n.pt` if not found.
- `config.yaml`: Configuration for the dataset
- `Dockerfile`: Instructions for building the Docker image
- `docker-compose.yml`: Configuration for Docker Compose