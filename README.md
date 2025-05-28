# Blueprint Detection Project

A YOLOv8-powered FastAPI application for detecting doors and windows in architectural blueprint images. This version introduces a fully interactive drag-and-drop interface, detailed detection outputs, image visualization, and intelligent model management.

````markdown
# 🏗️ Blueprint Detection Project

A YOLOv8-powered FastAPI application for detecting **doors and windows** in architectural blueprint images. This version introduces a fully **interactive drag-and-drop interface**, detailed detection outputs, image visualization, and intelligent model management.

---

## ✨ Features

- 🧠 YOLOv8 object detection (custom or fallback model)
- 🖼️ Interactive web interface at `/interactive`
- 📄 JSON API with bounding boxes and confidence scores
- 🎨 Optional image visualization with detections
- 📊 Inference time and system diagnostics
- ⚙️ Docker-ready deployment
- 🔒 CORS-enabled for frontend integration

---

## 🐳 Docker Setup

### 📦 Prerequisites

- Docker and Docker Compose installed
- Place a trained YOLOv8 model file in `models/` (e.g. `blueprint_detector.pt` or `best.pt`)  
  If not found, it falls back to `yolov8n.pt`.

### 🧱 Build and Run

```bash
docker-compose up --build
````

Then visit:

* 🔗 API root: [http://localhost:8000](http://localhost:8000)
* 🎮 Interactive UI: [http://localhost:8000/interactive](http://localhost:8000/interactive)
* 📚 API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* 📘 Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📤 Example API Request

Send an image to the `/detect` endpoint using `curl`:

```bash
curl -X POST -F "file=@path/to/image.jpg" \
"http://localhost:8000/detect?conf=0.25&iou=0.45&return_image=true"
```

✅ Returns JSON with detection info and optionally a base64 image with bounding boxes.

---

## 🛑 Stop the Container

```bash
docker-compose down
```

---

## 🧪 Available Endpoints

| Method | Endpoint       | Description                             |
| ------ | -------------- | --------------------------------------- |
| GET    | `/`            | Root API info                           |
| GET    | `/interactive` | Interactive drag-and-drop frontend      |
| POST   | `/detect`      | Submit an image for detection           |
| POST   | `/visualize`   | Get annotated image                     |
| GET    | `/health`      | Check model status and environment info |

---

## 🧑‍💻 Local Development (No Docker)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run with Uvicorn:

   ```bash
   uvicorn new_version:app --host 0.0.0.0 --port 8000 --reload
   ```

Access the app at:

* `http://localhost:8000/interactive`

> To make it available at `http://localhost/interactive`, run on port 80:

```bash
sudo uvicorn new_version:app --host 0.0.0.0 --port 80
```

---

## 🗂 Project Structure

```
📦project-root
├── new_version.py           # Main FastAPI app
├── static/                  # HTML, CSS, JS for frontend UI
├── models/                  # YOLOv8 model files
├── Dockerfile               # Docker build definition
├── docker-compose.yml       # Docker Compose setup
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🔍 Model Loading Strategy

1. Searches for custom weights:

   * `models/blueprint_detector.pt`
   * `runs/detect/train/weights/blueprint_detector.pt`
   * `blueprint_detector.pt`
2. Falls back to pretrained `yolov8n.pt` if custom model not found
3. Model runs on CPU and loads during startup

---

## 📊 Example Detection Response

```json
{
  "success": true,
  "detections": [
    {
      "label": "door",
      "confidence": 0.87,
      "bbox": [125.4, 210.8, 56.2, 118.7],
      "bbox_normalized": [0.12, 0.22, 0.05, 0.13]
    },
    ...
  ],
  "visualization": "data:image/png;base64,...",
  "metrics": {
    "response_time_seconds": 0.342,
    "inference_time_seconds": 0.021,
    "detection_count": 4
  }
}
```

---

## 👤 Author

**Dhiwin Samrich**
📧 [dhiwin@example.com](mailto:dhiwin@example.com)

---

## 🪪 License

This project is licensed under the **MIT License**.

```

---

Let me know if you want to include:
- Badges (build, license, Python version)
- GitHub Actions CI/CD instructions
- DockerHub or Hugging Face deployment guides

I can generate those instantly.
```
