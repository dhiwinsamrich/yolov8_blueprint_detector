from ultralytics import YOLO
import shutil
import os

model = YOLO('yolov8n.pt')  # or yolov8s.pt
results = model.train(
    data='config.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    name='doors_windows',
    exist_ok=True
)

# Copy the best model to the project root directory
best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
if os.path.exists(best_model_path):
    shutil.copy(best_model_path, "best.pt")
    print(f"Best model copied to {os.path.abspath('best.pt')}")
