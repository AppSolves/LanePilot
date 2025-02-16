import atexit
import os
import shutil

from ultralytics import YOLO

DATASET_PATH: str = os.path.join("Data", "Thermal camera", "person_detection")


def delete_cache():
    shutil.rmtree("runs", ignore_errors=True)
    if os.path.isfile("yolo11n.pt"):
        os.remove("yolo11n.pt")


def main():
    atexit.register(delete_cache)
    model = YOLO()
    model.train(
        data=os.path.join(DATASET_PATH, "data.yaml"),
        epochs=100,
        patience=7,
        device="cuda",
        workers=8,
        imgsz=416,
    )
    shutil.copy(
        "runs/detect/train/weights/best.pt",
        os.path.join(DATASET_PATH, "models", "person_thermal_detection.pt"),
    )


if __name__ == "__main__":
    main()
