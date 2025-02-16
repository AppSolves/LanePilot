import os

from ultralytics import YOLO

DATASET_PATH: str = os.path.join("Data", "Thermal camera", "person_detection")


def main():
    model = YOLO(os.path.join(DATASET_PATH, "models", "person_thermal_detection.pt"))
    for root, _, files in os.walk(os.path.join(DATASET_PATH, "test", "images")):
        for file in files:
            results = model.predict(os.path.join(root, file))
            results[0].show()


if __name__ == "__main__":
    main()
