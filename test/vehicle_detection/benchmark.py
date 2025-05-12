# Load both the pt and the trt model and test how long they took for inference
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from ai.vehicle_detection.core import logger
from shared_src.common import Config

CACHE_DIR: Path = Path(Config.get("global_cache_dir"), "vehicle_detection")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_path: Path):
    """Load the YOLO model from the specified path.

    Args:
        model_path (Path): Path to the model file.

    Returns:
        YOLO: Loaded YOLO model.
    """
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    model = YOLO(model_path, task="segment")
    return model


def benchmark_model(model: YOLO, image_path: Path):
    """Benchmark the model on a single image.

    Args:
        model (YOLO): The YOLO model to benchmark.
        image_path (Path): Path to the image file.

    Returns:
        float: Inference time in seconds.
    """
    # Load the image
    image = cv2.imread(str(image_path))

    # Perform inference
    model.predict(
        source=image,
        conf=0.5,
        project=Path(CACHE_DIR, "runs", "segment"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO model on a single image."
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "-i",
        "--image-path",
        type=str,
        required=True,
        help="Path to the image file.",
    )
    args = parser.parse_args()

    # Load the model
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)

    model = load_model(model_path)

    # Benchmark the model
    benchmark_model(model, image_path)


if __name__ == "__main__":
    main()
