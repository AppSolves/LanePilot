import argparse

from ultralytics import YOLO

from ai.vehicle_detection.core import Path, logger
from shared_src.common import Config

CACHE_DIR: Path = Path(Config.get("global_cache_dir"), "vehicle_detection")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def main(confidence: float = 0.5) -> None:
    dataset_path = Path(CACHE_DIR, "test", "images")
    model_path = Path(
        Config.get("global_assets_dir"),
        "trained_models",
        "vehicle_detection",
        "vehicle_detection.pt",
    )
    if not model_path.is_file():
        logger.error(f"Model not found at {model_path}")
        return

    logger.debug(f"Model path: {model_path}")
    logger.debug(f"Dataset path: {dataset_path}")

    # Load the model
    model = YOLO(model_path)
    logger.debug(f"Predicting images in {dataset_path}")
    model.predict(
        source=dataset_path,
        conf=confidence,
        show=True,
        save=True,
        project=Path(CACHE_DIR, "runs", "segment"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Detection Prediction")
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions",
    )
    args = parser.parse_args()
    logger.debug(f"Confidence: {args.confidence}")

    main(args.confidence)
