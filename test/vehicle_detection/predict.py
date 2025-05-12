from ultralytics import YOLO

from ai.vehicle_detection.core import Path, logger
from shared_src.common import Config

CACHE_DIR: Path = Path(Config.get("global_cache_dir"), "vehicle_detection")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def main():
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
        conf=0.5,
        show=True,
        save=True,
        project=Path(CACHE_DIR, "runs", "segment"),
    )


if __name__ == "__main__":
    main()
