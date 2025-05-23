import os
import shutil
from pathlib import Path

from ultralytics import YOLO

from shared_src.data_preprocessing import convert_class_to_segment, unpack_dataset
from shared_src.postprocessing import export_model_to_trt

from .core import MODULE_CONFIG, Config, logger


def train():
    """Main function to train the YOLO model.

    Raises:
        FileNotFoundError: If the config file is not found.
    """
    # Load the training configuration
    model_config = MODULE_CONFIG.get("model")
    dataset_config = MODULE_CONFIG.get("dataset")

    model_name = model_config.get("name")
    dataset_path = Path(dataset_config.get("path"))

    logger.info(f"Model name: {model_name}")
    logger.info(f"Dataset path: {dataset_path}")

    # Check if the model name and dataset path are provided
    if not model_name or not dataset_path:
        logger.error("Model name or dataset path is not provided.")
        raise ValueError("Model name and dataset path must be provided.")

    # Unpack the dataset and remove previous training runs
    dataset_path = unpack_dataset(dataset_path, "vehicle_detection")
    shutil.rmtree(Path(dataset_path, "runs/segment/train"), ignore_errors=True)

    # Convert classes to segments if specified in the config
    for class_id in dataset_config.get("convert_classes_with_ids", []):
        convert_class_to_segment(dataset_path, class_id, ignore_errors=True)

    # Start training the model
    model = YOLO(model=Path(dataset_path, model_name))
    num_epochs = model_config.get("epochs")
    num_workers = model_config.get("workers")
    imgsz = model_config.get("imgsz")

    logger.info(f"Training model: {model_name}")
    model.train(
        data=Path(dataset_path, "data.yaml"),
        epochs=num_epochs,
        device="cuda",
        workers=num_workers,
        imgsz=imgsz,
        project=Path(dataset_path, "runs", "segment"),
    )
    logger.info("Training completed")

    # Delete the temporary "yolo11n.pt" file created by YOLO
    os.remove(Path(Config.get("ROOT_DIR"), "yolo11n.pt"))

    # Save the best model weights
    best_weights_path = Path(dataset_path, "runs/segment/train/weights/best.pt")
    save_path = Path(
        Config.get("global_assets_dir"),
        "trained_models",
        "vehicle_detection",
        "vehicle_detection.pt",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if best_weights_path.exists():
        logger.info("Best model found, copying to assets/trained_models")
        shutil.copy(best_weights_path, save_path)
    else:
        logger.error("Best model weights not found after training.")
        raise FileNotFoundError("Best model weights not found after training.")

    logger.info(f"Model saved to '{save_path}'!")
    logger.info("Loading best model for export...")
    model = YOLO(save_path, task="segment")
    export_model_to_trt(model)


if __name__ == "__main__":
    train()
