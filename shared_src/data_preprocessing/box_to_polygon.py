import shutil
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from shapely import Polygon
from ultralytics.engine.results import Results

from .core import logger


class BoxShape(Enum):
    """Enum to represent different box shapes."""

    XYXY = "xyxy"
    XCYCWH = "xcycwh"


def box_to_polygon(box: tuple[float, ...], box_shape: BoxShape) -> tuple[float, ...]:
    """Convert a bounding box to a polygon."""

    # Check if the box is a tuple of the expected format
    if len(box) != 4:
        logger.error(f"Invalid box format: {box}")
        raise ValueError(
            "Box must be a list of four floats. Is the passed object really a box?"
        )

    # Unpack the box tuple and calculate the polygon coordinates
    match box_shape:
        case BoxShape.XYXY:
            x_min, y_min, x_max, y_max = box

        case BoxShape.XCYCWH:
            x_center, y_center, width, height = box
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2

        case _:
            logger.error(f"Invalid box shape: {box_shape}")
            raise ValueError("Box shape must be either 'xyxy' or 'xcycwh'.")

    polygon = (x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max)
    return polygon


def convert_class_to_segment(
    dataset_path: Path, class_id: int, ignore_errors: bool = False
) -> None:
    """Convert a bounding box class to a segmentation element."""

    # Validate dataset_path
    if not dataset_path.is_dir():
        logger.error(
            f"Dataset path does not exist or is not a directory: {dataset_path}"
        )
        raise ValueError("Dataset path must be an existing directory.")

    # Validate class_id type once
    if not isinstance(class_id, int):
        logger.error(f"Invalid class ID: {class_id}")
        raise ValueError("Class ID must be an integer.")

    class_id_str = str(class_id)

    def _process_file(file_path: Path) -> None:
        logger.debug(f"Processing file: '{file_path}'")

        # Use a temporary file for atomic write
        with (
            open(file_path, "r", encoding="utf-8") as f,
            NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp,
        ):
            lines_processed = 0
            for line in f:
                if line.startswith(class_id_str):
                    parts = line.split()
                    if len(parts) != 5:
                        logger.error(f"Invalid line format: {line.strip()}")
                        raise ValueError(
                            "Line must contain five elements. Is the class really a box object?"
                        )
                    box = tuple(map(float, parts[1:]))
                    polygon = (class_id,) + box_to_polygon(box, BoxShape.XCYCWH)
                    tmp.write(" ".join(map(str, polygon)) + "\n")
                else:
                    tmp.write(line)
                lines_processed += 1

            if lines_processed == 0:
                logger.error(f"File is empty: '{file_path}'. No labels found.")
                raise ValueError(f"File is empty: '{file_path}'. No labels found.")

        # Atomically replace original file
        shutil.move(tmp.name, file_path)

        logger.debug(f"Finished processing file: '{file_path}'")

    logger.debug(f"Converting class with ID '{class_id}' to segmentation element")

    dataset_subdirs = ("train", "valid", "test")

    for subdir in dataset_subdirs:
        subdir_path = dataset_path / subdir / "labels"
        if not subdir_path.is_dir():
            logger.error(
                f"Subdirectory '{subdir}' does not exist in dataset path: {dataset_path}"
            )
            raise ValueError(f"Subdirectory '{subdir}' does not exist in dataset path.")

        for file_path in subdir_path.iterdir():
            if file_path.is_file() and file_path.suffix == ".txt":
                try:
                    _process_file(file_path)
                except Exception as e:
                    logger.error(f"Error processing file '{file_path}': {e}")
                    if not ignore_errors:
                        raise

    logger.debug(
        f"Finished converting class with ID '{class_id}' to segmentation element"
    )


def parse_lane_polygons(result: Results, yolo_start_idx: int = 2) -> dict[int, Polygon]:
    """Parse lane polygons from a YOLO tracking data."""

    def _to_shapely(polygon: torch.Tensor) -> Polygon:
        """Convert a tensor polygon to a Shapely Polygon."""
        if polygon.dim() != 2 or polygon.size(1) != 2:
            logger.error(f"Invalid polygon shape: {polygon.shape}")
            raise ValueError("Polygon must be a 2D tensor with shape [n, 2].")
        return Polygon(polygon.cpu().numpy())

    if not result.masks or not result.boxes or not result.boxes.is_track:
        logger.warning("YOLO: No tracking information available")
        return {}

    return {
        i - yolo_start_idx: _to_shapely(result.masks.data[i])
        for i, cls in enumerate(result.boxes.cls)
        if int(cls) >= yolo_start_idx
    }
