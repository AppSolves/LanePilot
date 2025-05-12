import os
from enum import Enum
from pathlib import Path

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

    # Check if the dataset path is a directory
    if not isinstance(class_id, int):
        logger.error(f"Invalid class ID: {class_id}")
        raise ValueError("Class ID must be an integer.")

    def _process_file(file_path: Path) -> None:
        logger.debug(f"Processing file: '{file_path}'")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            logger.error(f"File is empty: '{file_path}'. No labels found.")
            raise ValueError(f"File is empty: '{file_path}'. No labels found.")

        new_lines = []
        for line in lines:
            if line.startswith(str(class_id)):
                parts = line.split()
                if len(parts) != 5:
                    logger.error(f"Invalid line format: {line}")
                    raise ValueError(
                        "Line must contain five elements. Is the class really a box object?"
                    )

                box = tuple(map(float, parts[1:]))
                polygon = class_id, *box_to_polygon(box, BoxShape.XCYCWH)
                new_lines.append(" ".join(map(str, polygon)) + "\n")
            else:
                new_lines.append(line)

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        logger.debug(f"Finished processing file: '{file_path}'")

    logger.debug(f"Converting class with ID '{class_id}' to segmentation element")
    dataset_subdirs = ["train", "valid", "test"]

    # Convert all labels in the specified subdirectories
    for subdir in dataset_subdirs:
        subdir_path = Path(dataset_path, subdir, "labels")
        if not subdir_path.is_dir():
            logger.error(
                f"Subdirectory '{subdir}' does not exist in dataset path: {dataset_path}"
            )
            raise ValueError(f"Subdirectory '{subdir}' does not exist in dataset path.")

        for file in os.listdir(subdir_path):
            file_path = Path(subdir_path, file)
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
