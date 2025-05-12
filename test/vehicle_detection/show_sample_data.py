import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from shared_src.common import Config

color_map = {
    0: (0, 255, 0),  # Green
    1: (0, 255, 255),  # Cyan
    2: (255, 0, 0),  # Red
    3: (0, 0, 255),  # Dark Blue
    4: (255, 165, 0),  # Orange
}


def read_image(image_path: Path):
    """Read an image from a file path using cv2"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be read.")
    return image


def read_label(label_path: Path):
    """Read a label file and return the bounding boxes and class IDs"""
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    masks = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        masks.append((class_id, [float(x) for x in parts[1:]]))
    return masks


def draw_masks(image, masks):
    """Draw masks on the image"""
    height, width, _ = image.shape  # Get image dimensions
    for mask in masks:
        class_id, coords = mask
        # Scale the normalized coordinates to pixel values
        points = np.array(coords).reshape(-1, 2) * [width, height]
        points = points.astype(np.int32)

        image = cv2.polylines(
            image, [points], isClosed=True, color=color_map[class_id], thickness=2
        )
    return image


def main(image_folder: Path, label_folder: Path, num_samples: int = 5):
    """Main function to read images and labels, and draw masks"""
    image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]
    for image_file in image_files[:num_samples]:
        image_path = Path(image_folder, image_file)
        label_path = Path(label_folder, os.path.splitext(image_file)[0] + ".txt")

        # Read the image
        image = read_image(image_path)

        # Read the labels
        masks = read_label(label_path)

        # Draw the masks on the image
        masked_image = draw_masks(image, masks)

        # Show the final result
        cv2.imshow("Masked Image", cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show sample data with masks")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to show",
    )
    args = parser.parse_args()

    samples_dir = Path(Config.get("global_cache_dir"), "vehicle_detection", "train")
    image_folder = Path(samples_dir, "images")
    label_folder = Path(samples_dir, "labels")
    num_samples = args.num_samples

    # Check if the provided paths exist
    if not image_folder.is_dir():
        raise ValueError(f"Image folder {image_folder} does not exist.")
    if not label_folder.is_dir():
        raise ValueError(f"Label folder {label_folder} does not exist.")

    main(image_folder, label_folder, num_samples)
