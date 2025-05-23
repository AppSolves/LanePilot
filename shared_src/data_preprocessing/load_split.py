from enum import Enum
from pathlib import Path

import torch
from torch_geometric.data import Data

from .core import logger
from .edge_index import build_edge_index


class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


def load_dataset_split(
    dataset_path: Path,
    dataset_split: DatasetSplit,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    max_distance_cm: float = 10.0,
) -> list[Data]:
    dataset = []
    feature_dim = None

    for file in Path(dataset_path, dataset_split.value).iterdir():
        if file.is_file() and file.suffix == ".pt":
            data = torch.load(file)
            x = data["x"]  # Shape: [num_vehicles, feature_dim]
            y = data["y"]  # Shape: [num_vehicles]

            if feature_dim is None:
                feature_dim = x.shape[1]
            elif feature_dim != x.shape[1]:
                logger.error(
                    f"Feature dimension mismatch in file '{file}': expected {feature_dim}, got {x.shape[1]}"
                )
                raise ValueError(f"Feature dimension mismatch in dataset files: {file}")

            edge_index = build_edge_index(x, max_distance=max_distance_cm)
            data_obj = Data(x=x, edge_index=edge_index, y=y).to(device)
            dataset.append(data_obj)

    return dataset
