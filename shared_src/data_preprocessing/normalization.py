from enum import Enum

import torch


class NormalizationMode(Enum):
    """
    Enum for normalization modes.
    """

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"


def normalize_data(data: torch.Tensor, mode: NormalizationMode) -> torch.Tensor:
    """
    Normalize the input data based on the specified normalization mode.

    :param data: Input data as a PyTorch tensor.
    :param mode: Normalization mode (min-max or z-score).
    :return: Normalized data as a PyTorch tensor.
    """
    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data must be a PyTorch tensor.")

    match mode:
        case NormalizationMode.MIN_MAX:
            min_val = data.min()
            max_val = data.max()
            normalized_data = (data - min_val) / (max_val - min_val)
        case NormalizationMode.Z_SCORE:
            mean = data.mean()
            std = data.std()
            if std == 0:
                std = 1.0
            normalized_data = (data - mean) / std
        case _:
            raise ValueError(f"Unsupported normalization mode: {mode}")

    return normalized_data
