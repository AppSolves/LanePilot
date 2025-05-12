import subprocess as sp
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO

from .core import logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_model_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor | dict,
    save_path: Path,
):
    """
    Export a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
    """
    model = model.eval().to(DEVICE)

    # Check if the model is a PyTorch model
    if isinstance(model, torch.nn.Module):
        # Convert the PyTorch model to ONNX format
        if dummy_input is None:
            raise ValueError("Dummy input is required for exporting the model.")

        torch.onnx.export(
            model,
            (
                tuple(dummy_input.values())
                if isinstance(dummy_input, dict)
                else dummy_input
            ),
            save_path,
            input_names=(
                list(dummy_input.keys()) if isinstance(dummy_input, dict) else None
            ),
            verbose=False,
        )
    else:
        raise ValueError("Unsupported model type. Only PyTorch models are supported.")

    logger.debug(f"ONNX model exported to '{save_path}'!")


def export_model_to_trt(
    model: torch.nn.Module | YOLO,
    dummy_input: Optional[torch.Tensor | dict] = None,
    save_path: Optional[Path] = None,
):
    if isinstance(model, YOLO):
        model.export(
            format="engine",
            dynamic=True,
            simplify=True,
            device="cuda",
        )
    elif isinstance(model, torch.nn.Module):
        if dummy_input is None:
            raise ValueError("Dummy input is required for exporting the model.")
        if save_path is None:
            raise ValueError("Save path is required for exporting the model.")

        onnx_path = save_path.with_suffix(".onnx")
        trt_path = save_path.with_suffix(".trt")

        # Export the model to ONNX first
        export_model_to_onnx(
            model,
            dummy_input,
            save_path=onnx_path,
        )

        # Properly quote the paths for the trtexec command
        command = (
            f'trtexec --onnx="{onnx_path.as_posix()}" '
            f'--saveEngine="{trt_path.as_posix()}" '
            f"--inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"
        )

        # Run the command
        sp.run(command, shell=True, check=True)

        logger.debug(f"TensorRT model exported to '{trt_path}'!")
    else:
        raise ValueError(
            "Unsupported model type. Only PyTorch and YOLO models are supported."
        )
