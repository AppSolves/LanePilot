import warnings

from shared_src.common import IS_DEBUG

warnings.filterwarnings("ignore", category=FutureWarning, module="onnxscript")
warnings.filterwarnings("ignore", category=UserWarning, message=".*dynamic_axes.*")

import subprocess as sp
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO

from .core import logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_model_to_onnx(
    model: torch.nn.Module,
    save_path: Path,
    dummy_input: tuple[torch.Tensor, ...],
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
):
    """
    Export a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        save_path (Path): The path to save the exported model.
        dummy_input (tuple[torch.Tensor, ...]): A dummy input for the model.
        input_names (list[str], optional): The names of the input tensors.
        output_names (list[str], optional): The names of the output tensors.
        dynamic_axes (dict[str, dict[int, str]], optional): Dynamic axes for the model.
    """
    model = model.eval().to(DEVICE)

    # Export the model to ONNX format
    logger.debug("Exporting the model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        verbose=IS_DEBUG,
        export_params=True,
        optimize=True,
        dynamo=True,
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    logger.debug(f"ONNX model exported to '{save_path}'!")
    return save_path


def export_model_to_trt(
    model: torch.nn.Module | YOLO,
    save_path: Optional[Path] = None,
    dummy_input: Optional[tuple[torch.Tensor, ...]] = None,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
) -> None:
    """
    Export a PyTorch model to TensorRT format.
    Args:
        model (torch.nn.Module | YOLO): The PyTorch model to export.
        save_path (Path, optional): The path to save the exported model.
        dummy_input (tuple[torch.Tensor, ...], optional): A dummy input for the model.
        input_names (list[str], optional): The names of the input tensors.
        output_names (list[str], optional): The names of the output tensors.
        dynamic_axes (dict[str, dict[int, str]], optional): Dynamic axes for the model.
    """
    if isinstance(model, YOLO):
        model.export(
            format="engine",
            dynamic=True,
            simplify=True,
            device="cuda",
        )
    else:
        if save_path is None:
            raise ValueError("Save path is required for exporting the model.")
        if dummy_input is None:
            raise ValueError("Dummy input is required for exporting the model.")

        onnx_path = save_path.with_suffix(".onnx")
        trt_path = save_path.with_suffix(".engine")

        # Export the model to ONNX first
        save_path = export_model_to_onnx(
            model,
            save_path=onnx_path,
            dummy_input=dummy_input,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        # Properly quote the paths for the trtexec command
        command = (
            f'trtexec --onnx="{save_path.as_posix()}" '
            f'--saveEngine="{trt_path.as_posix()}" '
            f"--fp16"
        )

        # Run the command
        sp.run(command, shell=True, check=True)

        logger.debug(f"TensorRT model exported to '{trt_path}'!")
