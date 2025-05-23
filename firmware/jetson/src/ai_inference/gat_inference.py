from pathlib import Path

import tensorrt as trt
import torch

from shared_src.common import StoppableThread, python_to_trt_level
from shared_src.inference import NUM_LANES

from .core import logger


class GATInference(StoppableThread):
    def __init__(
        self,
        engine_path: Path,
        *args,
        enable_host_code: bool = False,
        **kwargs,
    ):
        # Load the TensorRT engine
        self.engine_path = engine_path
        trt_level = python_to_trt_level(logger.level)
        self.logger = trt.Logger(trt.Logger.INFO.__class__(trt_level))
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            runtime.engine_host_code_allowed = enable_host_code
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize engine. Check runtime and engine compatibility."
            )

        self.context = self.engine.create_execution_context()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(*args, **kwargs)

        logger.debug(f"Model engine loaded from {self.engine_path}")

    def infer(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Perform inference using the TensorRT engine.

        Args:
            x (torch.Tensor): Input data of shape [num_nodes, num_features].
            edge_index (torch.Tensor): Edge index of shape [2, num_edges].

        Returns:
            torch.Tensor: Output data from the model.
        """
        # Check for empty input
        if x.numel() == 0 or edge_index.numel() == 0:
            logger.error("Input arrays must not be empty.")
            raise ValueError("Input arrays must not be empty.")
        if x.ndim != 2:
            logger.error("x should be of shape [num_nodes, num_features]")
            raise ValueError("x should be of shape [num_nodes, num_features]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            logger.error("edge_index should be of shape [2, num_edges]")
            raise ValueError("edge_index should be of shape [2, num_edges]")

        # Ensure tensors are on the correct device (GPU)
        x_tensor = x.to(self.device, non_blocking=True)
        edge_index_tensor = edge_index.to(self.device, non_blocking=True)
        output_shape = (x_tensor.shape[0], NUM_LANES)
        output_tensor = torch.empty(
            output_shape, dtype=torch.float32, device=self.device
        )

        # Bindings: device pointers
        bindings = [
            x_tensor.data_ptr(),
            edge_index_tensor.data_ptr(),
            output_tensor.data_ptr(),
        ]

        # Set dynamic shapes
        self.context.set_input_shape("x", x_tensor.shape)
        self.context.set_input_shape("edge_index", edge_index_tensor.shape)

        self.context.execute_v2(bindings)

        # Output is already a torch tensor on the correct device
        return output_tensor

    def dispose(self):
        """
        Dispose of the TensorRT context and engine.
        """
        self.stop()
        if self.context:
            del self.context
        if self.engine:
            del self.engine

        logger.debug("Model context and engine disposed.")
