from pathlib import Path
from typing import Any

import tensorrt as trt
import torch

from shared_src.common import python_to_trt_level
from shared_src.inference import NUM_LANES

from .core import logger
from .pipeline import Model


class GATInference(Model):
    """
    GATInference class for performing inference using a TensorRT engine.
    This class loads a TensorRT engine from a file and performs inference
    on input data using the engine.
    """

    def __init__(self, model_path: Path, enable_host_code: bool = False):
        self.enable_host_code = enable_host_code
        super().__init__(model_path)

    def _load(self):
        """
        Load the TensorRT engine from the specified model path.
        This method initializes the TensorRT runtime and creates an execution context.
        """
        trt_level = python_to_trt_level(logger.level)
        self.logger = trt.Logger(trt.Logger.INFO.__class__(trt_level))
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(self._model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            runtime.engine_host_code_allowed = self.enable_host_code
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize engine. Check runtime and engine compatibility."
            )

        self.context = self.engine.create_execution_context()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def infer(self, *data: Any) -> torch.Tensor:
        """
        Perform inference using the TensorRT engine.

        Args:
            x (torch.Tensor): Input data of shape [num_nodes, num_features].
            edge_index (torch.Tensor): Edge index of shape [2, num_edges].

        Returns:
            torch.Tensor: Output data from the model.
        """
        if len(data) != 2:
            raise ValueError(
                "Expected two inputs: x (node features) and edge_index (edge indices)."
            )
        x, edge_index = data
        if not isinstance(x, torch.Tensor) or not isinstance(edge_index, torch.Tensor):
            raise TypeError("Both x and edge_index must be torch.Tensor objects.")

        # Check for empty input
        self._check_inputs(x, edge_index)

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
        return output_tensor.argmax(dim=1)

    @staticmethod
    def _check_inputs(x: torch.Tensor, edge_index: torch.Tensor) -> bool:
        """
        Check the input tensors for validity.
        Args:
            x (torch.Tensor): Input data of shape [num_nodes, num_features].
            edge_index (torch.Tensor): Edge index of shape [2, num_edges].
        Raises:
            ValueError: If the input tensors are not valid.
        """
        if x.numel() == 0 or edge_index.numel() == 0:
            logger.error("Input arrays must not be empty.")
            raise ValueError("Input arrays must not be empty.")
        if x.ndim != 2:
            logger.error("x should be of shape [num_nodes, num_features]")
            raise ValueError("x should be of shape [num_nodes, num_features]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            logger.error("edge_index should be of shape [2, num_edges]")
            raise ValueError("edge_index should be of shape [2, num_edges]")
        return True

    def dispose(self):
        """
        Dispose of the TensorRT context and engine.
        """
        if self.context:
            del self.context
        if self.engine:
            del self.engine

        logger.info("Model context and engine disposed.")
