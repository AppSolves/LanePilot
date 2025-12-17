from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
import torch

from shared_src.common import python_to_trt_level

from .core import logger
from .pipeline import Model


class RLInference(Model):
    """
    RLInference class for performing lane allocation inference using a TensorRT engine.
    This class loads a TensorRT-optimized PPO policy network and performs inference
    on observation vectors (15-dimensional state representations).

    Expected input: Single observation vector of shape [15] containing:
    [lane, speed, acc, gaps, relative_speeds, lane_densities, lane_avg_speeds]

    Output: Lane change action (0=keep, 1=left, 2=right)
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
        self.logger = trt.Logger(trt.Logger.INFO.__class__(trt_level))  # type: ignore
        trt.init_libnvinfer_plugins(self.logger, "")  # type: ignore

        with open(self._model_path, "rb") as f, trt.Runtime(self.logger) as runtime:  # type: ignore
            runtime.engine_host_code_allowed = self.enable_host_code
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize engine. Check runtime and engine compatibility."
            )

        self.context = self.engine.create_execution_context()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def infer(self, *data: Any) -> int:
        """
        Perform inference using the TensorRT engine.

        Args:
            *data: Variable arguments. Expects single observation vector as first argument.
                   observation: np.ndarray or torch.Tensor of shape [15] containing normalized state features.

        Returns:
            int: Predicted action (0=keep_lane, 1=change_left, 2=change_right)
        """
        if len(data) == 0:
            raise ValueError("Expected at least one argument: observation vector")

        # Extract observation from args (handles both infer(obs) and infer(*data) calls)
        observation = data[0]

        # Validate input
        self._check_inputs(observation)

        # Convert to torch tensor if needed
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.from_numpy(observation).float()
        else:
            obs_tensor = observation.float()

        # Ensure correct shape: [1, 15] for batch inference
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Move to GPU
        obs_tensor = obs_tensor.to(self.device, non_blocking=True)

        # Prepare output tensor [1, 3] for action logits
        output_tensor = torch.empty((1, 3), dtype=torch.float32, device=self.device)

        # Bindings: device pointers
        bindings = [
            obs_tensor.data_ptr(),
            output_tensor.data_ptr(),
        ]

        # Set input shape
        self.context.set_input_shape("observation", obs_tensor.shape)

        # Execute inference
        self.context.execute_v2(bindings)

        # Return action with highest probability
        return int(output_tensor.argmax(dim=1).item())

    @staticmethod
    def _check_inputs(observation: np.ndarray | torch.Tensor) -> bool:
        """
        Check the input observation for validity.

        Args:
            observation: Observation vector, expected shape [15] or [1, 15].

        Raises:
            ValueError: If the observation is not valid.
        """
        # Convert to numpy for shape checking
        if isinstance(observation, torch.Tensor):
            obs_array = observation.cpu().numpy()
        else:
            obs_array = observation

        if obs_array.size == 0:
            logger.error("Observation must not be empty.")
            raise ValueError("Observation must not be empty.")

        # Check shape: should be [15] or [1, 15]
        if obs_array.ndim == 1:
            if obs_array.shape[0] != 15:
                logger.error(f"Expected observation shape [15], got {obs_array.shape}")
                raise ValueError(
                    f"Expected observation shape [15], got {obs_array.shape}"
                )
        elif obs_array.ndim == 2:
            if obs_array.shape != (1, 15):
                logger.error(
                    f"Expected observation shape [1, 15], got {obs_array.shape}"
                )
                raise ValueError(
                    f"Expected observation shape [1, 15], got {obs_array.shape}"
                )
        else:
            logger.error(f"Observation should be 1D or 2D, got {obs_array.ndim}D")
            raise ValueError(f"Observation should be 1D or 2D, got {obs_array.ndim}D")

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
