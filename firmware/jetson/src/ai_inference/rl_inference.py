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
    on observation vectors for multi-vehicle centralized traffic control.

    Expected input: Global observation vector of shape [81] containing:
        - 15 vehicles * 5 features = 75 dims: [lane, speed, acc, front_gap, rear_gap] per vehicle
        - 6 global features: [lane_densities (3), lane_avg_speeds (3)]

    Output: Array of actions for all vehicles, shape [15] where each action is:
        0 = keep_lane, 1 = change_left, 2 = change_right
    """

    def __init__(
        self,
        model_path: Path,
        enable_host_code: bool = False,
        max_vehicles: int = 15,
    ):
        self.enable_host_code = enable_host_code
        self.max_vehicles = max_vehicles
        self.obs_dim = max_vehicles * 5 + 6  # 81 for 15 vehicles
        self.action_dim = max_vehicles * 3  # 45 logits (15 vehicles * 3 actions)
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

    def infer(self, *data: Any) -> np.ndarray:
        """
        Perform inference using the TensorRT engine.

        Args:
            *data: Variable arguments. Expects single observation vector as first argument.
                   observation: np.ndarray or torch.Tensor of shape [81] containing:
                       - 75 vehicle features (15 vehicles * 5 features each)
                       - 6 global features (lane densities + avg speeds)

        Returns:
            np.ndarray: Array of predicted actions, shape [15], one action per vehicle.
                       Each action is: 0=keep_lane, 1=change_left, 2=change_right
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

        # Ensure correct shape: [1, 81] for batch inference
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Move to GPU
        obs_tensor = obs_tensor.to(self.device, non_blocking=True)

        # Prepare output tensor [1, 45] for action logits (15 vehicles * 3 actions)
        output_tensor = torch.empty(
            (1, self.action_dim), dtype=torch.float32, device=self.device
        )

        # Bindings: device pointers
        bindings = [
            obs_tensor.data_ptr(),
            output_tensor.data_ptr(),
        ]

        # Set input shape
        self.context.set_input_shape("observation", obs_tensor.shape)

        # Execute inference
        self.context.execute_v2(bindings)

        # Reshape logits to [15, 3] and get argmax per vehicle
        logits = output_tensor.view(self.max_vehicles, 3)
        actions = logits.argmax(dim=1).cpu().numpy()

        return actions

    def _check_inputs(self, observation: np.ndarray | torch.Tensor) -> bool:
        """
        Check the input observation for validity.

        Args:
            observation: Observation vector, expected shape [81] or [1, 81] for multi-vehicle.

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

        # Check shape: should be [obs_dim] or [1, obs_dim]
        if obs_array.ndim == 1:
            if obs_array.shape[0] != self.obs_dim:
                logger.error(
                    f"Expected observation shape [{self.obs_dim}], got {obs_array.shape}"
                )
                raise ValueError(
                    f"Expected observation shape [{self.obs_dim}], got {obs_array.shape}"
                )
        elif obs_array.ndim == 2:
            if obs_array.shape != (1, self.obs_dim):
                logger.error(
                    f"Expected observation shape [1, {self.obs_dim}], got {obs_array.shape}"
                )
                raise ValueError(
                    f"Expected observation shape [1, {self.obs_dim}], got {obs_array.shape}"
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
