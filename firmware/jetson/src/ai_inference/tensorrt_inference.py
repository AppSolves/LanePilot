from pathlib import Path

import numpy as np
import pycuda.autoinit as _
import pycuda.driver as cuda
import tensorrt as trt

from shared_src.common import StoppableThread, python_to_trt_level
from shared_src.inference import NUM_LANES

from .core import logger


class TensorRTInference(StoppableThread):
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
        self.stream = cuda.Stream()
        super().__init__(*args, **kwargs)

        logger.debug(f"TensorRT engine loaded from {self.engine_path}")

    def infer(self, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """
        Perform inference using the TensorRT engine.

        Args:
            x (np.ndarray): Input data of shape [num_nodes, num_features].
            edge_index (np.ndarray): Edge index of shape [2, num_edges].

        Returns:
            np.ndarray: Output data from the model.
        """
        # Check for empty input
        if x.size == 0 or edge_index.size == 0:
            raise ValueError("Input arrays must not be empty.")

        # Define input and output shapes
        input_shapes = [x.shape, edge_index.shape]
        output_shape = (1, NUM_LANES)

        # Create host buffers
        h_input_x = np.array(x, dtype=np.float32).reshape(input_shapes[0])
        h_input_edge_index = np.array(edge_index, dtype=np.int32).reshape(
            input_shapes[1]
        )
        h_output = np.empty(output_shape, dtype=np.float32)

        # Allocate device memory
        d_input_x = cuda.mem_alloc(h_input_x.nbytes)
        d_input_edge_index = cuda.mem_alloc(h_input_edge_index.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        # Copy data to device
        cuda.memcpy_htod(d_input_x, h_input_x)
        cuda.memcpy_htod(d_input_edge_index, h_input_edge_index)

        # Execute inference
        bindings = [int(d_input_x), int(d_input_edge_index), int(d_output)]

        # Set dynamic shapes
        self.context.set_input_shape("x", h_input_x.shape)
        self.context.set_input_shape("edge_index", h_input_edge_index.shape)

        self.context.execute_v2(bindings)

        # Copy output back to host
        cuda.memcpy_dtoh(h_output, d_output)

        return h_output

    def dispose(self):
        """
        Dispose of the TensorRT context and stream.
        """
        self.stop()
        if self.context:
            del self.context
        if self.stream:
            self.stream.synchronize()
            del self.stream
        if self.engine:
            del self.engine

        logger.debug("TensorRT context and stream disposed.")
