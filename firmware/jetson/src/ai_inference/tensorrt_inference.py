from pathlib import Path

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

from shared_src.common import StoppableThread

from .core import logger


class TensorRTInference(StoppableThread):
    def __init__(self, engine_path: Path, *args, **kwargs):
        # Load the TensorRT engine
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        super().__init__(*args, **kwargs)

        logger.debug(f"TensorRT engine loaded from {engine_path}")

    def infer(self, x, edge_index):
        """
        Perform inference using the TensorRT engine.
        Args:
            x (np.ndarray): Input data for the model.
            edge_index (np.ndarray): Edge index data for the model.
        Returns:
            np.ndarray: Output data from the model.
        """
        # Allocate buffers for input and output
        input_shapes = [(1, 5)] * 2
        output_shape = (1, 10)  # Adjust based on your output shape

        # Create host and device buffers
        h_input_src = np.array(src_input, dtype=np.int32).reshape(input_shapes[0])
        h_input_tgt = np.array(tgt_input, dtype=np.int32).reshape(input_shapes[1])
        h_output = np.empty(output_shape, dtype=np.float32)

        d_input_src = cuda.mem_alloc(h_input_src.nbytes)
        d_input_tgt = cuda.mem_alloc(h_input_tgt.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)

        # Copy inputs to device
        cuda.memcpy_htod(d_input_src, h_input_src)
        cuda.memcpy_htod(d_input_tgt, h_input_tgt)

        # Execute inference
        self.context.execute_v2([int(d_input_src), int(d_input_tgt), int(d_output)])

        # Copy output back to host
        cuda.memcpy_dtoh(h_output, d_output)

        return h_output

    def dispose(self):
        """
        Dispose of the TensorRT context and stream.
        """
        if self.context:
            del self.context
        if self.stream:
            self.stream.synchronize()
            del self.stream
        if self.engine:
            del self.engine

        logger.debug("TensorRT context and stream disposed.")
