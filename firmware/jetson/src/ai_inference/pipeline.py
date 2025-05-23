import queue
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, final

import torch

from shared_src.common import Final, StoppableThread
from shared_src.network.server_client import ServerClient

from .core import logger
from .yolo_inference import YOLOInference


class Model(ABC):
    """
    A base class for models used in the pipeline.
    This class provides a common interface for all models.
    """

    def __init__(self, model_path: Path):
        self._loaded = False
        self._model_path = model_path
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

        self._load()
        self._loaded = True
        logger.debug(f"Model loaded from {self._model_path}")

    @property
    def loaded(self) -> bool:
        """
        Check if the model is loaded.
        This property returns True if the model is successfully loaded, False otherwise.
        """
        return self._loaded

    @property
    def model_path(self) -> Path:
        """
        Get the path of the model file.
        This property returns the path of the model file used to load the model.
        """
        return self._model_path

    @final
    def __call__(self, *data: Any):
        """
        Call the model with input data.
        This method allows the model to be called like a function.
        """
        return self.infer(data)

    @abstractmethod
    def _load(self):
        """
        Load the model from the specified path.
        This method should be overridden by subclasses to implement specific loading logic.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def infer(self, *data: Any):
        """
        Perform inference on the input data.
        This method should be overridden by subclasses to implement specific inference logic.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def dispose(self):
        """
        Dispose of the model resources.
        This method should be overridden by subclasses to implement specific disposal logic.
        """
        raise NotImplementedError("Subclasses must implement this method")


class ModelPipeline(StoppableThread, metaclass=Final):
    """
    A class that manages a pipeline of models for inference.
    This class allows for the sequential processing of data through multiple models.
    """

    __pipeline_buffer: queue.Queue[torch.Tensor] = queue.Queue()

    def __init__(
        self, models: list[Model], server: ServerClient, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._disposed = False
        self.__models = models
        self._server = server
        if not self.__models:
            raise ValueError("Model list cannot be empty")
        if not all(isinstance(model, Model) for model in self.__models):
            raise TypeError("All models must be instances of the Model class")

        logger.debug(f"Pipeline initialized with {len(self.__models)} models")

    @property
    def models(self) -> list[Model]:
        """
        Get the list of models in the pipeline.
        This property allows access to the models for inspection or modification.
        """
        return self.__models

    def __call__(self, *data: Any):
        """
        Call the pipeline with input data.
        This method allows the pipeline to be called like a function.
        """
        return self.input(*data)

    def input(self, *data: Any):
        """
        Process the input data through the pipeline of models.
        This method is called when new data is available for inference.
        """
        for model in self.__models:
            output = model(*data)
        self.__pipeline_buffer.put(output)
        return output

    def run_with_exception_handling(self) -> None:
        try:
            while self.running:
                if not self.__pipeline_buffer.empty():
                    optimal_lane_ids = self.__pipeline_buffer.get()
                    vehicle_lanes = YOLOInference._last_infer_cache
                    from_to_map = zip(vehicle_lanes, optimal_lane_ids)
                    for from_lane, to_lane in from_to_map:
                        if from_lane != to_lane:
                            self._server.send("switch", (from_lane, to_lane))

                    self.__pipeline_buffer.task_done()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise  # Propagate the error for reconnect logic

        finally:
            self.dispose()

    def dispose(self):
        """
        Dispose of the models in the pipeline.
        This method is called to clean up resources when the pipeline is no longer needed.
        """
        if self._disposed:
            return
        self._disposed = True
        self.stop()
        for model in self.__models:
            model.dispose()
        logger.debug("Pipeline disposed")
