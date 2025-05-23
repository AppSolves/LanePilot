from shared_src.common.metaclasses import Singleton

from .core import logger


class ModelPipeline(metaclass=Singleton):
    """
    A class that manages a pipeline of models for inference.
    This class allows for the sequential processing of data through multiple models.
    """

    def __init__(self, models: list):
        self.models = models

    def __call__(self, data):
        """
        Call the pipeline with input data.
        This method allows the pipeline to be called like a function.
        """
        return self.on_input(data)

    def on_input(self, data):
        """
        Process the input data through the pipeline of models.
        This method is called when new data is available for inference.
        """
        for model in self.models:
            data = model.infer(data)
        return data

    def dispose(self):
        """
        Dispose of the models in the pipeline.
        This method is called to clean up resources when the pipeline is no longer needed.
        """
        for model in self.models:
            model.dispose()
        logger.debug("Pipeline disposed")
