import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from ultralytics import YOLO

from shared_src.common import Config
from shared_src.data_preprocessing import BoxShape, box_to_polygon, build_edge_index
from shared_src.inference import MODULE_CONFIG as VEHICLE_CONFIG
from shared_src.inference import VehicleState

from .core import logger
from .gat_inference import GATInference
from .pipeline import Model


class YOLOInference(Model):
    """
    YOLO inference class for vehicle detection and tracking.
    This class uses the YOLO model to perform inference on input frames and
    maintain the state of detected vehicles.
    """

    _last_infer_cache: tuple[int, ...] = ()
    _vehicle_states: dict[int | float, VehicleState] = {}

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.5,
        cleanup_interval: float = 5.0,
        cleanup_timeout: float = 10.0,
        return_tensors: bool = False,
    ):
        self.return_tensors = return_tensors
        self.confidence = confidence
        self.cache_dir = Path(Config.get("global_cache_dir"), "yolo", "runs", "segment")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_interval = cleanup_interval
        self.cleanup_timeout = cleanup_timeout
        self.last_cleanup_time = time.time()
        self.vehicle_config = VEHICLE_CONFIG.get("vehicle", {})
        self._tensor_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        super().__init__(model_path)

    def _load(self):
        """
        Load the YOLO model from the specified path.
        This method initializes the YOLO model and sets it to evaluation mode.
        """
        self.model = YOLO(self.model_path, task="segment")
        self.model.eval()

    def clean_vehicle_states(self):
        current_time = time.time()
        if not current_time - self.last_cleanup_time > self.cleanup_interval:
            return

        stale_ids = [
            vehicle_id
            for vehicle_id, state in self._vehicle_states.items()
            if current_time - state.last_updated.timestamp() > self.cleanup_timeout
        ]
        for vehicle_id in stale_ids:
            del self._vehicle_states[vehicle_id]

        self.last_cleanup_time = current_time
        logger.debug(f"YOLO: Cleaned up vehicle states: {stale_ids}")

    def infer(
        self, *data: Any
    ) -> Optional[dict[int | float, VehicleState] | tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform inference on the input frame using the YOLO model.
        Args:
            frame (np.ndarray): The input frame (OpenCV image).
        Returns:
            dict: Dictionary containing vehicle states with vehicle IDs as keys.
            tuple: Tuple containing the feature vectors and edge index if return_tensors is True.
        """
        if len(data) != 1:
            raise ValueError("Expected one input: frame (np.ndarray)")
        frame = data[0]
        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a numpy ndarray (OpenCV image).")

        # Perform inference
        result = self.model.track(
            frame,
            conf=self.confidence,
            persist=True,
            project=self.cache_dir,
        )[0]

        boxes = result.boxes
        if not boxes or not boxes.is_track:
            logger.warning("YOLO: No tracking information available")
            return

        coords = boxes.xyxy
        ids = boxes.id

        for id, box in zip(ids, coords):
            if box is not None and len(box) == 4:
                id = int(id)
                polygon = box_to_polygon(box, BoxShape.XYXY)
                if id not in self._vehicle_states:
                    self._vehicle_states[id] = VehicleState(
                        vehicle_id=id,
                        lane_id=1,
                        polygon_mask_px=polygon,
                    )
                else:
                    self._vehicle_states[id].update_mask(polygon)
            else:
                logger.warning(f"YOLO: Skipping invalid box format for ID {id}: {box}")

        self.clean_vehicle_states()

        self._last_infer_cache = tuple(
            vehicle.lane_id for vehicle in self._vehicle_states.values()
        )
        if self.return_tensors:
            return self._to_tensor()
        return self._vehicle_states

    def _to_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert vehicle states to tensors for model input.
        Returns:
            tuple: Tuple containing the feature vectors and edge index.
        """
        # Maintain a list of the vehicles' feature vectors
        ids, states = zip(*self._vehicle_states.items())
        id_hash = hash(ids)

        # Build edge index if needed
        if id_hash in self._tensor_cache:
            x, edge_index = self._tensor_cache[id_hash]
            return x, edge_index
        else:
            feature_vectors = [state.feature_vector for state in states]

            # Stack the feature vectors into a tensor
            x = torch.stack(feature_vectors).to(self.model.device)

            # Build edge index
            edge_index = build_edge_index(
                x,
                max_distance=self.vehicle_config.get("max_distance_cm", 10),
            )

            # Ensure input validity
            assert GATInference._check_inputs(x, edge_index)

            self._tensor_cache[id_hash] = (x, edge_index)
            return x, edge_index

    def dispose(self):
        """
        Dispose of the YOLO model and cleanup resources.
        """
        if self.model:
            del self.model
        self._vehicle_states.clear()
        self._tensor_cache.clear()

        logger.info("Model context and engine disposed.")
