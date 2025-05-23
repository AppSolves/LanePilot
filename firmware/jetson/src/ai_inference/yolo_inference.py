import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from shared_src.common import Config, Singleton
from shared_src.data_preprocessing import BoxShape, box_to_polygon
from shared_src.inference import VehicleState

from .core import logger


class YOLOInference(metaclass=Singleton):
    """
    YOLO inference class for vehicle detection and tracking.
    This class uses the YOLO model to perform inference on input frames and
    maintain the state of detected vehicles.
    """

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.5,
        cleanup_interval: float = 5.0,
        cleanup_timeout: float = 10.0,
    ):
        self.model_path = model_path
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self.confidence = confidence
        self.cache_dir = Path(Config.get("global_cache_dir"), "yolo", "runs", "segment")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(self.model_path, task="segment")
        self.vehicle_states: dict[int | float, VehicleState] = {}
        self.cleanup_interval = cleanup_interval
        self.cleanup_timeout = cleanup_timeout
        self.last_cleanup_time = time.time()

    def clean_vehicle_states(self):
        current_time = time.time()
        if not current_time - self.last_cleanup_time > self.cleanup_interval:
            return

        stale_ids = [
            vehicle_id
            for vehicle_id, state in self.vehicle_states.items()
            if current_time - state.last_updated.timestamp() > self.cleanup_timeout
        ]
        for vehicle_id in stale_ids:
            del self.vehicle_states[vehicle_id]

        self.last_cleanup_time = current_time

    def infer(self, frame: cv2.typing.MatLike):
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
                if id not in self.vehicle_states:
                    self.vehicle_states[id] = VehicleState(
                        vehicle_id=id,
                        lane_id=1,
                        polygon_mask_px=polygon,
                    )
                else:
                    self.vehicle_states[id].update_mask(polygon)
            else:
                logger.warning(f"YOLO: Skipping invalid box format for ID {id}: {box}")

        self.clean_vehicle_states()
        return self.vehicle_states

    def dispose(self):
        """
        Dispose of the YOLO model and cleanup resources.
        """
        if self.model:
            del self.model
        logger.debug("YOLO model disposed.")
        self.vehicle_states.clear()
        logger.debug("Vehicle states cleared.")
