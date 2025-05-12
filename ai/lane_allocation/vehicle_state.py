import math
from datetime import datetime

import torch

from shared_src.data_preprocessing import NormalizationMode, normalize_data

from .core import MODULE_CONFIG

# Camera and vehicle settings
_camera_settings = MODULE_CONFIG.get("camera", {})
_vehicle_settings = MODULE_CONFIG.get("vehicle", {})
_environment_settings = MODULE_CONFIG.get("environment", {})

CAMERA_FOV_DEG: float = _camera_settings.get("fov_deg")
CAMERA_RESOLUTION: tuple[int, int] = tuple(_camera_settings.get("resolution"))
VEHICLE_HEIGHT_CM: float = _vehicle_settings.get("height_cm")
NUM_LANES: int = _environment_settings.get("num_lanes")
MAX_VEHICLES_PER_LANE: int = _environment_settings.get("max_vehicles_per_lane")
NORMALIZATION_MODE: NormalizationMode = NormalizationMode(
    _environment_settings.get("normalization_mode")
)

LANE_UTILIZATION: dict[int, int] = {}


class VehicleState:
    """
    Class representing the state of a vehicle in the lane allocation system, with 3D-aware speed estimation.
    """

    def __init__(
        self,
        vehicle_id: int,
        lane_id: int,
        polygon_mask_px: tuple[float, ...],
    ) -> None:
        """
        Initialize the VehicleState.

        :param vehicle_id: The ID of the vehicle.
        :param lane_id: The lane the vehicle is in.
        :param polygon_mask_px: The polygon mask (4 points = 8 floats).
        """
        if not isinstance(polygon_mask_px, tuple) or len(polygon_mask_px) != 8:
            raise ValueError(
                f"Polygon mask must be a tuple of 8 float values, got: {polygon_mask_px}"
            )

        self.vehicle_id = vehicle_id
        self.lane_id = lane_id
        self.polygon_mask_px = polygon_mask_px
        self.speed = 0.0
        self.acceleration = 0.0
        self.last_updated = datetime.now()

        LANE_UTILIZATION.setdefault(lane_id, 0)
        LANE_UTILIZATION[lane_id] += 1

        # Save previous state
        self._last_center = self._calculate_center(polygon_mask_px)
        self._last_box_height = self._calculate_box_height(polygon_mask_px)
        self._last_depth = self._estimate_depth(self._last_box_height)

    def _calculate_center(self, mask: tuple[float, ...]) -> tuple[float, float]:
        """Calculate center (x, y) from the polygon mask."""
        x_coords = mask[0::2]
        y_coords = mask[1::2]
        return (sum(x_coords) / 4, sum(y_coords) / 4)

    def _calculate_box_height(self, mask: tuple[float, ...]) -> float:
        """Estimate bounding box height in pixels from top-bottom corners."""
        y_coords = mask[1::2]
        return max(y_coords) - min(y_coords)

    def _estimate_depth(self, box_height_px: float) -> float:
        """
        Estimate depth (Z) using simple pinhole projection:
        Z ∝ H_real / H_pixel * tan(FOV/2)

        Returns a depth value in cm (scaled by constant factor).
        """
        image_height_px = CAMERA_RESOLUTION[1]
        fov_rad = math.radians(CAMERA_FOV_DEG)

        # Prevent division by zero
        if box_height_px == 0:
            return float("inf")

        # Simplified pinhole model:
        focal_length_px = (image_height_px / 2) / math.tan(fov_rad / 2)
        depth = (focal_length_px * VEHICLE_HEIGHT_CM) / box_height_px
        return depth

    def calculate_speed(self, new_mask: tuple[float, ...]) -> float:
        """
        Estimate 3D speed (cm/s) from polygon mask change.

        :param new_mask: New polygon mask of the vehicle.
        :return: Estimated 3D speed in cm/s.
        """
        now = datetime.now()
        time_delta = (now - self.last_updated).total_seconds()
        if time_delta == 0:
            return 0.0

        new_center = self._calculate_center(new_mask)
        new_box_height = self._calculate_box_height(new_mask)
        new_depth = self._estimate_depth(new_box_height)

        # Movement in image space (x, y) in pixels
        dx = new_center[0] - self._last_center[0]
        dy = new_center[1] - self._last_center[1]

        # Lateral distance in pixels
        lateral_dist_px = math.hypot(dx, dy)

        # Approximate scale factor (we assume 1px ≈ X cm at old depth)
        # So we convert image-plane motion to real-world cm using the **old** depth scale
        scale_cm_per_px = VEHICLE_HEIGHT_CM / self._last_box_height
        lateral_dist_cm = lateral_dist_px * scale_cm_per_px

        # Depth movement
        dz = new_depth - self._last_depth  # positive = away from camera
        total_dist_cm = math.hypot(lateral_dist_cm, dz)

        return total_dist_cm / time_delta

    def update_mask(self, new_mask: tuple[float, ...]) -> None:
        """Update vehicle mask and estimate new speed."""
        if not isinstance(new_mask, tuple) or len(new_mask) != 8:
            raise ValueError("Polygon mask must be a tuple of 8 float values.")

        new_speed = self.calculate_speed(new_mask)
        now = datetime.now()

        self.acceleration = (new_speed - self.speed) / (
            now - self.last_updated
        ).total_seconds()
        self.speed = new_speed
        self.last_updated = now
        self.polygon_mask_px = new_mask

        # Update internal state
        self._last_center = self._calculate_center(new_mask)
        self._last_box_height = self._calculate_box_height(new_mask)
        self._last_depth = self._estimate_depth(self._last_box_height)

    @property
    def feature_vector(self) -> torch.Tensor:
        """
        Convert vehicle state to a feature vector for model input.

        :return: Feature vector as a PyTorch tensor.
        """
        raw_features = torch.tensor(
            [
                self.lane_id,
                self.speed,
                self.acceleration,
                self._last_depth,
                *(LANE_UTILIZATION.get(i, 0) for i in range(NUM_LANES)),
            ],
            dtype=torch.float32,
        )

        return normalize_data(raw_features, NORMALIZATION_MODE)

    @property
    def num_features(self) -> int:
        """Number of features in the feature vector."""
        return len(self.feature_vector)

    @property
    def lane_utilization(self) -> int:
        """Get the current lane utilization."""
        return LANE_UTILIZATION[self.lane_id]

    def remove(self) -> None:
        """Remove vehicle from lane utilization."""
        if self.lane_id in LANE_UTILIZATION:
            LANE_UTILIZATION[self.lane_id] -= 1
            if LANE_UTILIZATION[self.lane_id] <= 0:
                del LANE_UTILIZATION[self.lane_id]

    def __del__(self) -> None:
        """Destructor to clean up resources."""
        self.remove()
