import argparse
import time

import cv2
import numpy as np
import torch
from ultralytics.models import YOLO

from ai.vehicle_detection.core import Path, logger
from firmware.jetson.src.ai_inference.rl_inference import RLInference
from shared_src.common import Config
from shared_src.data_preprocessing import BoxShape, box_to_polygon, parse_lane_polygons
from shared_src.inference import LANE_POLYGONS, NUM_LANES, VehicleState

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR: Path = Path(str(Config.get("global_cache_dir"))) / "vehicle_detection"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def clean_vehicle_states(
    vehicle_states: dict[int | float, VehicleState], timeout: float
):
    current_time = time.time()
    stale_ids = [
        vehicle_id
        for vehicle_id, state in vehicle_states.items()
        if current_time - state.last_updated.timestamp() > timeout
    ]
    for vehicle_id in stale_ids:
        del vehicle_states[vehicle_id]


def compute_observation(
    ego_id: int,
    vehicle_states: dict[int | float, VehicleState],
    num_lanes: int = 3,
) -> np.ndarray:
    """
    Compute 15-dimensional observation for RL model.

    Returns: [lane, speed, acc, front_gap, rear_gap, front_rel_speed, rear_rel_speed,
              left_front_gap, left_rear_gap, right_front_gap, right_rear_gap,
              lane_density_0, lane_density_1, lane_density_2]
    """
    ego = vehicle_states[ego_id]
    ego_lane = ego.lane_id
    ego_center_y = ego._last_center[1]

    front_gap = 100.0
    rear_gap = 100.0
    front_rel_speed = 0.0
    rear_rel_speed = 0.0
    left_front_gap = 100.0
    left_rear_gap = 100.0
    right_front_gap = 100.0
    right_rear_gap = 100.0

    same_lane_vehicles = [
        v
        for v in vehicle_states.values()
        if v.lane_id == ego_lane and v.vehicle_id != ego_id
    ]
    if same_lane_vehicles:

        front_vehicles = [
            v for v in same_lane_vehicles if v._last_center[1] < ego_center_y
        ]
        rear_vehicles = [
            v for v in same_lane_vehicles if v._last_center[1] > ego_center_y
        ]

        if front_vehicles:
            front_vehicle = min(
                front_vehicles, key=lambda v: ego_center_y - v._last_center[1]
            )
            front_gap = max(1.0, ego_center_y - front_vehicle._last_center[1])
            front_rel_speed = ego.speed - front_vehicle.speed

        if rear_vehicles:
            rear_vehicle = min(
                rear_vehicles, key=lambda v: v._last_center[1] - ego_center_y
            )
            rear_gap = max(1.0, rear_vehicle._last_center[1] - ego_center_y)
            rear_rel_speed = ego.speed - rear_vehicle.speed

    if ego_lane > 0:
        left_lane_vehicles = [
            v for v in vehicle_states.values() if v.lane_id == ego_lane - 1
        ]
        front_left = [v for v in left_lane_vehicles if v._last_center[1] < ego_center_y]
        rear_left = [v for v in left_lane_vehicles if v._last_center[1] > ego_center_y]

        if front_left:
            left_front_gap = max(
                1.0,
                ego_center_y
                - min(
                    front_left, key=lambda v: ego_center_y - v._last_center[1]
                )._last_center[1],
            )
        if rear_left:
            left_rear_gap = max(
                1.0,
                min(
                    rear_left, key=lambda v: v._last_center[1] - ego_center_y
                )._last_center[1]
                - ego_center_y,
            )

    if ego_lane < num_lanes - 1:
        right_lane_vehicles = [
            v for v in vehicle_states.values() if v.lane_id == ego_lane + 1
        ]
        front_right = [
            v for v in right_lane_vehicles if v._last_center[1] < ego_center_y
        ]
        rear_right = [
            v for v in right_lane_vehicles if v._last_center[1] > ego_center_y
        ]

        if front_right:
            right_front_gap = max(
                1.0,
                ego_center_y
                - min(
                    front_right, key=lambda v: ego_center_y - v._last_center[1]
                )._last_center[1],
            )
        if rear_right:
            right_rear_gap = max(
                1.0,
                min(
                    rear_right, key=lambda v: v._last_center[1] - ego_center_y
                )._last_center[1]
                - ego_center_y,
            )

    lane_densities = []
    for lane in range(num_lanes):
        count = sum(1 for v in vehicle_states.values() if v.lane_id == lane)
        lane_densities.append(min(1.0, count / 10.0))

    obs = np.array(
        [
            ego_lane / float(num_lanes - 1),
            ego.speed / 30.0,
            np.clip(ego.acceleration / 5.0, -1.0, 1.0),
            np.clip(front_gap / 100.0, 0.0, 1.0),
            np.clip(rear_gap / 100.0, 0.0, 1.0),
            np.clip(front_rel_speed / 20.0, -1.0, 1.0),
            np.clip(rear_rel_speed / 20.0, -1.0, 1.0),
            np.clip(left_front_gap / 100.0, 0.0, 1.0),
            np.clip(left_rear_gap / 100.0, 0.0, 1.0),
            np.clip(right_front_gap / 100.0, 0.0, 1.0),
            np.clip(right_rear_gap / 100.0, 0.0, 1.0),
            *lane_densities,
        ],
        dtype=np.float32,
    )

    return obs


def main(PORT: int = 8000, confidence: float = 0.5) -> None:
    global LANE_POLYGONS
    pipeline = f"srtsrc uri=srt://0.0.0.0:{PORT}?mode=listener&latency=1 ! queue ! tsdemux ! h264parse ! nvh264dec ! videoconvert ! appsink sync=false"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        logger.error("Failed to open video stream")
        return

    yolo_model_path = (
        Path(str(Config.get("global_assets_dir")))
        / "trained_models"
        / "vehicle_detection"
        / "vehicle_detection.engine"
    )
    if not yolo_model_path.is_file():
        logger.error(f"Model not found at {yolo_model_path}")
        return

    logger.debug(f"Model path: {yolo_model_path}")

    yolo_model = YOLO(yolo_model_path, task="segment")

    rl_model_path = (
        Path(str(Config.get("global_assets_dir")))
        / "trained_models"
        / "lane_allocation"
        / "lane_allocation_rl.zip"
    )
    if not rl_model_path.is_file():
        logger.error(f"RL model not found at {rl_model_path}")
        logger.info("Train the model first: python -m ai.lane_allocation.train")
        return

    try:
        rl_inference = RLInference(model_path=rl_model_path)
        logger.info(f"RL model loaded from {rl_model_path}")
    except Exception as e:
        logger.error(f"Failed to load RL model: {e}")
        return

    logger.debug(f"Tracking video stream on PORT {PORT}")

    vehicle_states: dict[int | float, VehicleState] = {}
    last_cleanup_time = time.time()
    cleanup_interval = 5
    update_timeout = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("No frame received")
            continue

        result = yolo_model.track(
            frame,
            conf=confidence,
            persist=True,
            project=CACHE_DIR / "runs" / "segment",
        )[0]

        if not LANE_POLYGONS:
            LANE_POLYGONS = parse_lane_polygons(result)

        boxes = result.boxes
        if not boxes or not boxes.is_track:
            logger.warning("YOLO: No tracking information available")
            continue

        coords = boxes.xyxy
        ids = boxes.id

        if ids is None or coords is None:
            logger.warning("YOLO: No vehicle IDs or coordinates detected")
            return

        annotated_frame = frame.copy()
        for id, box in zip(ids, coords):
            if box is not None and len(box) > 0:
                if len(box) == 4:
                    id = int(id)
                    polygon = box_to_polygon(box, BoxShape.XYXY)
                    if id not in vehicle_states:
                        LANE_POLYGONS = {}
                        vehicle_states[id] = VehicleState(
                            vehicle_id=id,
                            polygon_mask_px=polygon,
                        )
                    else:
                        vehicle_states[id].update_mask(polygon)

                    current_lane = vehicle_states[id].lane_id

                    try:

                        obs = compute_observation(id, vehicle_states, NUM_LANES)

                        action = rl_inference.infer(obs)

                        if action == 1 and current_lane > 0:
                            optimal_lane_id = current_lane - 1
                        elif action == 2 and current_lane < NUM_LANES - 1:
                            optimal_lane_id = current_lane + 1
                        else:
                            optimal_lane_id = current_lane
                    except Exception as e:
                        logger.error(f"Error during RL lane allocation: {e}")
                        logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
                        optimal_lane_id = current_lane

                    annotated_frame = cv2.rectangle(
                        annotated_frame,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 255, 0),
                        2,
                    )
                    annotated_frame = cv2.putText(
                        annotated_frame,
                        f"ID: {id} | Speed: {vehicle_states[id].speed:.1f} cm/s | Optimal Lane: {optimal_lane_id}",
                        (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )
                else:
                    logger.warning(
                        f"Invalid mask format, skipping this mask: {box.shape}"
                    )

        current_time = time.time()
        if current_time - last_cleanup_time > cleanup_interval:
            clean_vehicle_states(vehicle_states, update_timeout)
            last_cleanup_time = current_time

        cv2.imshow("Vehicle Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.debug("Video stream closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Tracking Model")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the video stream (default: 8000)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detection (default: 0.5)",
    )
    args = parser.parse_args()

    main(args.port, args.confidence)
    logger.debug("Starting vehicle tracking model")
