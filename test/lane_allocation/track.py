import argparse
import time

import cv2
import torch
from ultralytics import YOLO

from ai.lane_allocation import MODULE_CONFIG as GAT_CONFIG
from ai.lane_allocation import LaneAllocationGAT
from ai.vehicle_detection.core import Path, logger
from shared_src.common import Config
from shared_src.data_preprocessing import BoxShape, box_to_polygon, build_edge_index
from shared_src.inference import NUM_LANES, VehicleState

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR: Path = Path(Config.get("global_cache_dir"), "vehicle_detection")
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


def main(PORT: int = 8000, confidence: float = 0.5) -> None:
    pipeline = f"srtsrc uri=srt://0.0.0.0:{PORT}?mode=listener&latency=1 ! queue ! tsdemux ! h264parse ! nvh264dec ! videoconvert ! appsink sync=false"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        logger.error("Failed to open video stream")
        return

    yolo_model_path = Path(
        Config.get("global_assets_dir"),
        "trained_models",
        "vehicle_detection",
        "vehicle_detection.engine",
    )
    if not yolo_model_path.is_file():
        logger.error(f"Model not found at {yolo_model_path}")
        return

    logger.debug(f"Model path: {yolo_model_path}")

    # Load the models
    # YOLO
    yolo_model = YOLO(yolo_model_path, task="segment")

    # GAT
    gat_config = GAT_CONFIG.get("model", {})
    num_heads = gat_config.get("num_heads")
    hidden_dim = gat_config.get("hidden_dim")
    input_dim = 4 + NUM_LANES

    max_distance_cm = GAT_CONFIG.get("vehicle", {}).get("max_distance_cm")

    gat_model = LaneAllocationGAT(
        input_dim=input_dim, hidden_dim=hidden_dim, heads=num_heads
    )
    gat_model.inference(
        model_path=Path(
            Config.get("global_assets_dir"),
            "trained_models",
            "lane_allocation",
            "lane_allocation.pt",
        ),
        device=DEVICE,
    )
    logger.debug(f"Tracking video stream on PORT {PORT}")

    vehicle_states: dict[int | float, VehicleState] = {}
    last_cleanup_time = time.time()
    cleanup_interval = 5  # Perform cleanup every 5 seconds
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
            project=Path(CACHE_DIR, "runs", "segment"),
        )[0]
        boxes = result.boxes
        if not boxes or not boxes.is_track:
            logger.warning("No tracking information available")
            continue

        coords = boxes.xyxy
        ids = boxes.id

        annotated_frame = frame.copy()
        for id, box in zip(ids, coords):
            if box is not None and len(box) > 0:
                if len(box) == 4:
                    id = int(id)
                    polygon = box_to_polygon(box, BoxShape.XYXY)
                    if id not in vehicle_states:
                        vehicle_states[id] = VehicleState(
                            vehicle_id=id,
                            lane_id=1,
                            polygon_mask_px=polygon,
                        )
                    else:
                        vehicle_states[id].update_mask(polygon)

                    # LANE ALLOCATION

                    # Maintain a list of vehicle IDs and their corresponding feature vectors
                    vehicle_ids = list(vehicle_states.keys())
                    feature_vectors = [
                        vehicle_states[id].feature_vector for id in vehicle_ids
                    ]

                    # Stack the feature vectors into a tensor
                    x = torch.stack(feature_vectors).to(DEVICE)

                    # Build edge index
                    edge_index = build_edge_index(x, max_distance=max_distance_cm)

                    # Perform lane allocation
                    try:
                        optimal_lane_ids: torch.Tensor = gat_model(
                            x, edge_index
                        ).argmax(dim=1)
                        # Map the optimal lane IDs back to the corresponding vehicles
                        optimal_lane_id = optimal_lane_ids[vehicle_ids.index(id)].item()
                    except Exception as e:
                        logger.error(f"Error during lane allocation: {e}")
                        optimal_lane_id = 1

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

        # Clean up stale entries
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
