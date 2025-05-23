from pathlib import Path

from ultralytics import YOLO

from shared_src.common import StoppableThread


class YOLOInference(StoppableThread):
    def __init__(self, model_path: Path, *args, confidence: float = 0.5, **kwargs):
        self.model_path = model_path
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        self.confidence = confidence
        self.model = YOLO(self.model_path, task="segment")
        super().__init__(*args, **kwargs)

    def infer(self, frame):
        # Perform inference
        self.model.predict(
            source=frame,
            conf=self.confidence,
            project=Path(CACHE_DIR, "runs", "segment"),
        )
