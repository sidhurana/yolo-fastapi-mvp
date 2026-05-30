import io
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.config import Settings
from app.model_loader import resolve_yolo_weights
from app.models_registry import PRESET_DESCRIPTIONS
from app.schemas import BoundingBox, Detection


class YoloDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._issue_class_names = {
            name.strip().lower()
            for name in settings.issue_classes.split(",")
            if name.strip()
        }
        self._issue_keywords = [
            kw.strip().lower()
            for kw in settings.issue_keywords.split(",")
            if kw.strip()
        ]
        self._issue_match_mode = settings.issue_match_mode.strip().lower()

        weights_path = resolve_yolo_weights(settings.yolo_model, settings.model_preset)
        print(f"Loading model: {weights_path} (preset={settings.model_preset}) …")
        self.model = YOLO(weights_path)
        self._device = settings.device or None
        self.model_name = weights_path
        self.preset_description = PRESET_DESCRIPTIONS.get(
            settings.model_preset, settings.model_preset
        )
        self.class_names = dict(self.model.names or {})

    def _parse_image(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)

    def _normalize_class_label(self, class_name: str) -> str:
        return class_name.replace("___", " — ").replace("_", " ").strip()

    def is_issue_class(self, class_name: str) -> bool:
        raw_lower = class_name.lower()
        norm_lower = self._normalize_class_label(class_name).lower()

        if self._issue_match_mode == "all":
            return True

        if self._issue_match_mode == "keywords":
            if "healthy" in raw_lower or "healthy" in norm_lower:
                return False
            if self._issue_class_names and raw_lower in self._issue_class_names:
                return True
            return any(kw in raw_lower or kw in norm_lower for kw in self._issue_keywords)

        # exact
        return raw_lower in self._issue_class_names

    def detect(self, image_bytes: bytes) -> Tuple[List[Detection], Optional[np.ndarray], float]:
        frame = self._parse_image(image_bytes)
        started = time.perf_counter()

        predict_kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.settings.confidence_threshold,
            "iou": self.settings.iou_threshold,
            "verbose": False,
        }
        if self._device:
            predict_kwargs["device"] = self._device
        results = self.model.predict(**predict_kwargs)
        inference_ms = (time.perf_counter() - started) * 1000

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names or {}
            for box in result.boxes:
                cls_id = int(box.cls.item())
                class_name = str(names.get(cls_id, cls_id))
                confidence = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=BoundingBox(
                            x1=xyxy[0],
                            y1=xyxy[1],
                            x2=xyxy[2],
                            y2=xyxy[3],
                        ),
                    )
                )

        annotated: Optional[np.ndarray] = None
        if self.settings.return_annotated_image and results:
            annotated = results[0].plot()

        return detections, annotated, inference_ms

    def issue_classes_matched(self, detections: List[Detection]) -> List[str]:
        matched: List[str] = []
        for det in detections:
            if self.is_issue_class(det.class_name):
                matched.append(self._normalize_class_label(det.class_name))
        return sorted(set(matched))

    def build_message(
        self,
        issue_found: bool,
        matched: List[str],
        detection_count: int,
    ) -> str:
        if issue_found:
            classes = ", ".join(matched[:3])
            extra = f" (+{len(matched) - 3} more)" if len(matched) > 3 else ""
            return f"Possible plant issue: {classes}{extra}."
        if detection_count:
            return (
                f"Detected {detection_count} object(s) but none matched disease/pest keywords. "
                "Try closer framing or a different angle."
            )
        return (
            "No plant disease detected. Ensure the leaf/plant fills the frame and lighting is good."
        )
