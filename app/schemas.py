from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    is_issue: bool = False


class DetectResponse(BaseModel):
    issue_found: bool
    issue_classes_matched: List[str]
    detections: List[Detection]
    detection_count: int
    inference_ms: float
    model: str
    model_preset: str = ""
    annotated_image_base64: Optional[str] = None
    message: str


class ModelInfoResponse(BaseModel):
    model: str
    model_preset: str
    description: str
    issue_match_mode: str
    class_count: int
    sample_classes: List[str]
    live_camera_requires_https: bool = True
