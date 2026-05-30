from typing import List, Optional

from datetime import datetime

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


class ClassAdvisoryOut(BaseModel):
    class_name: str
    display_name: str
    severity: str
    note: str
    suggestion: str


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
    event_id: Optional[int] = None
    summary_suggestion: Optional[str] = None
    advisories: List[ClassAdvisoryOut] = []


class DetectionEventOut(BaseModel):
    id: int
    created_at: datetime
    issue_found: bool
    message: str
    summary_suggestion: str
    model_preset: str
    model_path: str
    inference_ms: float
    detection_count: int
    issue_classes_matched: List[str]
    detections: List[Detection]
    advisories: List[ClassAdvisoryOut]
    annotated_image_url: Optional[str] = None
    source_ip: Optional[str] = None


class DetectionEventListOut(BaseModel):
    items: List[DetectionEventOut]
    total: int
    skip: int
    limit: int


class AdminStatsOut(BaseModel):
    total_events: int
    issues_found: int
    clear_scans: int
    avg_inference_ms: float


class ModelInfoResponse(BaseModel):
    model: str
    model_preset: str
    description: str
    issue_match_mode: str
    class_count: int
    sample_classes: List[str]
    live_camera_requires_https: bool = True
