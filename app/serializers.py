from pathlib import Path
from typing import List, Optional

from app.database import DetectionEvent
from app.schemas import ClassAdvisoryOut, Detection, DetectionEventOut


def event_to_out(event: DetectionEvent) -> DetectionEventOut:
    detections = [Detection(**d) for d in (event.detections or [])]
    advisories = [ClassAdvisoryOut(**a) for a in (event.advisories or [])]
    image_url = None
    if event.annotated_image_path:
        image_url = "/" + event.annotated_image_path.lstrip("/")
    return DetectionEventOut(
        id=event.id,
        created_at=event.created_at,
        issue_found=event.issue_found,
        message=event.message,
        summary_suggestion=event.summary_suggestion or "",
        model_preset=event.model_preset,
        model_path=event.model_path,
        inference_ms=event.inference_ms,
        detection_count=event.detection_count,
        issue_classes_matched=event.issue_classes_matched or [],
        detections=detections,
        advisories=advisories,
        annotated_image_url=image_url,
        source_ip=event.source_ip,
    )
