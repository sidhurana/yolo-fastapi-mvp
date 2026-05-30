from pathlib import Path
from typing import List, Optional, Tuple

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.advisory import ClassAdvisory, build_advisories, build_summary
from app.database import DetectionEvent
from app.schemas import Detection, DetectResponse


def _detections_to_json(detections: List[Detection]) -> list:
    return [d.model_dump() for d in detections]


def save_detection_event(
    db: Session,
    response: DetectResponse,
    advisories: List[ClassAdvisory],
    summary_suggestion: str,
    annotated_bytes: Optional[bytes],
    uploads_dir: Path,
    source_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> DetectionEvent:
    event = DetectionEvent(
        issue_found=response.issue_found,
        message=response.message,
        summary_suggestion=summary_suggestion,
        model_preset=response.model_preset,
        model_path=response.model,
        inference_ms=response.inference_ms,
        detection_count=response.detection_count,
        issue_classes_matched=response.issue_classes_matched,
        detections=_detections_to_json(response.detections),
        advisories=[a.model_dump() for a in advisories],
        source_ip=source_ip,
        user_agent=user_agent,
    )
    db.add(event)
    db.flush()

    if annotated_bytes:
        image_path = uploads_dir / "events" / f"{event.id}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(annotated_bytes)
        event.annotated_image_path = f"uploads/events/{event.id}.jpg"

    db.commit()
    db.refresh(event)
    return event


def list_events(
    db: Session,
    skip: int = 0,
    limit: int = 50,
    issue_only: Optional[bool] = None,
) -> Tuple[List[DetectionEvent], int]:
    q = db.query(DetectionEvent)
    if issue_only is True:
        q = q.filter(DetectionEvent.issue_found.is_(True))
    elif issue_only is False:
        q = q.filter(DetectionEvent.issue_found.is_(False))
    total = q.count()
    rows = q.order_by(desc(DetectionEvent.created_at)).offset(skip).limit(limit).all()
    return rows, total


def get_event(db: Session, event_id: int) -> Optional[DetectionEvent]:
    return db.query(DetectionEvent).filter(DetectionEvent.id == event_id).first()


def get_stats(db: Session) -> dict:
    total = db.query(func.count(DetectionEvent.id)).scalar() or 0
    issues = (
        db.query(func.count(DetectionEvent.id))
        .filter(DetectionEvent.issue_found.is_(True))
        .scalar()
        or 0
    )
    avg_ms = db.query(func.avg(DetectionEvent.inference_ms)).scalar() or 0.0
    return {
        "total_events": total,
        "issues_found": issues,
        "clear_scans": total - issues,
        "avg_inference_ms": round(float(avg_ms), 2),
    }


def build_advisory_payload(
    detections: List[Detection],
    issue_classes_matched: List[str],
    issue_found: bool,
) -> Tuple[List[ClassAdvisory], str]:
    names_for_advice = issue_classes_matched or [d.class_name for d in detections if d.is_issue]
    if not names_for_advice and detections:
        names_for_advice = [d.class_name for d in detections]
    advisories = build_advisories(names_for_advice, issue_only=False)
    summary = build_summary(issue_found, issue_classes_matched, advisories)
    return advisories, summary
