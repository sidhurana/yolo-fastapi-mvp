from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db, init_db
from app.detector import YoloDetector
from app.event_service import (
    build_advisory_payload,
    get_event,
    get_stats,
    list_events,
    save_detection_event,
)
from app.models_registry import PRESET_DESCRIPTIONS, PRESETS
from app.schemas import (
    AdminStatsOut,
    ClassAdvisoryOut,
    DetectResponse,
    Detection,
    DetectionEventListOut,
    DetectionEventOut,
    ModelInfoResponse,
)
from app.serializers import event_to_out

detector: Optional[YoloDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    settings = get_settings()
    init_db(settings)
    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
    detector = YoloDetector(settings)
    yield
    detector = None


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Upload or stream camera frames for YOLO-based plant / weed inspection.",
        version="0.3.0",
        lifespan=lifespan,
    )

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory="static"), name="static")
    uploads_path = Path(settings.uploads_dir)
    uploads_path.mkdir(parents=True, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")

    @app.get("/health")
    async def health(db: Session = Depends(get_db)) -> dict[str, str]:
        from sqlalchemy import text

        try:
            db.execute(text("SELECT 1"))
            return {"status": "ok", "database": "ok"}
        except Exception:
            return {"status": "degraded", "database": "error"}

    @app.get("/api/v1/model-info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        if detector is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        names = list(detector.class_names.values())
        return ModelInfoResponse(
            model=detector.model_name,
            model_preset=settings.model_preset,
            description=detector.preset_description,
            issue_match_mode=settings.issue_match_mode,
            class_count=len(names),
            sample_classes=names[:12],
            live_camera_requires_https=True,
        )

    @app.get("/api/v1/presets")
    async def list_presets() -> dict[str, object]:
        return {
            "presets": {
                k: {"description": PRESET_DESCRIPTIONS.get(k, k), "yolo_model": v["yolo_model"]}
                for k, v in PRESETS.items()
            },
            "active": settings.model_preset,
        }

    @app.get("/", response_class=HTMLResponse)
    async def camera_page() -> HTMLResponse:
        with open("static/camera.html", encoding="utf-8") as f:
            return HTMLResponse(
                f.read(),
                headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
            )

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page() -> HTMLResponse:
        with open("static/admin.html", encoding="utf-8") as f:
            return HTMLResponse(
                f.read(),
                headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
            )

    @app.get("/api/v1/admin/stats", response_model=AdminStatsOut)
    async def admin_stats(db: Session = Depends(get_db)) -> AdminStatsOut:
        return AdminStatsOut(**get_stats(db))

    @app.get("/api/v1/admin/events", response_model=DetectionEventListOut)
    async def admin_list_events(
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
        issue_only: Optional[bool] = Query(None),
        db: Session = Depends(get_db),
    ) -> DetectionEventListOut:
        rows, total = list_events(db, skip=skip, limit=limit, issue_only=issue_only)
        return DetectionEventListOut(
            items=[event_to_out(r) for r in rows],
            total=total,
            skip=skip,
            limit=limit,
        )

    @app.get("/api/v1/admin/events/{event_id}", response_model=DetectionEventOut)
    async def admin_get_event(event_id: int, db: Session = Depends(get_db)) -> DetectionEventOut:
        event = get_event(db, event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        return event_to_out(event)

    @app.post("/api/v1/detect", response_model=DetectResponse)
    async def detect_image(
        request: Request,
        file: UploadFile = File(..., description="JPEG/PNG from phone camera"),
        db: Session = Depends(get_db),
    ) -> DetectResponse:
        if detector is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Expected an image file")

        data = await file.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large (max {settings.max_upload_bytes} bytes)",
            )
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        detections, annotated, inference_ms = detector.detect(data)
        matched = detector.issue_classes_matched(detections)
        issue_found = len(matched) > 0

        out_detections: list[Detection] = []
        for det in detections:
            out_detections.append(
                Detection(
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    is_issue=detector.is_issue_class(det.class_name),
                )
            )

        annotated_b64: Optional[str] = None
        annotated_bytes: Optional[bytes] = None
        if settings.return_annotated_image and annotated is not None:
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="JPEG", quality=85)
            annotated_bytes = buf.getvalue()
            annotated_b64 = base64.b64encode(annotated_bytes).decode("ascii")

        message = detector.build_message(issue_found, matched, len(detections))
        advisories, summary = build_advisory_payload(out_detections, matched, issue_found)
        advisory_out = [ClassAdvisoryOut(**a.model_dump()) for a in advisories]

        response = DetectResponse(
            issue_found=issue_found,
            issue_classes_matched=matched,
            detections=out_detections,
            detection_count=len(detections),
            inference_ms=round(inference_ms, 2),
            model=detector.model_name,
            model_preset=settings.model_preset,
            annotated_image_base64=annotated_b64,
            message=message,
            summary_suggestion=summary,
            advisories=advisory_out,
        )

        try:
            event = save_detection_event(
                db=db,
                response=response,
                advisories=advisories,
                summary_suggestion=summary,
                annotated_bytes=annotated_bytes if settings.store_event_images else None,
                uploads_dir=Path(settings.uploads_dir),
                source_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
            response.event_id = event.id
        except Exception as exc:
            db.rollback()
            print(f"Warning: failed to save detection event: {exc}")

        return response

    return app


app = create_app()
