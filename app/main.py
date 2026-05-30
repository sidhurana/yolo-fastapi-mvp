from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.config import Settings, get_settings
from app.detector import YoloDetector
from app.models_registry import PRESET_DESCRIPTIONS, PRESETS
from app.schemas import DetectResponse, Detection, ModelInfoResponse

detector: Optional[YoloDetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    settings = get_settings()
    detector = YoloDetector(settings)
    yield
    detector = None


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Upload or stream camera frames for YOLO-based plant / weed inspection.",
        version="0.2.0",
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

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate",
                    "Pragma": "no-cache",
                },
            )

    @app.post("/api/v1/detect", response_model=DetectResponse)
    async def detect_image(
        file: UploadFile = File(..., description="JPEG/PNG from phone camera"),
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
        if settings.return_annotated_image and annotated is not None:
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="JPEG", quality=85)
            annotated_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return DetectResponse(
            issue_found=issue_found,
            issue_classes_matched=matched,
            detections=out_detections,
            detection_count=len(detections),
            inference_ms=round(inference_ms, 2),
            model=detector.model_name,
            model_preset=settings.model_preset,
            annotated_image_base64=annotated_b64,
            message=detector.build_message(issue_found, matched, len(detections)),
        )

    return app


app = create_app()
