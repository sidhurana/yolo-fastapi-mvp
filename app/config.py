from functools import lru_cache
from typing import Any, Dict

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models_registry import PRESETS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # plant_disease | agrosight | plant_local | coco_demo | custom
    model_preset: str = "plant_disease"

    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = ""

    # exact = only ISSUE_CLASSES | keywords = ISSUE_KEYWORDS in class name | all = any detection
    issue_match_mode: str = "keywords"
    issue_classes: str = ""
    issue_keywords: str = ""

    return_annotated_image: bool = True
    max_upload_bytes: int = 10 * 1024 * 1024

    app_name: str = "YOLO Plant Inspector"
    cors_origins: str = "*"

    @model_validator(mode="before")
    @classmethod
    def apply_model_preset(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        preset_name = (data.get("model_preset") or "plant_disease").strip()
        if preset_name == "custom":
            return data
        preset = PRESETS.get(preset_name)
        if not preset:
            return data
        merged: Dict[str, Any] = dict(preset)
        merged.update(data)
        return merged


@lru_cache
def get_settings() -> Settings:
    return Settings()
