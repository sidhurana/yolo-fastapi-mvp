"""Known YOLO presets — plant/crop models (not COCO)."""

from typing import Any, Dict

# Hugging Face weights load via ultralytics: YOLO("hf://org/repo/file.pt")
PRESETS: Dict[str, Dict[str, Any]] = {
    # Default COCO — poor for plants; kept for offline/smoke tests only.
    "coco_demo": {
        "yolo_model": "yolov8n.pt",
        "confidence_threshold": 0.35,
        "issue_match_mode": "exact",
        "issue_classes": "potted plant",
        "issue_keywords": "",
    },
    # ~116 crop disease classes (lab + field). Best default for this app.
    # https://huggingface.co/JK-TK/PlantDiseaseDetection
    # Downloaded to weights/plant_disease.pt on first start if missing.
    "plant_disease": {
        "yolo_model": "weights/plant_disease.pt",
        "confidence_threshold": 0.25,
        "issue_match_mode": "keywords",
        "issue_classes": "",
        "issue_keywords": (
            "blight,rust,scab,mold,rot,spot,virus,wilt,canker,pest,weed,mildew,"
            "smut,anthracnose,chlorosis,necrosis,disease,scorch,curl,mosaic,"
            "yellow,powdery,leaf spot,fire blight"
        ),
    },
    # African staple crops — pests, deficiencies, diseases.
    # https://huggingface.co/Nick-Maximillien/Agrosight-YOLOv11-Crop-Disease
    "agrosight": {
        "yolo_model": "weights/agrosight.pt",
        "confidence_threshold": 0.25,
        "issue_match_mode": "keywords",
        "issue_classes": "",
        "issue_keywords": (
            "disease,pest,deficiency,blight,rust,rot,virus,wilt,spot,mold,weed,"
            "damage,infected"
        ),
    },
    # Local file after: python scripts/download_model.py --preset plant_disease
    "plant_local": {
        "yolo_model": "weights/plant_disease.pt",
        "confidence_threshold": 0.25,
        "issue_match_mode": "keywords",
        "issue_classes": "",
        "issue_keywords": (
            "blight,rust,scab,mold,rot,spot,virus,wilt,canker,pest,weed,mildew,"
            "smut,anthracnose,chlorosis,necrosis,disease"
        ),
    },
    "crop_stress": {
        "yolo_model": "weights/crop_stress.pt",
        "confidence_threshold": 0.25,
        "issue_match_mode": "keywords",
        "issue_classes": "",
        "issue_keywords": "stress,disease,deficiency,blight,rust,rot,virus,spot,mold,pest",
    },
}

PRESET_DESCRIPTIONS = {
    "coco_demo": "COCO yolov8n (not for plants — demo only)",
    "plant_disease": "JK-TK PlantDiseaseDetection YOLOv11x (~116 disease classes)",
    "agrosight": "Agrosight YOLOv11 crop disease / pest / deficiency",
    "plant_local": "Cached weights at weights/plant_disease.pt",
    "crop_stress": "TransFPN-YOLOv8 crop stress detector (HF)",
    "custom": "Use YOLO_MODEL and ISSUE_* from .env only",
}
