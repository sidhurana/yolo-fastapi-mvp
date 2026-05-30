"""Resolve YOLO weight paths; download from Hugging Face when needed."""

from pathlib import Path
from typing import Optional, Tuple

# repo_id, filename
HF_SOURCES: dict[str, Tuple[str, str]] = {
    # https://huggingface.co/JK-TK/PlantDiseaseDetection
    "plant_disease": ("JK-TK/PlantDiseaseDetection", "PlantDiseaseDetection.pt"),
    # https://huggingface.co/Nick-Maximillien/Agrosight-YOLOv11-Crop-Disease
    "agrosight": ("Nick-Maximillien/Agrosight-YOLOv11-Crop-Disease", "best.pt"),
    # https://huggingface.co/iamnotpalak/yolov8-transfpn-crop-disease-detection
    "crop_stress": (
        "iamnotpalak/yolov8-transfpn-crop-disease-detection",
        "weights/best.pt",
    ),
}


def _download_hf(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    print(f"Downloading {repo_id}/{filename} from Hugging Face (first run may take a few minutes)…")
    local = hf_hub_download(repo_id=repo_id, filename=filename)
    return Path(local)


def resolve_yolo_weights(yolo_model: str, model_preset: str = "") -> str:
    """
    Return a local filesystem path suitable for ultralytics.YOLO().

    - weights/plant_disease.pt → download from HF if missing (when preset matches)
    - hf://org/repo/file.pt → download via huggingface_hub (ultralytics does not support hf://)
    """
    path = Path(yolo_model)

    # Legacy / broken: ultralytics turns hf:// into hf:/ and fails
    if yolo_model.startswith("hf:/"):
        yolo_model = "hf://" + yolo_model[5:]
    if yolo_model.startswith("hf://"):
        repo_and_file = yolo_model[len("hf://") :]
        repo_id, filename = repo_and_file.rsplit("/", 1)
        local = _download_hf(repo_id, filename)
        return str(local)

    # Local weights missing — try preset HF fallback
    if not path.is_file() and model_preset in HF_SOURCES:
        repo_id, filename = HF_SOURCES[model_preset]
        dest = Path("weights") / f"{model_preset}.pt"
        if dest.is_file():
            return str(dest.resolve())
        dest.parent.mkdir(parents=True, exist_ok=True)
        local = _download_hf(repo_id, filename)
        import shutil

        shutil.copy2(local, dest)
        print(f"Cached weights at {dest}")
        return str(dest.resolve())

    if path.is_file():
        return str(path.resolve())

    # yolov8n.pt etc. — let ultralytics download from its own hub
    return yolo_model
