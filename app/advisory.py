"""Generate agronomic notes and treatment suggestions from YOLO class names."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class ClassAdvisory(BaseModel):
    class_name: str
    display_name: str
    severity: str  # low | medium | high
    note: str
    suggestion: str


# Keyword → advisory template (matched against normalized class name)
_KEYWORD_ADVICE: List[Dict[str, str]] = [
    {
        "keywords": "late blight",
        "severity": "high",
        "note": "Late blight is a fast-spreading fungal-like disease favored by cool, wet weather.",
        "suggestion": "Remove infected leaves immediately. Improve airflow, avoid overhead watering, and apply copper-based fungicide. Do not compost infected material.",
    },
    {
        "keywords": "early blight",
        "severity": "medium",
        "note": "Early blight causes concentric leaf spots and defoliation over time.",
        "suggestion": "Prune lower leaves for airflow. Mulch around base. Rotate crops yearly and use chlorothalonil or copper spray at first sign.",
    },
    {
        "keywords": "blight",
        "severity": "high",
        "note": "Blight diseases can spread rapidly under humid conditions.",
        "suggestion": "Isolate affected plants, remove infected tissue, improve drainage and spacing, and apply appropriate fungicide for your crop.",
    },
    {
        "keywords": "rust",
        "severity": "medium",
        "note": "Rust appears as orange/brown pustules on leaves and reduces photosynthesis.",
        "suggestion": "Remove affected leaves. Avoid wetting foliage. Apply sulfur or fungicide labeled for rust on your crop type.",
    },
    {
        "keywords": "powdery mildew",
        "severity": "medium",
        "note": "Powdery mildew forms white dusty patches, common in warm dry days and cool nights.",
        "suggestion": "Increase spacing and sunlight. Spray neem oil, potassium bicarbonate, or sulfur. Remove heavily infected leaves.",
    },
    {
        "keywords": "mildew",
        "severity": "medium",
        "note": "Mildew infections weaken plants and reduce yield.",
        "suggestion": "Improve ventilation. Apply fungicide early. Remove infected leaves and avoid excess nitrogen.",
    },
    {
        "keywords": "mosaic virus",
        "severity": "high",
        "note": "Viral mosaic causes mottled discoloration; there is no cure once established.",
        "suggestion": "Remove and destroy infected plants. Control aphids (vectors). Use virus-free seeds and resistant varieties next season.",
    },
    {
        "keywords": "virus",
        "severity": "high",
        "note": "Plant viruses are systemic and spread by insects, tools, or contact.",
        "suggestion": "Remove infected plants. Disinfect tools. Control insect vectors and plant resistant varieties.",
    },
    {
        "keywords": "scab",
        "severity": "medium",
        "note": "Scab causes lesions on leaves and fruit, thriving in wet spring weather.",
        "suggestion": "Rake and remove fallen leaves. Apply dormant-season spray. Choose scab-resistant cultivars.",
    },
    {
        "keywords": "rot",
        "severity": "high",
        "note": "Rot diseases often indicate excess moisture or poor drainage.",
        "suggestion": "Improve drainage, reduce watering, remove rotted tissue. Treat with fungicide if appropriate for the crop.",
    },
    {
        "keywords": "wilt",
        "severity": "high",
        "note": "Wilting may indicate fungal wilt, bacterial infection, or root damage.",
        "suggestion": "Check roots and stem base. Remove severely affected plants. Rotate crops and use disease-free soil.",
    },
    {
        "keywords": "spot",
        "severity": "low",
        "note": "Leaf spot diseases cause localized lesions and gradual leaf loss.",
        "suggestion": "Remove spotted leaves. Water at soil level. Apply copper or fungicide if spots spread.",
    },
    {
        "keywords": "pest",
        "severity": "medium",
        "note": "Pest damage can mimic disease and weaken plant defenses.",
        "suggestion": "Identify the pest. Use targeted treatment (insecticidal soap, neem, or biological controls). Monitor weekly.",
    },
    {
        "keywords": "weed",
        "severity": "low",
        "note": "Weeds compete for nutrients, water, and light; some harbor pests and diseases.",
        "suggestion": "Hand-pull or mulch around plants. Maintain clean rows. Consider pre-emergent herbicide only if labeled for your crop.",
    },
    {
        "keywords": "chlorosis",
        "severity": "medium",
        "note": "Yellowing may indicate nutrient deficiency, root stress, or disease.",
        "suggestion": "Test soil pH and nutrients. Check watering. Rule out root rot before fertilizing.",
    },
    {
        "keywords": "canker",
        "severity": "high",
        "note": "Cankers are sunken dead areas on stems/branches that can girdle plants.",
        "suggestion": "Prune infected branches below the canker. Disinfect tools between cuts. Avoid wounding plants.",
    },
    {
        "keywords": "curl",
        "severity": "low",
        "note": "Leaf curl can be caused by viruses, aphids, or environmental stress.",
        "suggestion": "Inspect undersides for aphids. If viral, remove plant. Otherwise improve consistent watering.",
    },
    {
        "keywords": "mold",
        "severity": "medium",
        "note": "Mold growth indicates persistent humidity on leaf surfaces.",
        "suggestion": "Reduce humidity, improve spacing, remove affected parts, and apply fungicide if needed.",
    },
    {
        "keywords": "healthy",
        "severity": "low",
        "note": "No significant disease indicators detected for this class.",
        "suggestion": "Continue regular monitoring, balanced watering, and good sanitation practices.",
    },
]

_DEFAULT_ISSUE = {
    "severity": "medium",
    "note": "A potential plant health issue was detected by the vision model.",
    "suggestion": "Inspect the plant closely, isolate if spreading, and consult a local agronomist for treatment specific to your crop and region.",
}

_DEFAULT_CLEAR = {
    "severity": "low",
    "note": "No disease keywords matched in this scan.",
    "suggestion": "Retake the photo with the leaf filling the frame. Continue routine monitoring.",
}


def _normalize(name: str) -> str:
    return name.replace("___", " ").replace("_", " ").lower().strip()


def _display_name(class_name: str) -> str:
    return class_name.replace("___", " — ").replace("_", " ").strip()


def _match_advice(normalized: str) -> Dict[str, str]:
    for entry in _KEYWORD_ADVICE:
        if entry["keywords"] in normalized:
            return entry
    return _DEFAULT_ISSUE


def build_advisories(class_names: List[str], issue_only: bool = True) -> List[ClassAdvisory]:
    seen = set()
    advisories: List[ClassAdvisory] = []
    for raw in class_names:
        norm = _normalize(raw)
        if norm in seen:
            continue
        seen.add(norm)
        if issue_only and "healthy" in norm:
            continue
        advice = _match_advice(norm)
        advisories.append(
            ClassAdvisory(
                class_name=raw,
                display_name=_display_name(raw),
                severity=advice["severity"],
                note=advice["note"],
                suggestion=advice["suggestion"],
            )
        )
    return advisories


def build_summary(
    issue_found: bool,
    matched_labels: List[str],
    advisories: List[ClassAdvisory],
) -> str:
    if not issue_found:
        return _DEFAULT_CLEAR["suggestion"]
    if not advisories:
        return _DEFAULT_ISSUE["suggestion"]
    top = advisories[0]
    parts = [f"Primary concern: {top.display_name} ({top.severity} severity)."]
    parts.append(top.suggestion)
    if len(advisories) > 1:
        others = ", ".join(a.display_name for a in advisories[1:3])
        parts.append(f"Also detected: {others}. Review each item below.")
    return " ".join(parts)
