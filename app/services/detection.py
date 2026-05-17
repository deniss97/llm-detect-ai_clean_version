from app.config import settings
from app.models import DetectionResult
from app.services.detector import Detector


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\r", "\n").split())


def make_verdict(probability: float) -> tuple[str, float]:
    if probability >= settings.threshold:
        return "AI_GENERATED", probability
    return "HUMAN_WRITTEN", 1.0 - probability


def validate_detection_text(text: str) -> str:
    normalized = normalize_text(text)
    if len(normalized) < 20:
        raise ValueError("Текст слишком короткий для проверки.")
    if len(normalized) > settings.max_text_length:
        raise ValueError(f"Текст слишком длинный. Максимум: {settings.max_text_length} символов.")
    return normalized


def build_detection_result(
    text: str,
    detector: Detector,
    source: str = "image_ocr_or_manual",
) -> DetectionResult:
    probability = detector.predict_probability(text)
    verdict, confidence = make_verdict(probability)
    return DetectionResult(
        text=text,
        ai_probability=probability,
        ai_percent=round(probability * 100, 2),
        verdict=verdict,
        confidence=round(confidence, 4),
        source=source,
    )
