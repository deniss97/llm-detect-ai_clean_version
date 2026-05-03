from io import BytesIO
import re

from PIL import Image, ImageOps
import pytesseract


def _cleanup_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def recognize_image(image_bytes: bytes, lang: str = "rus+eng") -> str:
    """OCR для изображения.

    Для русского языка в системе должен быть установлен пакет tesseract-ocr-rus.
    """
    if not image_bytes:
        raise ValueError("Пустой файл изображения.")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Не удалось открыть изображение.") from exc

    # Лёгкая предобработка: grayscale + auto contrast.
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)

    try:
        raw_text = pytesseract.image_to_string(image, lang=lang, config="--psm 6")
    except pytesseract.TesseractError as exc:
        raise RuntimeError(
            "Ошибка OCR. Проверьте, что установлен Tesseract и языки rus/eng."
        ) from exc

    return _cleanup_text(raw_text)
