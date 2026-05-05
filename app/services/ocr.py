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


def recognize_pdf(
    pdf_bytes: bytes,
    lang: str = "rus+eng",
    max_pages: int = 20,
    render_dpi: int = 200,
) -> str:
    """OCR для PDF: рендерим страницы в изображения и распознаём Tesseract."""

    if not pdf_bytes:
        raise ValueError("Пустой PDF-файл.")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Установите зависимость pymupdf для OCR PDF-файлов.") from exc

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("Не удалось открыть PDF-файл.") from exc

    try:
        if document.page_count == 0:
            raise ValueError("PDF-файл не содержит страниц.")
        if document.page_count > max_pages:
            raise ValueError(f"PDF слишком длинный. Максимум страниц: {max_pages}.")

        embedded_texts = []
        for page_index in range(document.page_count):
            page_text = document.load_page(page_index).get_text("text")
            page_text = _cleanup_text(page_text)
            if page_text:
                embedded_texts.append(page_text)
        embedded_text = _cleanup_text("\n\n".join(embedded_texts))
        if len(embedded_text) >= 20:
            return embedded_text

        zoom = max(render_dpi, 72) / 72
        matrix = fitz.Matrix(zoom, zoom)
        page_texts = []

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            text = recognize_image(pixmap.tobytes("png"), lang=lang)
            if text:
                page_texts.append(f"Страница {page_index + 1}\n{text}")

        result = "\n\n".join(page_texts)
        if not result:
            raise ValueError("Не удалось распознать текст в PDF.")
        return _cleanup_text(result)
    finally:
        document.close()
