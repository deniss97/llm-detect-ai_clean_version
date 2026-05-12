from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import logging
import os
import re

import cv2
import numpy as np
from PIL import Image


DEFAULT_MODEL_NAME = "cyrillic-trocr/trocr-handwritten-cyrillic"
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def cleanup_text(text: str) -> str:
    text = text.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class LineSegmentationOptions:
    min_line_height: int = 20
    threshold_ratio: float = 0.02
    padding: int = 15


def decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Пустой файл изображения.")

    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось открыть изображение.")
    return image


def load_and_preprocess(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = decode_image(image_bytes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=10,
    )
    return image, gray, binary


def encode_preview(image: np.ndarray, max_width: int = 1000) -> str:
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(
            image,
            (max_width, max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    pil_image = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def segment_lines(
    binary_image: np.ndarray,
    original_image: np.ndarray,
    options: LineSegmentationOptions,
) -> list[np.ndarray]:
    horizontal_projection = np.sum(binary_image, axis=1)
    max_projection = float(np.max(horizontal_projection)) if horizontal_projection.size else 0.0
    if max_projection <= 0:
        return [original_image]

    threshold = max_projection * options.threshold_ratio
    lines: list[tuple[int, int]] = []
    in_line = False
    start = 0

    for index, value in enumerate(horizontal_projection):
        if value > threshold and not in_line:
            in_line = True
            start = index
        elif value <= threshold and in_line:
            in_line = False
            if index - start > options.min_line_height:
                lines.append((start, index))

    if in_line and len(horizontal_projection) - start > options.min_line_height:
        lines.append((start, len(horizontal_projection)))

    if not lines:
        return [original_image]

    line_images = []
    height = original_image.shape[0]
    for y_start, y_end in lines:
        y_s = max(0, y_start - options.padding)
        y_e = min(height, y_end + options.padding)
        line_images.append(original_image[y_s:y_e, :])
    return line_images


class TrOCRRecognizer:
    mode = "trocr"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, local_files_only: bool = False):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Загружаем OCR-модель %s, device=%s, local_files_only=%s. "
            "Первый запуск может несколько минут скачивать model.safetensors.",
            model_name,
            self.device,
            local_files_only,
        )
        self.processor = TrOCRProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        logger.info("OCR-модель %s загружена.", model_name)

    def recognize_line(self, line_image: np.ndarray) -> str:
        if len(line_image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(line_image).convert("RGB")

        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with self.torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=128,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def recognize_document(self, line_images: list[np.ndarray]) -> str:
        text_lines = [self.recognize_line(line) for line in line_images]
        return cleanup_text("\n".join(line for line in text_lines if line))


@lru_cache(maxsize=2)
def get_recognizer(
    model_name: str = DEFAULT_MODEL_NAME,
    local_files_only: bool = False,
) -> TrOCRRecognizer:
    return TrOCRRecognizer(model_name=model_name, local_files_only=local_files_only)


def recognize_image(
    image_bytes: bytes,
    model_name: str = DEFAULT_MODEL_NAME,
    local_files_only: bool = False,
    min_line_height: int = 20,
    line_threshold_ratio: float = 0.02,
    line_padding: int = 15,
) -> str:
    original, _, binary = load_and_preprocess(image_bytes)
    line_images = segment_lines(
        binary,
        original,
        LineSegmentationOptions(
            min_line_height=min_line_height,
            threshold_ratio=line_threshold_ratio,
            padding=line_padding,
        ),
    )
    logger.info("OCR: найдено строк для распознавания: %s", len(line_images))
    text = get_recognizer(model_name, local_files_only).recognize_document(line_images)
    if not text:
        raise ValueError("Не удалось распознать текст на изображении.")
    return text


def segment_image_previews(
    image_bytes: bytes,
    min_line_height: int = 20,
    line_threshold_ratio: float = 0.02,
    line_padding: int = 15,
) -> list[str]:
    original, _, binary = load_and_preprocess(image_bytes)
    line_images = segment_lines(
        binary,
        original,
        LineSegmentationOptions(
            min_line_height=min_line_height,
            threshold_ratio=line_threshold_ratio,
            padding=line_padding,
        ),
    )
    return [encode_preview(line) for line in line_images]


def recognize_pdf(
    pdf_bytes: bytes,
    model_name: str = DEFAULT_MODEL_NAME,
    local_files_only: bool = False,
    max_pages: int = 20,
    render_dpi: int = 200,
    min_line_height: int = 20,
    line_threshold_ratio: float = 0.02,
    line_padding: int = 15,
) -> str:
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

        zoom = max(render_dpi, 72) / 72
        matrix = fitz.Matrix(zoom, zoom)
        page_texts = []

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            try:
                text = recognize_image(
                    pixmap.tobytes("png"),
                    model_name=model_name,
                    local_files_only=local_files_only,
                    min_line_height=min_line_height,
                    line_threshold_ratio=line_threshold_ratio,
                    line_padding=line_padding,
                )
            except ValueError:
                text = ""
            if text:
                page_texts.append(f"Страница {page_index + 1}\n{text}")

        result = cleanup_text("\n\n".join(page_texts))
        if not result:
            raise ValueError("Не удалось распознать текст в PDF.")
        return result
    finally:
        document.close()
