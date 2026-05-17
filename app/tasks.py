import base64
import logging
from functools import lru_cache
from time import perf_counter
from typing import Any

from app.celery_app import celery_app
from app.config import settings
from app.database import SessionLocal, init_db
from app.models import Job
from app.services.detection import build_detection_result, validate_detection_text
from app.services.detector import Detector, build_detector
from app.services.ocr import recognize_image, recognize_pdf


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_worker_detector() -> Detector:
    return build_detector(settings)


def set_job_running(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} не найден.")
        job.status = "running"
        job.error_message = None
        db.commit()
    finally:
        db.close()


def set_job_finished(job_id: int, result: dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            raise ValueError(f"Job {job_id} не найден.")
        job.status = "finished"
        job.result_json = result
        job.error_message = None
        db.commit()
    finally:
        db.close()


def set_job_failed(job_id: int, error: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            logger.error("Не удалось записать ошибку: job %s не найден.", job_id)
            return
        job.status = "failed"
        job.error_message = error
        db.commit()
    finally:
        db.close()


@celery_app.task(name="app.tasks.run_ocr_job")
def run_ocr_job(
    job_id: int,
    file_base64: str,
    filename: str,
    content_type: str,
    line_options: dict[str, Any],
) -> dict[str, Any]:
    started_at = perf_counter()
    logger.info("OCR job %s started: filename=%s content_type=%s", job_id, filename, content_type)
    init_db()
    try:
        set_job_running(job_id)
        file_bytes = base64.b64decode(file_base64.encode("ascii"))
        is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
        if is_pdf:
            text = recognize_pdf(
                file_bytes,
                model_name=settings.ocr_model_name,
                local_files_only=settings.ocr_model_local_files_only,
                max_pages=settings.pdf_max_pages,
                render_dpi=settings.pdf_render_dpi,
                **line_options,
            )
        else:
            text = recognize_image(
                file_bytes,
                model_name=settings.ocr_model_name,
                local_files_only=settings.ocr_model_local_files_only,
                **line_options,
            )
        result = {"text": text}
        set_job_finished(job_id, result)
        logger.info("OCR job %s finished in %.2fs", job_id, perf_counter() - started_at)
        return result
    except Exception as exc:
        set_job_failed(job_id, str(exc))
        logger.exception("OCR job %s failed in %.2fs", job_id, perf_counter() - started_at)
        raise


@celery_app.task(name="app.tasks.run_detection_job")
def run_detection_job(job_id: int, text: str) -> dict[str, Any]:
    started_at = perf_counter()
    logger.info("Detection job %s started", job_id)
    init_db()
    try:
        set_job_running(job_id)
        normalized_text = validate_detection_text(text)
        detection = build_detection_result(
            normalized_text,
            get_worker_detector(),
            source="async_job",
        )

        db = SessionLocal()
        try:
            db.add(detection)
            db.commit()
            db.refresh(detection)
            result = {
                "id": detection.id,
                "text": detection.text,
                "ai_probability": detection.ai_probability,
                "ai_percent": detection.ai_percent,
                "verdict": detection.verdict,
                "confidence": detection.confidence,
                "created_at": detection.created_at.isoformat(),
            }
        finally:
            db.close()

        set_job_finished(job_id, result)
        logger.info("Detection job %s finished in %.2fs", job_id, perf_counter() - started_at)
        return result
    except Exception as exc:
        set_job_failed(job_id, str(exc))
        logger.exception("Detection job %s failed in %.2fs", job_id, perf_counter() - started_at)
        raise
