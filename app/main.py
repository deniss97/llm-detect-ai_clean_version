from pathlib import Path
import logging

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, init_db
from app.models import DetectionResult
from app.schemas import DetectRequest, DetectResponse, HealthResponse, OCRResponse, ResultResponse
from app.services.detector import Detector, build_detector
from app.services.ocr import recognize_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

detector: Detector | None = None


@app.on_event("startup")
def on_startup() -> None:
    global detector
    logger.info("Инициализируем БД...")
    init_db()
    logger.info("Инициализируем детектор...")
    detector = build_detector(settings)
    logger.info("Приложение готово. Детектор: %s", detector.mode)


def get_detector() -> Detector:
    if detector is None:
        raise HTTPException(status_code=503, detail="Детектор ещё не загружен.")
    return detector


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\r", "\n").split())


def make_verdict(probability: float) -> tuple[str, float]:
    if probability >= settings.threshold:
        return "AI_GENERATED", probability
    return "HUMAN_WRITTEN", 1.0 - probability


def to_response(row: DetectionResult) -> DetectResponse:
    return DetectResponse(
        id=row.id,
        text=row.text,
        ai_probability=row.ai_probability,
        ai_percent=row.ai_percent,
        verdict=row.verdict,
        confidence=row.confidence,
        created_at=row.created_at,
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "threshold": settings.threshold,
            "detector_mode": detector.mode if detector else "loading",
        },
    )


@app.get("/api/health", response_model=HealthResponse)
def health(detector_service: Detector = Depends(get_detector)):
    return HealthResponse(status="ok", detector_mode=detector_service.mode)


@app.post("/api/ocr", response_model=OCRResponse)
async def ocr_image(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Загрузите файл изображения.")

    image_bytes = await image.read()
    if len(image_bytes) > settings.upload_max_bytes:
        raise HTTPException(status_code=413, detail="Файл слишком большой.")

    try:
        text = recognize_image(image_bytes, lang=settings.ocr_lang)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return OCRResponse(text=text)


@app.post("/api/detect", response_model=DetectResponse)
def detect_text(
    payload: DetectRequest,
    db: Session = Depends(get_db),
    detector_service: Detector = Depends(get_detector),
):
    text = normalize_text(payload.text)

    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Текст слишком короткий для проверки.")
    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"Текст слишком длинный. Максимум: {settings.max_text_length} символов.",
        )

    probability = detector_service.predict_probability(text)
    verdict, confidence = make_verdict(probability)

    row = DetectionResult(
        text=text,
        ai_probability=probability,
        ai_percent=round(probability * 100, 2),
        verdict=verdict,
        confidence=round(confidence, 4),
        source="image_ocr_or_manual",
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return to_response(row)


@app.get("/api/results", response_model=list[ResultResponse])
def list_results(db: Session = Depends(get_db), limit: int = 20):
    limit = max(1, min(limit, 100))
    rows = (
        db.query(DetectionResult)
        .order_by(DetectionResult.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        ResultResponse(
            id=row.id,
            text=row.text,
            ai_probability=row.ai_probability,
            ai_percent=row.ai_percent,
            verdict=row.verdict,
            confidence=row.confidence,
            source=row.source,
            created_at=row.created_at,
        )
        for row in rows
    ]


@app.get("/api/results/{result_id}", response_model=ResultResponse)
def get_result(result_id: int, db: Session = Depends(get_db)):
    row = db.get(DetectionResult, result_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Результат не найден.")

    return ResultResponse(
        id=row.id,
        text=row.text,
        ai_probability=row.ai_probability,
        ai_percent=row.ai_percent,
        verdict=row.verdict,
        confidence=row.confidence,
        source=row.source,
        created_at=row.created_at,
    )
