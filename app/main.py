from pathlib import Path
import base64
import hashlib
import json
import logging
import secrets
from datetime import date, timedelta
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, init_db
from app.database import SessionLocal
from app.models import DetectionResult, GradeEntry, SchoolClass, Student
from app.schemas import (
    AuthConfigResponse,
    ClassResponse,
    DetectRequest,
    DetectResponse,
    GradeCreateRequest,
    GradeResponse,
    HealthResponse,
    JournalResponse,
    JournalStudentResponse,
    OCRResponse,
    ResultResponse,
    StudentResponse,
    UserResponse,
)
from app.services.detector import Detector, build_detector
from app.services.ocr import recognize_image, recognize_pdf


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
bearer_scheme = HTTPBearer(auto_error=False)
ACCESS_TOKEN_COOKIE = "ai_detector_access_token"
ID_TOKEN_COOKIE = "ai_detector_id_token"
OAUTH_STATE_COOKIE = "ai_detector_oauth_state"
OAUTH_VERIFIER_COOKIE = "ai_detector_pkce_verifier"
OAUTH_NEXT_COOKIE = "ai_detector_next"


class CurrentUser:
    def __init__(
        self,
        username: str,
        roles: list[str],
        is_admin: bool = False,
        display_name: str | None = None,
        identifiers: set[str] | None = None,
    ):
        self.username = username
        self.roles = roles
        self.is_admin = is_admin
        self.display_name = display_name or username
        self.identifiers = identifiers or {username}


@app.on_event("startup")
def on_startup() -> None:
    global detector
    logger.info("Инициализируем БД...")
    init_db()
    seed_demo_data()
    logger.info("Инициализируем детектор...")
    detector = build_detector(settings)
    logger.info("Приложение готово. Детектор: %s", detector.mode)


def get_detector() -> Detector:
    if detector is None:
        raise HTTPException(status_code=503, detail="Детектор ещё не загружен.")
    return detector


def keycloak_realm_url() -> str:
    base_url = settings.keycloak_public_base_url or settings.keycloak_base_url
    if not base_url or not settings.keycloak_realm:
        raise HTTPException(status_code=500, detail="Keycloak не настроен.")
    return f"{base_url.rstrip('/')}/realms/{settings.keycloak_realm}"


def keycloak_internal_realm_url() -> str:
    base_url = (
        settings.keycloak_internal_base_url
        or settings.keycloak_public_base_url
        or settings.keycloak_base_url
    )
    if not base_url or not settings.keycloak_realm:
        raise HTTPException(status_code=500, detail="Keycloak не настроен.")
    return f"{base_url.rstrip('/')}/realms/{settings.keycloak_realm}"


@lru_cache(maxsize=1)
def get_keycloak_jwks() -> dict[str, Any]:
    with urlopen(f"{keycloak_internal_realm_url()}/protocol/openid-connect/certs", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_roles(payload: dict[str, Any]) -> list[str]:
    realm_roles = payload.get("realm_access", {}).get("roles", [])
    client_id = settings.keycloak_client_id
    client_roles = []
    if client_id:
        client_roles = payload.get("resource_access", {}).get(client_id, {}).get("roles", [])
    return sorted(set(realm_roles + client_roles))


def configured_admin_usernames() -> set[str]:
    return {
        username.strip()
        for username in settings.admin_usernames.split(",")
        if username.strip()
    }


def token_to_current_user(token: str) -> CurrentUser:
    try:
        from jose import jwt
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Установите зависимость python-jose для проверки JWT Keycloak.",
        ) from exc

    issuer = keycloak_realm_url()
    audience = settings.keycloak_audience
    try:
        header = jwt.get_unverified_header(token)
        jwks = get_keycloak_jwks()
        signing_key = next(
            (key for key in jwks.get("keys", []) if key.get("kid") == header.get("kid")),
            None,
        )
        if signing_key is None:
            get_keycloak_jwks.cache_clear()
            jwks = get_keycloak_jwks()
            signing_key = next(
                (key for key in jwks.get("keys", []) if key.get("kid") == header.get("kid")),
                None,
            )
        if signing_key is None:
            raise HTTPException(status_code=401, detail="Ключ подписи Keycloak не найден.")

        payload = jwt.decode(
            token,
            signing_key,
            algorithms=[item.strip() for item in settings.keycloak_algorithms.split(",")],
            audience=audience,
            issuer=issuer,
            options={"verify_aud": bool(audience)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Недействительный токен Keycloak.") from exc

    roles = extract_roles(payload)
    username = payload.get("preferred_username") or payload.get("email") or payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="В токене нет имени пользователя.")

    display_name = payload.get("name") or " ".join(
        item for item in [payload.get("given_name"), payload.get("family_name")] if item
    )
    admin_names = configured_admin_usernames()
    identity_values = {
        item
        for item in [
            username,
            payload.get("preferred_username"),
            payload.get("email"),
            payload.get("sub"),
        ]
        if item
    }
    email = payload.get("email")
    if email and "@" in email:
        identity_values.add(email.split("@", 1)[0])
    is_admin = settings.admin_role in roles or bool(identity_values & admin_names)
    return CurrentUser(
        username=username,
        roles=roles,
        is_admin=is_admin,
        display_name=display_name or username,
        identifiers=identity_values,
    )


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> CurrentUser:
    if not settings.auth_enabled:
        return CurrentUser(
            username=settings.demo_teacher_username,
            roles=[settings.teacher_role],
            is_admin=False,
            identifiers={settings.demo_teacher_username},
        )

    token = credentials.credentials if credentials else request.cookies.get(ACCESS_TOKEN_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Требуется авторизация.")

    return token_to_current_user(token)


def pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def exchange_authorization_code(code: str, verifier: str, redirect_uri: str) -> dict[str, Any]:
    token_url = f"{keycloak_internal_realm_url()}/protocol/openid-connect/token"
    payload = urlencode(
        {
            "grant_type": "authorization_code",
            "client_id": settings.keycloak_client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }
    ).encode("utf-8")
    request = UrlRequest(
        token_url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.warning("Keycloak token exchange failed: %s", detail)
        raise HTTPException(status_code=401, detail="Keycloak не подтвердил код входа.") from exc


def set_session_cookies(response: Response, token_data: dict[str, Any]) -> None:
    max_age = int(token_data.get("expires_in") or 300)
    response.set_cookie(
        ACCESS_TOKEN_COOKIE,
        token_data["access_token"],
        max_age=max_age,
        httponly=True,
        samesite="lax",
    )
    if token_data.get("id_token"):
        response.set_cookie(
            ID_TOKEN_COOKIE,
            token_data["id_token"],
            max_age=max_age,
            httponly=True,
            samesite="lax",
        )


def clear_session_cookies(response: Response) -> None:
    for cookie_name in [
        ACCESS_TOKEN_COOKIE,
        ID_TOKEN_COOKIE,
        OAUTH_STATE_COOKIE,
        OAUTH_VERIFIER_COOKIE,
        OAUTH_NEXT_COOKIE,
    ]:
        response.delete_cookie(cookie_name)


def require_class_access(
    db: Session,
    class_id: int,
    user: CurrentUser,
) -> SchoolClass:
    school_class = db.get(SchoolClass, class_id)
    if school_class is None:
        raise HTTPException(status_code=404, detail="Класс не найден.")
    if not user.is_admin and school_class.teacher_username not in user.identifiers:
        raise HTTPException(status_code=403, detail="Этот класс недоступен текущему учителю.")
    return school_class


def seed_demo_data() -> None:
    demo_classes = {
        "8А": {
            "teacher_username": settings.demo_teacher_username,
            "students": [
                "Иванова Мария Дмитриевна",
                "Петрова Мария Дмитриевна",
                "Сидорова Мария Дмитриевна",
                "Иванова Иван Александрович",
                "Петрова Иван Александрович",
                "Сидорова Иван Александрович",
            ],
        },
        "9Б": {
            "teacher_username": "second.teacher",
            "students": [
                "Смирнова Анна Сергеевна",
            ],
        },
    }
    db = SessionLocal()
    try:
        for class_name, class_data in demo_classes.items():
            school_class = (
                db.query(SchoolClass)
                .filter(SchoolClass.name == class_name)
                .one_or_none()
            )
            if school_class is None:
                school_class = SchoolClass(
                    name=class_name,
                    teacher_username=class_data["teacher_username"],
                )
                db.add(school_class)
                db.flush()
            elif school_class.teacher_username != class_data["teacher_username"]:
                school_class.teacher_username = class_data["teacher_username"]

            existing_students = {
                student.full_name
                for student in db.query(Student).filter(Student.class_id == school_class.id)
            }
            for full_name in class_data["students"]:
                if full_name not in existing_students:
                    db.add(Student(full_name=full_name, class_id=school_class.id))

        db.commit()
        class_count = db.query(SchoolClass).count()
        student_count = db.query(Student).count()
        logger.info("Демо-данные БД: классов=%s, учеников=%s", class_count, student_count)
    finally:
        db.close()


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


def grade_to_response(row: GradeEntry) -> GradeResponse:
    return GradeResponse(
        id=row.id,
        student_id=row.student_id,
        class_id=row.class_id,
        detection_result_id=row.detection_result_id,
        work_date=row.work_date,
        work_type=row.work_type,
        description=row.description,
        grade=row.grade,
        ai_percent=row.ai_percent,
        created_at=row.created_at,
    )


@app.get("/login")
def login(request: Request, next_url: str = "/"):
    if not settings.auth_enabled:
        return RedirectResponse(url="/")

    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(64)
    redirect_uri = str(request.url_for("auth_callback"))
    params = urlencode(
        {
            "client_id": settings.keycloak_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid profile email",
            "state": state,
            "code_challenge": pkce_challenge(verifier),
            "code_challenge_method": "S256",
        }
    )
    response = RedirectResponse(
        url=f"{keycloak_realm_url()}/protocol/openid-connect/auth?{params}"
    )
    response.set_cookie(OAUTH_STATE_COOKIE, state, max_age=300, httponly=True, samesite="lax")
    response.set_cookie(
        OAUTH_VERIFIER_COOKIE,
        verifier,
        max_age=300,
        httponly=True,
        samesite="lax",
    )
    response.set_cookie(
        OAUTH_NEXT_COOKIE,
        next_url if next_url.startswith("/") else "/",
        max_age=300,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/auth/callback")
def auth_callback(request: Request, code: str | None = None, state: str | None = None):
    expected_state = request.cookies.get(OAUTH_STATE_COOKIE)
    verifier = request.cookies.get(OAUTH_VERIFIER_COOKIE)
    if not code or not state or not expected_state or state != expected_state or not verifier:
        response = RedirectResponse(url="/login")
        clear_session_cookies(response)
        return response

    token_data = exchange_authorization_code(code, verifier, str(request.url_for("auth_callback")))
    next_url = request.cookies.get(OAUTH_NEXT_COOKIE) or "/"
    response = RedirectResponse(url=next_url if next_url.startswith("/") else "/")
    clear_session_cookies(response)
    set_session_cookies(response, token_data)
    return response


@app.get("/logout")
def logout(request: Request):
    response = RedirectResponse(url="/")
    clear_session_cookies(response)
    if settings.auth_enabled:
        params = urlencode(
            {
                "client_id": settings.keycloak_client_id,
                "post_logout_redirect_uri": str(request.base_url),
            }
        )
        response = RedirectResponse(
            url=f"{keycloak_realm_url()}/protocol/openid-connect/logout?{params}"
        )
        clear_session_cookies(response)
    return response


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    current_user = None
    if settings.auth_enabled:
        token = request.cookies.get(ACCESS_TOKEN_COOKIE)
        if not token:
            return RedirectResponse(url="/login")
        try:
            current_user = token_to_current_user(token)
        except HTTPException:
            response = RedirectResponse(url="/login")
            clear_session_cookies(response)
            return response

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "threshold": settings.threshold,
            "detector_mode": detector.mode if detector else "loading",
            "current_user": current_user,
        },
    )


@app.get("/api/auth/config", response_model=AuthConfigResponse)
def auth_config():
    if not settings.auth_enabled:
        return AuthConfigResponse(enabled=False)

    realm_url = keycloak_realm_url()
    return AuthConfigResponse(
        enabled=True,
        auth_url=f"{realm_url}/protocol/openid-connect/auth",
        token_url=f"{realm_url}/protocol/openid-connect/token",
        client_id=settings.keycloak_client_id,
    )


@app.get("/api/me", response_model=UserResponse)
def me(user: CurrentUser = Depends(get_current_user)):
    return UserResponse(
        username=user.username,
        display_name=user.display_name,
        roles=user.roles,
        is_admin=user.is_admin,
    )


@app.get("/api/health", response_model=HealthResponse)
def health(detector_service: Detector = Depends(get_detector)):
    return HealthResponse(status="ok", detector_mode=detector_service.mode)


@app.get("/api/classes", response_model=list[ClassResponse])
def list_classes(
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    query = db.query(SchoolClass).order_by(SchoolClass.name)
    if not user.is_admin:
        query = query.filter(SchoolClass.teacher_username.in_(user.identifiers))

    school_classes = query.all()
    logger.info(
        "Classes request: user=%s identifiers=%s roles=%s is_admin=%s returned=%s",
        user.username,
        sorted(user.identifiers),
        user.roles,
        user.is_admin,
        len(school_classes),
    )

    return [
        ClassResponse(
            id=school_class.id,
            name=school_class.name,
            teacher_username=school_class.teacher_username,
            students=[
                StudentResponse(id=student.id, full_name=student.full_name)
                for student in sorted(school_class.students, key=lambda item: item.full_name)
            ],
        )
        for school_class in school_classes
    ]


@app.post("/api/grades", response_model=GradeResponse)
def create_grade(
    payload: GradeCreateRequest,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    school_class = require_class_access(db, payload.class_id, user)
    student = db.get(Student, payload.student_id)
    if student is None or student.class_id != school_class.id:
        raise HTTPException(status_code=400, detail="Ученик не найден в выбранном классе.")

    detection_result = None
    ai_percent = None
    if payload.detection_result_id is not None:
        detection_result = db.get(DetectionResult, payload.detection_result_id)
        if detection_result is None:
            raise HTTPException(status_code=404, detail="Результат проверки не найден.")
        ai_percent = detection_result.ai_percent

    row = GradeEntry(
        student_id=student.id,
        class_id=school_class.id,
        detection_result_id=detection_result.id if detection_result else None,
        work_date=payload.work_date,
        work_type=payload.work_type,
        description=payload.description,
        grade=payload.grade,
        ai_percent=ai_percent,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return grade_to_response(row)


@app.get("/api/journal/{class_id}", response_model=JournalResponse)
def get_journal(
    class_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
    date_from: date | None = None,
    date_to: date | None = None,
):
    school_class = require_class_access(db, class_id, user)
    if date_to is None:
        date_to = date.today()
    if date_from is None:
        date_from = date_to - timedelta(days=13)
    if date_from > date_to:
        raise HTTPException(status_code=400, detail="Дата начала периода позже даты конца.")
    if (date_to - date_from).days > 366:
        raise HTTPException(status_code=400, detail="Период журнала не должен превышать 1 год.")

    grades = (
        db.query(GradeEntry)
        .filter(GradeEntry.class_id == school_class.id)
        .filter(GradeEntry.work_date >= date_from)
        .filter(GradeEntry.work_date <= date_to)
        .order_by(GradeEntry.work_date, GradeEntry.created_at)
        .all()
    )
    visible_dates = sorted({row.work_date for row in grades if date_from <= row.work_date <= date_to})
    if not visible_dates:
        visible_dates = [date_from, date_to] if date_from != date_to else [date_from]

    students = sorted(school_class.students, key=lambda item: item.full_name)
    student_rows = []
    for student in students:
        student_grades = [row for row in grades if row.student_id == student.id]
        average = None
        if student_grades:
            average = round(sum(row.grade for row in student_grades) / len(student_grades), 2)
        student_rows.append(
            JournalStudentResponse(
                id=student.id,
                full_name=student.full_name,
                average_grade=average,
                grades=[grade_to_response(row) for row in student_grades],
            )
        )

    return JournalResponse(
        class_id=school_class.id,
        class_name=school_class.name,
        dates=visible_dates,
        students=student_rows,
    )


@app.post("/api/ocr", response_model=OCRResponse)
async def ocr_image(
    image: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    file_bytes = await image.read()
    if len(file_bytes) > settings.upload_max_bytes:
        raise HTTPException(status_code=413, detail="Файл слишком большой.")

    content_type = image.content_type or ""
    filename = (image.filename or "").lower()
    is_pdf = content_type == "application/pdf" or filename.endswith(".pdf")
    is_image = content_type.startswith("image/") or filename.endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    )
    if not is_pdf and not is_image:
        raise HTTPException(status_code=400, detail="Загрузите изображение или PDF-файл.")

    try:
        if is_pdf:
            text = recognize_pdf(
                file_bytes,
                lang=settings.ocr_lang,
                max_pages=settings.pdf_max_pages,
                render_dpi=settings.pdf_render_dpi,
            )
        else:
            text = recognize_image(file_bytes, lang=settings.ocr_lang)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return OCRResponse(text=text)


@app.post("/api/detect", response_model=DetectResponse)
def detect_text(
    payload: DetectRequest,
    db: Session = Depends(get_db),
    detector_service: Detector = Depends(get_detector),
    user: CurrentUser = Depends(get_current_user),
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
def list_results(
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
    limit: int = 20,
):
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
def get_result(
    result_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
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
