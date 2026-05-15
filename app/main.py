from pathlib import Path
import base64
import csv
import hashlib
from io import StringIO
import json
import logging
import secrets
from threading import Lock, Thread
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
from sqlalchemy import or_

from app.config import settings
from app.database import get_db, init_db
from app.database import SessionLocal
from app.models import DetectionResult, GradeEntry, SchoolClass, Student
from app.schemas import (
    AuthConfigResponse,
    ClassImportResponse,
    ClassResponse,
    DetectRequest,
    DetectResponse,
    GradeCreateRequest,
    GradeResponse,
    HealthResponse,
    JournalResponse,
    JournalStudentResponse,
    OCRResponse,
    OCRSegmentResponse,
    ResultResponse,
    StudentResponse,
    UserResponse,
)
from app.services.detector import Detector, build_detector
from app.services.ocr import (
    get_recognizer,
    recognize_image,
    recognize_pdf,
    segment_image_previews,
    segment_pdf_previews,
)


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
detector_lock = Lock()
model_status_lock = Lock()
detector_status: dict[str, str | None] = {
    "state": "pending",
    "message": "Детектор ожидает предзагрузки.",
    "mode": None,
}
ocr_status: dict[str, str | None] = {
    "state": "pending" if settings.ocr_preload_on_startup else "disabled",
    "message": (
        "OCR-модель ожидает предзагрузки."
        if settings.ocr_preload_on_startup
        else "OCR-модель загрузится при первом распознавании."
    ),
    "mode": None,
}
bearer_scheme = HTTPBearer(auto_error=False)
ACCESS_TOKEN_COOKIE = "ai_detector_access_token"
ID_TOKEN_COOKIE = "ai_detector_id_token"
REFRESH_TOKEN_COOKIE = "ai_detector_refresh_token"
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
        subject: str | None = None,
        class_access_keys: set[str] | None = None,
    ):
        self.username = username
        self.roles = roles
        self.is_admin = is_admin
        self.display_name = display_name or username
        self.identifiers = identifiers or {username}
        self.subject = subject or username
        self.class_access_keys = class_access_keys or set()


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Инициализируем БД...")
    init_db()
    seed_demo_data()
    logger.info("Запускаем фоновую предзагрузку моделей.")
    Thread(target=preload_models, name="model-preload", daemon=True).start()
    logger.info("Приложение готово. Статус предзагрузки доступен в /api/model-status.")


def set_model_status(component: str, state: str, message: str, mode: str | None = None) -> None:
    target = detector_status if component == "detector" else ocr_status
    with model_status_lock:
        target["state"] = state
        target["message"] = message
        if mode is not None:
            target["mode"] = mode


def preload_detector() -> None:
    global detector
    set_model_status("detector", "loading", "Загружаем детектор AI-текста...")
    logger.info("Предзагружаем детектор...")
    try:
        with detector_lock:
            if detector is None:
                detector = build_detector(settings)
            mode = detector.mode
        set_model_status("detector", "ready", "Детектор готов к проверке.", mode)
        logger.info("Детектор предзагружен: %s", mode)
    except Exception:
        logger.exception("Не удалось предзагрузить детектор.")
        set_model_status("detector", "error", "Не удалось загрузить детектор. Проверьте логи контейнера.")


def preload_ocr() -> None:
    if not settings.ocr_preload_on_startup:
        return
    set_model_status("ocr", "loading", "Загружаем OCR-модель...")
    logger.info("Предзагружаем OCR-модель %s...", settings.ocr_model_name)
    try:
        recognizer = get_recognizer(settings.ocr_model_name, settings.ocr_model_local_files_only)
        set_model_status("ocr", "ready", "OCR-модель готова к распознаванию.", recognizer.mode)
        logger.info("OCR-модель предзагружена.")
    except Exception:
        logger.exception("Не удалось предзагрузить OCR-модель.")
        set_model_status("ocr", "error", "Не удалось загрузить OCR-модель. Проверьте логи контейнера.")


def preload_models() -> None:
    preload_detector()
    preload_ocr()


def get_detector() -> Detector:
    global detector
    if detector is None:
        with model_status_lock:
            state = detector_status["state"]
        if state == "loading":
            raise HTTPException(status_code=503, detail="Детектор ещё загружается. Подождите завершения предзагрузки.")
        with detector_lock:
            if detector is None:
                logger.info("Инициализируем детектор при первом запросе...")
                set_model_status("detector", "loading", "Загружаем детектор AI-текста...")
                try:
                    detector = build_detector(settings)
                except Exception:
                    logger.exception("Не удалось инициализировать детектор.")
                    set_model_status("detector", "error", "Не удалось загрузить детектор. Проверьте логи контейнера.")
                    raise HTTPException(status_code=503, detail="Детектор не удалось загрузить.")
                set_model_status("detector", "ready", "Детектор готов к проверке.", detector.mode)
                logger.info("Детектор загружен: %s", detector.mode)
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


def claim_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.replace(";", ",").split(",") if item.strip()}
    if isinstance(value, list | tuple | set):
        result = set()
        for item in value:
            result.update(claim_values(item))
        return result
    return {str(value).strip()} if str(value).strip() else set()


def extract_class_access_keys(payload: dict[str, Any]) -> set[str]:
    claim_name = settings.class_access_claim
    keys = claim_values(payload.get(claim_name))
    attributes = payload.get("attributes")
    if isinstance(attributes, dict):
        keys.update(claim_values(attributes.get(claim_name)))
    return keys


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
    subject = payload.get("sub") or username
    return CurrentUser(
        username=username,
        roles=roles,
        is_admin=is_admin,
        display_name=display_name or username,
        identifiers=identity_values,
        subject=subject,
        class_access_keys=extract_class_access_keys(payload),
    )


def get_current_user(
    request: Request,
    response: Response,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> CurrentUser:
    if not settings.auth_enabled:
        return CurrentUser(
            username=settings.demo_teacher_username,
            roles=[settings.teacher_role],
            is_admin=False,
            identifiers={settings.demo_teacher_username},
            subject=settings.demo_teacher_username,
            class_access_keys={"8А"},
        )

    token = credentials.credentials if credentials else request.cookies.get(ACCESS_TOKEN_COOKIE)
    if not token:
        refresh_token = request.cookies.get(REFRESH_TOKEN_COOKIE)
        if refresh_token and not credentials:
            try:
                token_data = refresh_keycloak_session(refresh_token)
                set_session_cookies(response, token_data)
                return token_to_current_user(token_data["access_token"])
            except HTTPException:
                clear_session_cookies(response)
        raise HTTPException(status_code=401, detail="Требуется авторизация.")

    try:
        return token_to_current_user(token)
    except HTTPException as exc:
        refresh_token = request.cookies.get(REFRESH_TOKEN_COOKIE)
        if refresh_token and not credentials:
            try:
                token_data = refresh_keycloak_session(refresh_token)
                set_session_cookies(response, token_data)
                return token_to_current_user(token_data["access_token"])
            except HTTPException:
                clear_session_cookies(response)
        raise HTTPException(status_code=401, detail="Сессия истекла.") from exc


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


def refresh_keycloak_session(refresh_token: str) -> dict[str, Any]:
    token_url = f"{keycloak_internal_realm_url()}/protocol/openid-connect/token"
    payload = urlencode(
        {
            "grant_type": "refresh_token",
            "client_id": settings.keycloak_client_id,
            "refresh_token": refresh_token,
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
        logger.info("Keycloak refresh failed: %s", detail)
        raise HTTPException(status_code=401, detail="Сессия истекла.") from exc


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
    if token_data.get("refresh_token"):
        refresh_max_age = int(token_data.get("refresh_expires_in") or max_age)
        response.set_cookie(
            REFRESH_TOKEN_COOKIE,
            token_data["refresh_token"],
            max_age=refresh_max_age,
            httponly=True,
            samesite="lax",
        )


def clear_session_cookies(response: Response) -> None:
    for cookie_name in [
        ACCESS_TOKEN_COOKIE,
        ID_TOKEN_COOKIE,
        REFRESH_TOKEN_COOKIE,
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
    if not user_can_access_class(user, school_class):
        raise HTTPException(status_code=403, detail="Этот класс недоступен текущему учителю.")
    return school_class


def user_can_access_class(user: CurrentUser, school_class: SchoolClass) -> bool:
    if user.is_admin:
        return True
    if school_class.owner_subject and school_class.owner_subject == user.subject:
        return True
    if school_class.access_key and school_class.access_key in user.class_access_keys:
        return True
    return False


def normalize_csv_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def pick_csv_value(row: dict[str, str], keys: set[str]) -> str:
    normalized = {normalize_csv_key(key): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(key)
        if value and value.strip():
            return value.strip()
    return ""


def parse_classes_csv(file_bytes: bytes) -> dict[str, list[str]]:
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_bytes.decode("cp1251")

    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
    except csv.Error:
        dialect = csv.excel

    class_keys = {"class", "class_name", "klass", "класс", "название_класса"}
    student_keys = {"student", "student_name", "full_name", "name", "ученик", "фио", "фамилия_имя"}
    dict_reader = csv.DictReader(StringIO(text), dialect=dialect)
    fieldnames = {normalize_csv_key(field or "") for field in (dict_reader.fieldnames or [])}
    has_known_headers = bool(fieldnames & class_keys) and bool(fieldnames & student_keys)

    if has_known_headers:
        rows = list(dict_reader)
    else:
        reader = csv.reader(StringIO(text), dialect=dialect)
        rows = [
            {"class": row[0], "student": row[1]}
            for row in reader
            if len(row) >= 2 and row[0].strip() and row[1].strip()
        ]

    grouped: dict[str, list[str]] = {}
    for row in rows:
        class_name = pick_csv_value(row, class_keys)
        student_name = pick_csv_value(row, student_keys)
        if not class_name or not student_name:
            continue
        grouped.setdefault(class_name, [])
        if student_name not in grouped[class_name]:
            grouped[class_name].append(student_name)

    if not grouped:
        raise ValueError(
            "CSV должен содержать колонки class/class_name/класс и student/full_name/фио."
        )
    return grouped


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
                    access_key=class_name,
                )
                db.add(school_class)
                db.flush()
            elif school_class.teacher_username != class_data["teacher_username"]:
                school_class.teacher_username = class_data["teacher_username"]
            if not school_class.access_key:
                school_class.access_key = class_name

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
    refreshed_token_data = None
    if settings.auth_enabled:
        token = request.cookies.get(ACCESS_TOKEN_COOKIE)
        refresh_token = request.cookies.get(REFRESH_TOKEN_COOKIE)
        if not token:
            if refresh_token:
                try:
                    refreshed_token_data = refresh_keycloak_session(refresh_token)
                    current_user = token_to_current_user(refreshed_token_data["access_token"])
                except HTTPException:
                    response = RedirectResponse(url="/login")
                    clear_session_cookies(response)
                    return response
            else:
                return RedirectResponse(url="/login")
        try:
            if current_user is None and token:
                current_user = token_to_current_user(token)
        except HTTPException:
            if refresh_token:
                try:
                    refreshed_token_data = refresh_keycloak_session(refresh_token)
                    current_user = token_to_current_user(refreshed_token_data["access_token"])
                except HTTPException:
                    response = RedirectResponse(url="/login")
                    clear_session_cookies(response)
                    return response
            else:
                response = RedirectResponse(url="/login")
                clear_session_cookies(response)
                return response

    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "threshold": settings.threshold,
            "detector_mode": detector.mode if detector else "not_loaded",
            "current_user": current_user,
        },
    )
    if refreshed_token_data:
        set_session_cookies(response, refreshed_token_data)
    return response


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
def health():
    return HealthResponse(status="ok", detector_mode=detector.mode if detector else "not_loaded")


@app.get("/api/model-status")
def model_status():
    with model_status_lock:
        return {
            "detector": dict(detector_status),
            "ocr": dict(ocr_status),
            "ready": detector_status["state"] == "ready",
        }


@app.get("/api/classes", response_model=list[ClassResponse])
def list_classes(
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    query = db.query(SchoolClass).order_by(SchoolClass.name)
    if not user.is_admin:
        conditions = [SchoolClass.owner_subject == user.subject]
        if user.class_access_keys:
            conditions.append(SchoolClass.access_key.in_(user.class_access_keys))
        query = query.filter(or_(*conditions))

    school_classes = query.all()
    logger.info(
        "Classes request: user=%s subject=%s class_access_keys=%s roles=%s is_admin=%s returned=%s",
        user.username,
        user.subject,
        sorted(user.class_access_keys),
        user.roles,
        user.is_admin,
        len(school_classes),
    )

    return [
        ClassResponse(
            id=school_class.id,
            name=school_class.name,
            teacher_username=school_class.teacher_username,
            owner_subject=school_class.owner_subject,
            access_key=school_class.access_key,
            students=[
                StudentResponse(id=student.id, full_name=student.full_name)
                for student in sorted(school_class.students, key=lambda item: item.full_name)
            ],
        )
        for school_class in school_classes
    ]


@app.post("/api/classes/import-csv", response_model=ClassImportResponse)
async def import_classes_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    filename = (file.filename or "").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Загрузите CSV-файл.")

    file_bytes = await file.read()
    if len(file_bytes) > settings.upload_max_bytes:
        raise HTTPException(status_code=413, detail="Файл слишком большой.")

    try:
        grouped = parse_classes_csv(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    classes_created = 0
    classes_updated = 0
    students_created = 0
    skipped_students = 0
    class_ids: list[int] = []

    for class_name, student_names in grouped.items():
        school_class = (
            db.query(SchoolClass)
            .filter(SchoolClass.name == class_name)
            .one_or_none()
        )
        if school_class is None:
            school_class = SchoolClass(
                name=class_name,
                teacher_username=user.username,
                owner_subject=user.subject,
                access_key=class_name,
            )
            db.add(school_class)
            db.flush()
            classes_created += 1
        else:
            if not user_can_access_class(user, school_class):
                raise HTTPException(
                    status_code=409,
                    detail=f"Класс '{class_name}' уже существует, но недоступен текущему пользователю.",
                )
            classes_updated += 1
            if not school_class.access_key:
                school_class.access_key = class_name

        existing_students = {
            student.full_name
            for student in db.query(Student).filter(Student.class_id == school_class.id)
        }
        for student_name in student_names:
            if student_name in existing_students:
                skipped_students += 1
                continue
            db.add(Student(full_name=student_name, class_id=school_class.id))
            existing_students.add(student_name)
            students_created += 1

        class_ids.append(school_class.id)

    db.commit()
    return ClassImportResponse(
        classes_created=classes_created,
        classes_updated=classes_updated,
        students_created=students_created,
        skipped_students=skipped_students,
        class_ids=class_ids,
    )


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
    min_line_height: int | None = None,
    line_threshold_ratio: float | None = None,
    line_padding: int | None = None,
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
        line_options = {
            "min_line_height": min_line_height if min_line_height is not None else settings.ocr_min_line_height,
            "line_threshold_ratio": (
                line_threshold_ratio
                if line_threshold_ratio is not None
                else settings.ocr_line_threshold_ratio
            ),
            "line_padding": line_padding if line_padding is not None else settings.ocr_line_padding,
        }
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return OCRResponse(text=text)


@app.post("/api/ocr/segment", response_model=OCRSegmentResponse)
async def segment_ocr_image(
    image: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
    min_line_height: int | None = None,
    line_threshold_ratio: float | None = None,
    line_padding: int | None = None,
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
        line_options = {
            "min_line_height": min_line_height if min_line_height is not None else settings.ocr_min_line_height,
            "line_threshold_ratio": (
                line_threshold_ratio
                if line_threshold_ratio is not None
                else settings.ocr_line_threshold_ratio
            ),
            "line_padding": line_padding if line_padding is not None else settings.ocr_line_padding,
        }
        if is_pdf:
            pages = segment_pdf_previews(
                file_bytes,
                max_pages=settings.pdf_max_pages,
                render_dpi=settings.pdf_render_dpi,
                **line_options,
            )
        else:
            lines = segment_image_previews(file_bytes, **line_options)
            pages = [{"page": 1, "lines": lines, "line_count": len(lines)}]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return OCRSegmentResponse(
        pages=pages,
        line_count=sum(int(page["line_count"]) for page in pages),
    )


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
