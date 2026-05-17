import logging
from typing import Any, Callable

from app.config import settings


logger = logging.getLogger(__name__)


try:
    from celery import Celery
except ImportError:  # pragma: no cover - used only before dependencies are installed
    Celery = None  # type: ignore[assignment]


class MissingCeleryTask:
    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.__name__ = func.__name__

    def delay(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Celery не установлен. Установите зависимости из requirements.txt.")

    apply_async = delay


class MissingCeleryApp:
    def task(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], MissingCeleryTask]:
        def decorator(func: Callable[..., Any]) -> MissingCeleryTask:
            return MissingCeleryTask(func)

        return decorator


broker_url = settings.celery_broker_url or settings.redis_url
result_backend = settings.celery_result_backend or settings.redis_url

if Celery is None:
    celery_app = MissingCeleryApp()
else:
    celery_app = Celery(
        "ai_detector",
        broker=broker_url,
        backend=result_backend,
        include=["app.tasks"],
    )
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
    )
