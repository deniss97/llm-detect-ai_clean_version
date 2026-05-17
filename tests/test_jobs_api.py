import os
from pathlib import Path


os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = "sqlite:///./data/test_jobs.db"
os.environ["USE_MOCK_DETECTOR"] = "true"
os.environ["OCR_PRELOAD_ON_STARTUP"] = "false"

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.database import init_db  # noqa: E402
from app.main import app  # noqa: E402


class FakeAsyncResult:
    id = "celery-test-task-id"


def setup_module() -> None:
    Path("./data/test_jobs.db").unlink(missing_ok=True)
    init_db()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client


@pytest.mark.anyio
async def test_health_response_structure(client):
    response = await client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "detector_mode" in payload


@pytest.mark.anyio
async def test_create_detect_job(monkeypatch, client):
    def fake_enqueue(job_id: int, text: str) -> FakeAsyncResult:
        assert job_id > 0
        assert len(text) >= 20
        return FakeAsyncResult()

    monkeypatch.setattr("app.main.enqueue_detection_job", fake_enqueue)

    response = await client.post(
        "/api/detect/jobs",
        json={"text": "Это достаточно длинный текст для проверки через очередь."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] > 0
    assert payload["status"] == "queued"
    assert payload["task_type"] == "detect"


@pytest.mark.anyio
async def test_get_job_status(monkeypatch, client):
    monkeypatch.setattr("app.main.enqueue_detection_job", lambda job_id, text: FakeAsyncResult())

    created = (
        await client.post(
            "/api/detect/jobs",
            json={"text": "Еще один достаточно длинный текст для постановки задачи."},
        )
    ).json()
    response = await client.get(f"/api/jobs/{created['job_id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == created["job_id"]
    assert payload["task_id"] == "celery-test-task-id"
    assert payload["task_type"] == "detect"
    assert payload["status"] == "queued"


@pytest.mark.anyio
async def test_detect_job_rejects_short_text(monkeypatch, client):
    called = False

    def fake_enqueue(job_id: int, text: str) -> FakeAsyncResult:
        nonlocal called
        called = True
        return FakeAsyncResult()

    monkeypatch.setattr("app.main.enqueue_detection_job", fake_enqueue)

    response = await client.post("/api/detect/jobs", json={"text": "коротко"})

    assert response.status_code == 400
    assert "слишком короткий" in response.json()["detail"]
    assert called is False
