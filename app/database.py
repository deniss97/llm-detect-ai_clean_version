from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    db_path = settings.database_url.replace("sqlite:///", "")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app.models import DetectionResult, GradeEntry, SchoolClass, Student  # noqa: F401
    Base.metadata.create_all(bind=engine)
    ensure_school_class_access_columns()


def ensure_school_class_access_columns() -> None:
    inspector = inspect(engine)
    if not inspector.has_table("school_classes"):
        return

    columns = {column["name"] for column in inspector.get_columns("school_classes")}
    statements = []
    if "owner_subject" not in columns:
        statements.append("ALTER TABLE school_classes ADD COLUMN owner_subject VARCHAR(128)")
    if "access_key" not in columns:
        statements.append("ALTER TABLE school_classes ADD COLUMN access_key VARCHAR(128)")

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
        connection.execute(text("UPDATE school_classes SET access_key = name WHERE access_key IS NULL"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
