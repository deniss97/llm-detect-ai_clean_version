from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    ai_probability: Mapped[float] = mapped_column(Float, nullable=False)
    ai_percent: Mapped[float] = mapped_column(Float, nullable=False)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="image_ocr")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SchoolClass(Base):
    __tablename__ = "school_classes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    teacher_username: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    students: Mapped[list["Student"]] = relationship(
        back_populates="school_class",
        cascade="all, delete-orphan",
    )


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("school_classes.id"), nullable=False)

    school_class: Mapped[SchoolClass] = relationship(back_populates="students")
    grade_entries: Mapped[list["GradeEntry"]] = relationship(
        back_populates="student",
        cascade="all, delete-orphan",
    )


class GradeEntry(Base):
    __tablename__ = "grade_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False, index=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("school_classes.id"), nullable=False, index=True)
    detection_result_id: Mapped[int | None] = mapped_column(
        ForeignKey("detection_results.id"),
        nullable=True,
        index=True,
    )
    work_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    work_type: Mapped[str] = mapped_column(String(64), nullable=False, default="essay")
    description: Mapped[str | None] = mapped_column(String(220), nullable=True)
    grade: Mapped[int] = mapped_column(Integer, nullable=False)
    ai_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    student: Mapped[Student] = relationship(back_populates="grade_entries")
    school_class: Mapped[SchoolClass] = relationship()
    detection_result: Mapped[DetectionResult | None] = relationship()
