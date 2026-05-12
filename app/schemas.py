from datetime import date, datetime

from pydantic import BaseModel, Field


class OCRResponse(BaseModel):
    text: str


class OCRSegmentResponse(BaseModel):
    lines: list[str]
    line_count: int


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1)


class DetectResponse(BaseModel):
    id: int
    text: str
    ai_probability: float
    ai_percent: float
    verdict: str
    confidence: float
    created_at: datetime


class ResultResponse(DetectResponse):
    source: str


class HealthResponse(BaseModel):
    status: str
    detector_mode: str


class AuthConfigResponse(BaseModel):
    enabled: bool
    auth_url: str | None = None
    token_url: str | None = None
    client_id: str | None = None


class UserResponse(BaseModel):
    username: str
    display_name: str
    roles: list[str]
    is_admin: bool


class StudentResponse(BaseModel):
    id: int
    full_name: str


class ClassResponse(BaseModel):
    id: int
    name: str
    teacher_username: str
    students: list[StudentResponse] = []


class GradeCreateRequest(BaseModel):
    student_id: int
    class_id: int
    grade: int = Field(..., ge=1, le=5)
    work_date: date
    work_type: str = Field("essay", min_length=1, max_length=64)
    description: str | None = Field(None, max_length=220)
    detection_result_id: int | None = None


class GradeResponse(BaseModel):
    id: int
    student_id: int
    class_id: int
    detection_result_id: int | None
    work_date: date
    work_type: str
    description: str | None
    grade: int
    ai_percent: float | None
    created_at: datetime


class JournalStudentResponse(BaseModel):
    id: int
    full_name: str
    average_grade: float | None
    grades: list[GradeResponse]


class JournalResponse(BaseModel):
    class_id: int
    class_name: str
    dates: list[date]
    students: list[JournalStudentResponse]
