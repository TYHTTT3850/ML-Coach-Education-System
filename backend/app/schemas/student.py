from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AnonymousStudentResponse(BaseModel):
    student_id: str
    created_at: datetime
