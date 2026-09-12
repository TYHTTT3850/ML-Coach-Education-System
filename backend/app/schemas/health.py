from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
