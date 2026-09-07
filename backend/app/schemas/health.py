from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    embedding_configured: bool
    vector_store_ready: bool
