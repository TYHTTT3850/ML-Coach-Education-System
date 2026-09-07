from fastapi import APIRouter, Request

from app.schemas.health import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        llm_configured=settings.llm_configured,
        embedding_configured=settings.embedding_configured,
        vector_store_ready=bool(request.app.state.vector_store.ready),
    )
