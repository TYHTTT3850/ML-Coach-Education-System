from fastapi import APIRouter, Request

from app.schemas.health import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        vector_store_ready=bool(request.app.state.vector_store.ready),
    )
