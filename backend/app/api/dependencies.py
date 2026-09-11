from __future__ import annotations

from fastapi import Depends, Header, Request

from app.core.config import Settings
from app.rag.hyde import HyDEGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.prompt_builder import PromptBuilder
from app.rag.retriever import Retriever
from app.services.embedding.base import EmbeddingClient
from app.services.embedding.factory import create_embedding_client
from app.services.evaluation_service import EvaluationService
from app.services.knowledge_service import KnowledgeService
from app.services.learning_grounding import LearningGroundingService
from app.services.learning_repository import LearningRepository
from app.services.learning_service import LearningService
from app.services.llm.base import LLMClient
from app.services.llm.factory import LazyLLMClient
from app.services.practice_service import PracticeService
from app.services.remediation_service import RemediationService


def settings_from_request(request: Request) -> Settings:
    return request.app.state.settings


def embedding_from_request(request: Request) -> EmbeddingClient:
    if request.app.state.embedding_client is None:
        request.app.state.embedding_client = create_embedding_client(request.app.state.settings)
    return request.app.state.embedding_client


def llm_from_request(request: Request) -> LLMClient:
    if request.app.state.llm_client is None:
        request.app.state.llm_client = LazyLLMClient(request.app.state.settings)
    return request.app.state.llm_client


def learning_repository_from_request(request: Request) -> LearningRepository:
    """Return the single application-wide learning persistence boundary."""

    return request.app.state.learning_repository


def require_student_from_header(
    authorization: str = Header(..., alias="Authorization"),
    repository: LearningRepository = Depends(learning_repository_from_request),
) -> dict:
    """Resolve the authenticated student from the opaque Bearer session token."""

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        from app.core.exceptions import StudentNotFoundError
        raise StudentNotFoundError()
    return repository.student_from_token(token, touch=True)


def student_id_from_authorization(
    authorization: str | None,
    repository: LearningRepository,
    *,
    touch: bool = True,
) -> str | None:
    """Return a session's internal student ID, preserving anonymous read routes."""

    if authorization is None:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        from app.core.exceptions import StudentNotFoundError
        raise StudentNotFoundError()
    return repository.student_from_token(token, touch=touch)["id"]


def knowledge_service(
    request: Request,
    embedding_client: EmbeddingClient = Depends(embedding_from_request),
) -> KnowledgeService:
    return KnowledgeService(
        settings=request.app.state.settings,
        registry=request.app.state.registry,
        vector_store=request.app.state.vector_store,
        embedding_client=embedding_client,
    )


def build_rag_pipeline(request: Request) -> RAGPipeline:
    """Build the retrieval pipeline only after the route validates the request."""
    settings = request.app.state.settings
    embedding_client = embedding_from_request(request)
    llm_client = llm_from_request(request)
    retriever = Retriever(
        request.app.state.vector_store,
        embedding_client,
        top_k=settings.rag_top_k,
        distance_threshold=settings.rag_distance_threshold,
    )
    return RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        prompt_builder=PromptBuilder(),
        strict_mode=settings.rag_strict_mode,
        temperature=settings.llm_temperature,
        hyde_enabled=settings.rag_hyde_enabled,
    )


def learning_grounding_from_request(
    request: Request,
    embedding_client: EmbeddingClient = Depends(embedding_from_request),
    llm_client: LLMClient = Depends(llm_from_request),
) -> LearningGroundingService:
    """Reuse the Phase 1 Retriever and HyDE implementation for learning flows."""

    settings = request.app.state.settings
    retriever = Retriever(
        request.app.state.vector_store,
        embedding_client,
        top_k=settings.rag_top_k,
        distance_threshold=settings.rag_distance_threshold,
    )
    return LearningGroundingService(
        retriever=retriever,
        hyde_generator=HyDEGenerator(
            llm_client,
            temperature=min(settings.practice_generation_temperature, 0.3),
        ),
        strict_mode=settings.rag_strict_mode,
        hyde_enabled=settings.rag_hyde_enabled,
    )


def practice_service_from_request(
    settings: Settings = Depends(settings_from_request),
    repository: LearningRepository = Depends(learning_repository_from_request),
    grounding: LearningGroundingService = Depends(learning_grounding_from_request),
    llm_client: LLMClient = Depends(llm_from_request),
) -> PracticeService:
    return PracticeService(
        settings=settings,
        repository=repository,
        grounding=grounding,
        llm_client=llm_client,
    )


def evaluation_service_from_request(
    settings: Settings = Depends(settings_from_request),
    repository: LearningRepository = Depends(learning_repository_from_request),
    llm_client: LLMClient = Depends(llm_from_request),
) -> EvaluationService:
    return EvaluationService(
        settings=settings,
        repository=repository,
        llm_client=llm_client,
    )


def remediation_service_from_request(
    settings: Settings = Depends(settings_from_request),
    repository: LearningRepository = Depends(learning_repository_from_request),
    llm_client: LLMClient = Depends(llm_from_request),
) -> RemediationService:
    return RemediationService(
        settings=settings,
        repository=repository,
        llm_client=llm_client,
    )


def learning_service_from_request(
    settings: Settings = Depends(settings_from_request),
    repository: LearningRepository = Depends(learning_repository_from_request),
) -> LearningService:
    return LearningService(settings=settings, repository=repository)
