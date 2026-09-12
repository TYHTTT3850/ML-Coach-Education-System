from __future__ import annotations

import logging
import inspect
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.chat import router as chat_router
from app.api.curriculum import router as curriculum_router
from app.api.health import router as health_router
from app.api.knowledge import router as knowledge_router
from app.api.learning import router as learning_router
from app.api.practice import router as practice_router
from app.api.students import router as students_router
from app.api.content import router as content_router
from app.api.course_content import router as course_content_router
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, VectorStoreError
from app.core.logging import configure_logging
from app.curriculum.seed import seed_curriculum
from app.rag.vector_store import PgVectorStore
from app.services.document_registry import DocumentRegistry
from app.services.embedding.factory import create_embedding_client
from app.services.learning_repository import LearningRepository
from app.services.llm.factory import LazyLLMClient


logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    llm_client: Any = None,
    embedding_client: Any = None,
    vector_store: Any = None,
    registry: DocumentRegistry | None = None,
    learning_repository: LearningRepository | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(
        app_settings.app_log_level,
        (
            app_settings.admin_token.get_secret_value(),
        ),
    )
    app_registry = registry or DocumentRegistry(app_settings.database_url)
    app_learning_repository = learning_repository or LearningRepository(
        app_settings.database_url
    )
    app_vector_store = vector_store or PgVectorStore(
        app_settings.database_url,
        app_settings.vector_distance_metric,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        app_settings.ensure_data_directories()
        # The bundled curriculum is configuration data.  Its upsert is
        # intentionally idempotent; database schema must be initialized first
        # with operations/database/initialize.py.
        seed_curriculum(app_learning_repository)
        if not getattr(app_vector_store, "ready", False):
            try:
                app_vector_store.initialize()
            except VectorStoreError:
                logger.error("向量库启动失败；健康检查将报告未就绪")
        try:
            yield
        finally:
            for client_name in ("llm_client", "embedding_client"):
                client = getattr(application.state, client_name, None)
                close = getattr(client, "close", None)
                if close is None:
                    continue
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.exception("关闭 %s 失败", client_name)

    app = FastAPI(
        title="Course Coach API",
        version=__version__,
        description="机器学习与高等数学课程问答 Agent 后端",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.registry = app_registry
    app.state.learning_repository = app_learning_repository
    app.state.vector_store = app_vector_store
    app.state.embedding_client = embedding_client
    app.state.llm_client = llm_client

    @app.middleware("http")
    async def user_model_clients(request: Request, call_next):
        llm_api_key = request.headers.get("X-LLM-API-Key", "").strip()
        embedding_api_key = request.headers.get("X-Embedding-API-Key", "").strip()
        llm_base_url = request.headers.get("X-LLM-Base-URL", "").strip()
        embedding_base_url = request.headers.get("X-Embedding-Base-URL", "").strip()
        llm_model = request.headers.get("X-LLM-Model", "").strip()
        embedding_model = request.headers.get("X-Embedding-Model", "").strip()
        if llm_api_key:
            request.state.llm_client = LazyLLMClient(app_settings, llm_api_key, llm_base_url, llm_model)
        if embedding_api_key:
            request.state.embedding_client = create_embedding_client(app_settings, embedding_api_key, embedding_base_url, embedding_model)
        try:
            return await call_next(request)
        finally:
            for client_name in ("llm_client", "embedding_client"):
                client = getattr(request.state, client_name, None)
                close = getattr(client, "close", None)
                if close is None:
                    continue
                result = close()
                if inspect.isawaitable(result):
                    await result

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Admin-Token", "X-LLM-API-Key", "X-Embedding-API-Key", "X-LLM-Base-URL", "X-Embedding-Base-URL", "X-LLM-Model", "X-Embedding-Model"],
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("应用服务异常 [%s]: %s", exc.code, exc.internal_message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info("请求参数校验失败: %s", exc)
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "请求参数无效"}},
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": str(exc.detail)}},
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理的服务器异常: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "服务器暂时无法处理请求"}},
        )

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(knowledge_router)
    app.include_router(students_router)
    app.include_router(curriculum_router)
    app.include_router(practice_router)
    app.include_router(learning_router)
    app.include_router(content_router)
    app.include_router(course_content_router)
    return app


app = create_app()
