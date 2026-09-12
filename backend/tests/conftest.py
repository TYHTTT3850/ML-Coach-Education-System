from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.mocks import MockEmbeddingClient, MockLLMClient


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        sqlite_path=tmp_path / "documents.db",
        upload_path=tmp_path / "uploads",
        admin_token="test-admin-token",
        embedding_provider="local",
        local_embedding_model="mock-model",
        chunk_size=240,
        chunk_overlap=40,
        rag_top_k=6,
        rag_strict_mode=True,
    )


@pytest.fixture
def mock_embedding() -> MockEmbeddingClient:
    return MockEmbeddingClient()


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def client(
    settings: Settings,
    mock_embedding: MockEmbeddingClient,
    mock_llm: MockLLMClient,
) -> Iterator[TestClient]:
    application = create_app(
        settings,
        embedding_client=mock_embedding,
        llm_client=mock_llm,
    )
    with TestClient(application) as test_client:
        yield test_client
