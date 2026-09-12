from app.core.config import Settings


def test_env_example_allows_blank_optional_numbers(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EMBEDDING_DIMENSION=\nRAG_DISTANCE_THRESHOLD=\nVECTOR_DISTANCE_METRIC=cosine\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file, app_env="test")
    assert settings.embedding_dimension is None
    assert settings.rag_distance_threshold is None
