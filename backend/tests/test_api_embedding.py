from types import SimpleNamespace

from app.services.embedding.api_embedding import APIEmbeddingClient


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        assert isinstance(inputs, list)
        data = [
            SimpleNamespace(
                index=index,
                embedding=[float(text.removeprefix("text-")), 0.0],
            )
            for index, text in reversed(list(enumerate(inputs)))
        ]
        return SimpleNamespace(data=data)


def test_api_embedding_splits_large_inputs_and_preserves_order() -> None:
    embeddings = FakeEmbeddings()
    client = APIEmbeddingClient.__new__(APIEmbeddingClient)
    client.client = SimpleNamespace(embeddings=embeddings)
    client.model = "text-embedding-v4"
    client.dimension = 2

    vectors = client.embed_documents([f"text-{index}" for index in range(23)])

    assert [len(call["input"]) for call in embeddings.calls] == [10, 10, 3]
    assert all(call["model"] == "text-embedding-v4" for call in embeddings.calls)
    assert all(call["dimensions"] == 2 for call in embeddings.calls)
    assert vectors == [[float(index), 0.0] for index in range(23)]
