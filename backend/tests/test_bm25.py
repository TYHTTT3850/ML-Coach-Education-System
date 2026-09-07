import pytest

from app.rag.bm25 import BM25Index, tokenize


def test_tokenize_normalizes_latin_numbers_and_builds_chinese_ngrams() -> None:
    assert tokenize("ＢＭ２５ 3.14 逻辑回归") == [
        "bm25",
        "3.14",
        "逻辑",
        "辑回",
        "回归",
        "逻辑回",
        "辑回归",
    ]


def test_tokenize_keeps_single_chinese_concepts_searchable() -> None:
    assert tokenize("熵") == ["熵"]


def test_search_ranks_the_strongest_chinese_match_first() -> None:
    index = BM25Index(
        [
            "线性回归使用均方误差。",
            "逻辑回归常使用交叉熵损失函数。",
            "决策树通过信息增益选择特征。",
        ]
    )

    results = index.search("逻辑回归与交叉熵", limit=3)

    assert results[0][0] == 1
    assert all(score > 0 for _, score in results)


def test_search_omits_zero_scores_honors_limit_and_breaks_ties_by_index() -> None:
    index = BM25Index(["梯度下降", "梯度下降", "概率分布"])

    assert [document_index for document_index, _ in index.search("梯度", limit=1)] == [0]
    assert index.search("傅里叶", limit=3) == []
    assert index.search("梯度", limit=0) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"k1": 0}, "k1"), ({"b": -0.1}, "b"), ({"b": 1.1}, "b")],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BM25Index(["文档"], **kwargs)
