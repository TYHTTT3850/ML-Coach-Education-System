from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict


_TOKEN_PATTERN = re.compile(
    r"[a-z][a-z0-9_+.-]*"
    r"|\d+(?:\.\d+)?"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]+"
)

_ENGLISH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "explain",
    "for",
    "how",
    "is",
    "of",
    "please",
    "the",
    "this",
    "to",
    "what",
    "why",
}
_CHINESE_STOP_PHRASES = {
    "一下",
    "为什么",
    "是什么",
    "什么",
    "作用",
    "使用",
    "可以",
    "定义",
    "定理",
    "方法",
    "如何",
    "怎么",
    "概念",
    "过程",
    "结果",
    "意义",
    "是否",
    "本节",
    "材料",
    "知识",
    "课程",
    "解释",
    "表示",
    "证明",
    "请问",
    "这个",
    "那个",
    "问题",
}
_CHINESE_STOP_CHARACTERS = {
    "的",
    "了",
    "呢",
    "吗",
    "是",
    "与",
    "和",
    "及",
    "或",
    "在",
    "对",
    "中",
    "请",
}


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese and Latin course text for lexical retrieval.

    Latin identifiers and numbers are kept as whole tokens. Contiguous Chinese
    text is represented by overlapping character bigrams and trigrams, which
    avoids relying on a general-purpose dictionary for course-specific terms.
    A single Chinese character is retained as a fallback so one-character
    concepts such as ``熵`` remain searchable.
    """

    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        value = match.group(0)
        if "\u3400" <= value[0] <= "\u9fff":
            cleaned = value
            for phrase in sorted(_CHINESE_STOP_PHRASES, key=len, reverse=True):
                cleaned = cleaned.replace(phrase, " ")
            for character in _CHINESE_STOP_CHARACTERS:
                cleaned = cleaned.replace(character, " ")
            for sequence in cleaned.split():
                if len(sequence) == 1:
                    tokens.append(sequence)
                    continue
                for width in (2, 3):
                    tokens.extend(
                        sequence[index : index + width]
                        for index in range(len(sequence) - width + 1)
                    )
        else:
            if value not in _ENGLISH_STOP_WORDS:
                tokens.append(value)
    return tokens


class BM25Index:
    """Small, dependency-free BM25 index over an immutable document list."""

    def __init__(
        self,
        documents: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 必须大于 0")
        if not 0 <= b <= 1:
            raise ValueError("b 必须在 0 到 1 之间")

        self.k1 = float(k1)
        self.b = float(b)
        self._document_count = len(documents)
        self._document_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()

        total_length = 0
        for document_index, document in enumerate(documents):
            frequencies = Counter(tokenize(document))
            document_length = sum(frequencies.values())
            self._document_lengths.append(document_length)
            total_length += document_length
            document_frequency.update(frequencies.keys())
            for term, frequency in frequencies.items():
                self._postings[term].append((document_index, frequency))

        self._average_document_length = (
            total_length / self._document_count if self._document_count else 0.0
        )
        self._idf = {
            term: math.log(
                1.0
                + (self._document_count - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Return ``(document_index, score)`` pairs ordered by BM25 score.

        Results with no lexical match are omitted. Equal scores preserve the
        source document order through the document-index tie-breaker.
        """

        if limit <= 0 or not self._document_count:
            return []
        query_frequencies = Counter(tokenize(query))
        if not query_frequencies or self._average_document_length <= 0:
            return []

        scores = [0.0] * self._document_count
        for term, query_frequency in query_frequencies.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            for document_index, term_frequency in self._postings.get(term, ()):
                length_ratio = (
                    self._document_lengths[document_index]
                    / self._average_document_length
                )
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                scores[document_index] += (
                    query_frequency
                    * idf
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                )

        ranked = [
            (document_index, score)
            for document_index, score in enumerate(scores)
            if score > 0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:limit]
