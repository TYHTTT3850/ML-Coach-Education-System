from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from app.core.types import Course
from app.rag.loaders import ParsedDocument, ParsedSection


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


class StructuredChunker:
    """Structure-aware character chunker with boundary-aware overlap."""

    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size 必须为正数，且 overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(
        self,
        document: ParsedDocument,
        *,
        document_id: str,
        course: Course,
        file_hash: str,
    ) -> list[Chunk]:
        pieces: list[tuple[str, ParsedSection]] = []
        pending_text = ""
        pending_section: ParsedSection | None = None

        def flush_pending() -> None:
            nonlocal pending_text, pending_section
            if pending_text.strip() and pending_section is not None:
                for piece in self._split_long(pending_text.strip()):
                    if piece.strip():
                        pieces.append((piece.strip(), pending_section))
            pending_text = ""
            pending_section = None

        for source_section in document.sections:
            text = source_section.text.strip()
            if not text:
                continue
            same_context = pending_section is not None and self._same_context(
                pending_section, source_section
            )
            combined = f"{pending_text}\n\n{text}".strip() if pending_text else text
            if same_context and len(combined) <= self.chunk_size:
                pending_text = combined
                continue
            flush_pending()
            pending_text = text
            pending_section = source_section
        flush_pending()

        chunks: list[Chunk] = []
        for chunk_index, (text, source_section) in enumerate(pieces):
            chunk_id = hashlib.sha256(
                f"{course.value}:{document_id}:{chunk_index}:{text}".encode("utf-8")
            ).hexdigest()
            metadata: dict[str, Any] = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "course": course.value,
                "source_file": document.source_file,
                "title": source_section.title,
                "chapter": source_section.chapter,
                "section": source_section.section,
                "page": source_section.page if source_section.page is not None else -1,
                "chunk_index": chunk_index,
                "file_hash": file_hash,
            }
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))
        return chunks

    # A concise alias makes the component convenient in callers and tests.
    chunk = chunk_document

    @staticmethod
    def _same_context(left: ParsedSection, right: ParsedSection) -> bool:
        return (
            left.title,
            left.chapter,
            left.section,
            left.page,
        ) == (
            right.title,
            right.chapter,
            right.section,
            right.page,
        )

    @staticmethod
    def _protected_spans(text: str) -> list[tuple[int, int]]:
        pattern = re.compile(
            r"```[\s\S]*?```"
            r"|~~~[\s\S]*?~~~"
            r"|\$\$[\s\S]*?\$\$"
            r"|(?<!\\)\$(?!\$)(?:\\.|[^$\n\\])+(?<!\\)\$"
        )
        return [(match.start(), match.end()) for match in pattern.finditer(text)]

    @staticmethod
    def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
        return any(start < position < end for start, end in spans)

    def _split_long(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        spans = self._protected_spans(text)
        boundaries = [
            match.end()
            for match in re.finditer(r"(?:\n\s*\n|[。！？!?；;]\s*|[，,、]\s+)", text)
            if not self._inside(match.end(), spans)
        ]
        output: list[str] = []
        start = 0
        length = len(text)
        while start < length:
            target = min(start + self.chunk_size, length)
            extended_for_protected_span = False
            if target == length:
                end = length
            else:
                minimum = start + max(1, int(self.chunk_size * 0.55))
                candidates = [point for point in boundaries if minimum <= point <= target]
                end = candidates[-1] if candidates else target
                if self._inside(end, spans):
                    containing = next((span for span in spans if span[0] < end < span[1]), None)
                    if containing:
                        end = containing[1]
                        extended_for_protected_span = True
            piece = text[start:end].strip()
            if piece:
                output.append(piece)
            if end >= length:
                break
            if extended_for_protected_span:
                # Keeping an atomic formula/fence intact is more important than
                # overlap at that single boundary, especially for over-size blocks.
                start = end
                continue
            next_start = max(start + 1, end - self.overlap)
            containing_start = next(
                (span for span in spans if span[0] < next_start < span[1]),
                None,
            )
            if containing_start:
                next_start = (
                    containing_start[0]
                    if containing_start[0] > start
                    else containing_start[1]
                )
            start = next_start
        return output
