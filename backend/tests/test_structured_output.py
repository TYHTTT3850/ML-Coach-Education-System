from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from app.core.exceptions import StructuredOutputInvalidError
from app.services.structured_output import extract_json, generate_validated


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int


class SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, messages, temperature=None):
        self.last_messages = messages
        self.calls += 1
        return self.responses.pop(0)


def test_extract_json_accepts_fence_but_not_trailing_prose() -> None:
    assert extract_json('```json\n{"score": 80}\n```') == {"score": 80}
    with pytest.raises(ValueError):
        extract_json('{"score": 80} explanation')


@pytest.mark.asyncio
async def test_invalid_json_is_repaired_at_most_once() -> None:
    llm = SequenceLLM(["not json", '{"score": 72}'])
    result = await generate_validated(
        llm_client=llm,  # type: ignore[arg-type]
        messages=[{"role": "user", "content": "grade"}],
        model=Payload,
        temperature=0,
        repair_attempts=1,
    )
    assert result.score == 72
    assert llm.calls == 2
    repair_prompt = llm.last_messages[1]["content"]
    assert "目标 JSON Schema" in repair_prompt
    assert '"score"' in repair_prompt
    assert "json_invalid" in repair_prompt
    assert "UNTRUSTED_OUTPUT" in repair_prompt


@pytest.mark.asyncio
async def test_failed_repair_stops_without_unbounded_calls() -> None:
    llm = SequenceLLM(["bad", "still bad"])
    with pytest.raises(StructuredOutputInvalidError):
        await generate_validated(
            llm_client=llm,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": "grade"}],
            model=Payload,
            temperature=0,
            repair_attempts=1,
        )
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_schema_repair_receives_locations_without_original_input() -> None:
    llm = SequenceLLM(['{"points": 72}', '{"score": 72}'])
    result = await generate_validated(
        llm_client=llm,  # type: ignore[arg-type]
        messages=[{"role": "user", "content": "sensitive grading facts"}],
        model=Payload,
        temperature=0,
        repair_attempts=1,
    )

    assert result.score == 72
    repair_prompt = llm.last_messages[1]["content"]
    assert "missing" in repair_prompt
    assert '"loc":["score"]' in repair_prompt
    assert "sensitive grading facts" not in repair_prompt
