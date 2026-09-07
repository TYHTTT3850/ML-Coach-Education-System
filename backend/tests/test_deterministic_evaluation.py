from app.services.evaluation_service import (
    parse_numeric_answer,
    score_multiple_choice,
    score_numeric,
)


def test_multiple_choice_scoring_is_deterministic() -> None:
    assert score_multiple_choice(" 选项 b ", "B") == (100.0, True)
    assert score_multiple_choice("A", "B") == (0.0, True)
    assert score_multiple_choice("", "B") == (0.0, False)


def test_numeric_scoring_accepts_scientific_notation_and_tolerance() -> None:
    assert parse_numeric_answer(" 1.25e-2 ") == 0.0125
    assert score_numeric("9.999e1", 100.0, 0.02) == (100.0, True)
    assert score_numeric("99.9", 100.0, 0.02) == (0.0, True)


def test_numeric_scoring_rejects_invalid_or_non_finite_input() -> None:
    assert parse_numeric_answer("不是数值") is None
    assert parse_numeric_answer("NaN") is None
    assert parse_numeric_answer("Infinity") is None
    assert score_numeric("--", 1.0, 0.1) == (0.0, False)
