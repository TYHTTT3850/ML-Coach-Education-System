from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.curriculum import router as curriculum_router
from app.api.students import router as students_router
from app.curriculum.seed import seed_curriculum
from app.services.learning_repository import LearningRepository, StudentNotFoundError


def _client(tmp_path) -> tuple[TestClient, LearningRepository]:
    repository = LearningRepository(tmp_path / "api.db")
    repository.initialize()
    seed_curriculum(repository)
    app = FastAPI()
    app.state.learning_repository = repository
    app.include_router(students_router)
    app.include_router(curriculum_router)
    return TestClient(app), repository


def test_anonymous_student_is_created_by_backend(tmp_path) -> None:
    client, repository = _client(tmp_path)
    response = client.post("/api/v1/students/anonymous")

    assert response.status_code == 201
    payload = response.json()
    assert payload["student_id"]
    assert repository.get_student(payload["student_id"]) is not None


def test_missing_student_is_rejected(tmp_path) -> None:
    _, repository = _client(tmp_path)

    try:
        repository.require_student("6ab2192e-e096-4018-bc08-f452230c5ca7")
    except StudentNotFoundError as exc:
        assert exc.code == "STUDENT_NOT_FOUND"
    else:
        raise AssertionError("missing student should be rejected")


def test_learning_request_touches_last_active_at(tmp_path, monkeypatch) -> None:
    client, repository = _client(tmp_path)
    monkeypatch.setattr(
        LearningRepository,
        "_now",
        staticmethod(lambda: "2026-09-01T10:00:00+00:00"),
    )
    student = repository.create_anonymous_student()
    monkeypatch.setattr(
        LearningRepository,
        "_now",
        staticmethod(lambda: "2026-09-01T10:05:00+00:00"),
    )

    response = client.get(
        "/api/v1/curriculum/topics",
        params={"course": "machine_learning"},
        headers={"X-Student-ID": student["id"]},
    )

    assert response.status_code == 200
    assert repository.get_student(student["id"])["last_active_at"] == (
        "2026-09-01T10:05:00+00:00"
    )


def test_curriculum_response_hides_unstarted_beta_prior(tmp_path) -> None:
    client, repository = _client(tmp_path)
    student = repository.create_anonymous_student()

    response = client.get(
        "/api/v1/curriculum/topics",
        params={"course": "machine_learning"},
        headers={"X-Student-ID": student["id"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["course"] == "machine_learning"
    first = payload["stages"][0]["topics"][0]
    second = payload["stages"][0]["topics"][1]
    assert first["status"] == "not_started"
    assert first["mastery"] is None
    assert first["mastery_display"] is None
    assert second["status"] == "not_started"
    assert second["locked_reason"] is None


def test_topic_detail_exposes_objectives_and_empty_dimension_state(tmp_path) -> None:
    client, repository = _client(tmp_path)
    student = repository.create_anonymous_student()

    response = client.get(
        "/api/v1/curriculum/topics/ml.logistic_regression",
        headers={"X-Student-ID": student["id"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "ml.logistic_regression"
    assert payload["learning_objectives"]
    assert payload["required_dimensions"] == [
        "concept",
        "mathematics",
        "mechanism",
        "engineering",
        "application",
    ]
    assert all(item["mastery"] is None for item in payload["dimensions"])
