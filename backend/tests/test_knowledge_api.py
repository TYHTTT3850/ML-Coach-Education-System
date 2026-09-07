from app.core.types import Course


def upload_markdown(client, token: str = "test-admin-token"):
    return client.post(
        "/api/v1/knowledge/upload",
        headers={"X-Admin-Token": token},
        data={"course": "machine_learning"},
        files={"file": ("lesson.md", "# 逻辑回归\n\n逻辑回归使用交叉熵。", "text/markdown")},
    )


def test_admin_token_is_required(client) -> None:
    response = client.post(
        "/api/v1/knowledge/upload",
        data={"course": "machine_learning"},
        files={"file": ("lesson.md", "# 内容", "text/markdown")},
    )
    assert response.status_code == 401
    assert upload_markdown(client, "wrong-token").status_code == 401


def test_same_file_is_deduplicated_in_same_course(client) -> None:
    first = upload_markdown(client)
    second = upload_markdown(client)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["duplicate"] is False
    assert second.json()["duplicate"] is True
    assert first.json()["document"]["id"] == second.json()["document"]["id"]
    documents = client.get(
        "/api/v1/knowledge/documents", params={"course": "machine_learning"}
    ).json()["documents"]
    assert len(documents) == 1


def test_failed_document_is_not_reported_as_successful_duplicate(client) -> None:
    request_kwargs = {
        "headers": {"X-Admin-Token": "test-admin-token"},
        "data": {"course": "machine_learning"},
        "files": {"file": ("blank.md", " \n\t", "text/markdown")},
    }
    first = client.post("/api/v1/knowledge/upload", **request_kwargs)
    assert first.status_code == 422
    documents = client.get(
        "/api/v1/knowledge/documents", params={"course": "machine_learning"}
    ).json()["documents"]
    assert len(documents) == 1
    assert documents[0]["status"] == "failed"

    second = client.post("/api/v1/knowledge/upload", **request_kwargs)
    assert second.status_code == 409
    assert "上次导入失败" in second.json()["error"]["message"]


def test_delete_removes_registry_vectors_and_file(client) -> None:
    uploaded = upload_markdown(client)
    assert uploaded.status_code == 200, uploaded.text
    document = uploaded.json()["document"]
    document_id = document["id"]
    stored_path = client.app.state.settings.upload_path / document["stored_filename"]
    assert stored_path.exists()
    response = client.delete(
        f"/api/v1/knowledge/documents/{document_id}",
        headers={"X-Admin-Token": "test-admin-token"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"deleted": True, "document_id": document_id}
    assert client.app.state.registry.get_document(document_id) is None
    assert client.app.state.vector_store.count(Course.MACHINE_LEARNING) == 0
    assert not stored_path.exists()
    missing = client.delete(
        f"/api/v1/knowledge/documents/{document_id}",
        headers={"X-Admin-Token": "test-admin-token"},
    )
    assert missing.status_code == 404
