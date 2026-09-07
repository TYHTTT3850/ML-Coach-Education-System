import io
import logging

from app.core.logging import RedactingFormatter


def test_redacting_formatter_removes_secret_from_message_and_traceback() -> None:
    secret = "sk-complete-secret-value"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(levelname)s %(message)s", (secret,)))
    logger = logging.getLogger("coursecoach.test.redaction")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.ERROR)
    try:
        raise RuntimeError(f"SDK failed with key {secret}")
    except RuntimeError:
        logger.exception("request used %s and Bearer visible-token", secret)
    output = stream.getvalue()
    assert secret not in output
    assert "visible-token" not in output
    assert "***" in output
