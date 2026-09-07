from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request, status


def require_admin_token(
    request: Request,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    expected = request.app.state.settings.admin_token.get_secret_value()
    if not expected or not x_admin_token or not secrets.compare_digest(expected, x_admin_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="管理员令牌无效",
        )
