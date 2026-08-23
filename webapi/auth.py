"""API token for the web control endpoints (M6).

The simulation control endpoints can stop/reset the paper account and
rewrite its fee configuration, so mutating requests must be authorized
whenever the API is reachable from a non-loopback address.  The shared
token lives in ``config/.webapi_token`` (gitignored): when the file
exists every mutating request must present its content in the
``X-API-Token`` header; when it does not exist, only loopback clients
may mutate (a convenience for single-user local deployments) and
non-loopback mutating requests are rejected with a hint to create the
token file.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import HTTPException, Request

TOKEN_FILENAME = ".webapi_token"
_TOKEN_DIR = "config"
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def token_path() -> Path:
    """Token file path; tests override ``ASHARE_WEBAPI_ROOT``."""
    root = Path(
        os.environ.get(
            "ASHARE_WEBAPI_ROOT",
            Path(__file__).resolve().parents[1],
        )
    )
    return root / _TOKEN_DIR / TOKEN_FILENAME


async def require_mutation_token(request: Request) -> None:
    """FastAPI dependency guarding every mutating endpoint."""

    path = token_path()
    if path.exists():
        expected = path.read_text(encoding="utf-8").strip()
        given = request.headers.get("X-API-Token", "")
        if expected and hmac.compare_digest(given, expected):
            return
        raise HTTPException(
            status_code=401, detail="invalid or missing X-API-Token header"
        )
    client_host = request.client.host if request.client is not None else "unknown"
    if client_host in _LOOPBACK_HOSTS:
        return
    raise HTTPException(
        status_code=403,
        detail=(
            "control endpoints are disabled for non-loopback clients; "
            f"create {_TOKEN_DIR}/{TOKEN_FILENAME} to enable them"
        ),
    )
