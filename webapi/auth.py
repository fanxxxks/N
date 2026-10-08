"""API token for the web control endpoints (M6).

The simulation control endpoints can stop/reset the paper account and
rewrite its fee configuration, so mutating requests must be authorized
whenever the API is reachable from a non-loopback address.  The shared
token lives in ``config/.webapi_token`` (gitignored): when the file
exists every mutating request must present its content in the
``X-API-Token`` header; when it does not exist, only requests that are
loopback on BOTH ends of the TCP chain may mutate (a convenience for
single-user local deployments): the client address seen by the server
must be loopback AND the ``Host`` header must address a loopback
name/address.  The Host check closes the "remote -> local reverse proxy
/ ssh -L / VS Code port-forward -> 127.0.0.1:8000" topology, where every
forwarded request arrives with a loopback client address but carries the
origin's external Host header.  With a token file present the Host check
does not apply — a correctly tokened request is the documented way to
expose the API beyond loopback.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import HTTPException, Request

TOKEN_FILENAME = ".webapi_token"
_TOKEN_DIR = "config"
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


def token_path() -> Path:
    """Token file path; tests override ``ASHARE_WEBAPI_ROOT``."""
    root = Path(
        os.environ.get(
            "ASHARE_WEBAPI_ROOT",
            Path(__file__).resolve().parents[1],
        )
    )
    return root / _TOKEN_DIR / TOKEN_FILENAME


def _host_header_name(host_header: str) -> str:
    """The host part of a ``Host`` header, lowercased, port stripped.

    Handles the bracketed IPv6 form (``[::1]:8000`` -> ``::1``) and the
    usual ``name:port`` form (``127.0.0.1:8000`` -> ``127.0.0.1``); a bare
    ``::1`` or ``::ffff:127.0.0.1`` (multiple colons, no port) is kept
    as-is."""

    host = (host_header or "").strip()
    if host.startswith("["):
        return host.split("]", 1)[0].lstrip("[").lower()
    if host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    return host.lower()


def _is_loopback_host_header(host_header: str) -> bool:
    """True when the Host header addresses this machine's loopback.

    An empty Host (HTTP/1.0-style clients, minimal test drivers) counts
    as loopback: the caller has already established that the TCP peer is
    loopback, and a hostile proxy that omits Host could just as well set
    it to ``127.0.0.1`` — this check exists to catch proxies that
    faithfully FORWARD the origin's Host, not to outsmart one that lies.
    """

    host = _host_header_name(host_header)
    return not host or host in _LOOPBACK_HOSTS


async def require_mutation_token(request: Request) -> None:
    """FastAPI dependency guarding every mutating endpoint."""

    path = token_path()
    if path.exists():
        try:
            expected = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            # An unreadable token file is an invalid token file: fall
            # through to the 401 below instead of crashing with a 500.
            expected = ""
        given = request.headers.get("X-API-Token", "")
        # hmac.compare_digest rejects non-ASCII str inputs with a
        # TypeError (-> 500); treat any non-ASCII side as a plain
        # mismatch instead (P1-2).
        if (
            expected
            and expected.isascii()
            and given.isascii()
            and hmac.compare_digest(given, expected)
        ):
            return
        raise HTTPException(
            status_code=401, detail="invalid or missing X-API-Token header"
        )
    client_host = request.client.host if request.client is not None else "unknown"
    host_header = request.headers.get("host", "")
    if client_host in _LOOPBACK_HOSTS and _is_loopback_host_header(host_header):
        return
    raise HTTPException(
        status_code=403,
        detail=(
            "control endpoints are disabled for non-loopback or forwarded "
            f"clients; create {_TOKEN_DIR}/{TOKEN_FILENAME} to enable them"
        ),
    )
