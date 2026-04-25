# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
"""Stateless HTTP helpers for OpenEnv: persist episode via X-Session-Id (see server/app.py)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import requests

SESSION_HEADER = "X-Session-Id"


def post_reset(base_url: str, json_body: Dict[str, Any], timeout: float = 10) -> Tuple[str, Dict[str, Any]]:
    """
    POST /reset. Returns (session_id, full_json_response).
    The session id must be sent on all subsequent /step and /state calls.
    """
    r = requests.post(
        f"{base_url.rstrip('/')}/reset",
        json=json_body,
        timeout=timeout,
    )
    r.raise_for_status()
    sid = r.headers.get(SESSION_HEADER) or r.headers.get(SESSION_HEADER.lower())
    if not sid:
        raise RuntimeError(
            "Response missing X-Session-Id header. "
            "Use a server build with persistent HTTP sessions (dispatch_grid app)."
        )
    return sid, r.json()


def post_step(
    base_url: str,
    session_id: str,
    action: Dict[str, Any],
    timeout: float = 10,
) -> Dict[str, Any]:
    """POST /step with wrapped action and session header."""
    r = requests.post(
        f"{base_url.rstrip('/')}/step",
        json={"action": action},
        headers={SESSION_HEADER: session_id},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def get_state(base_url: str, session_id: str, timeout: float = 10) -> Dict[str, Any]:
    r = requests.get(
        f"{base_url.rstrip('/')}/state",
        headers={SESSION_HEADER: session_id},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()
