# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Emergency Response Dispatch System.

Standard OpenEnv endpoints:
    POST /reset   → Reset environment
    POST /step    → Agent takes action
    GET  /state   → Current state
    GET  /schema  → Action/observation schemas
    WS   /ws      → WebSocket sessions

Hackathon-required endpoints:
    GET  /tasks    → List all 3 tasks
    POST /grader   → Score a single action
    POST /baseline → Run baseline agent on all tasks
"""

import asyncio
import json
import os
import threading
import time
from typing import Any, Dict, Tuple, cast
from uuid import uuid4

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.serialization import deserialize_action, serialize_observation
    from openenv.core.env_server.types import (
        ResetRequest,
        ResetResponse,
        State,
        StepRequest,
        StepResponse,
    )
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from models import DispatchGridAction, DispatchGridObservation
    from server.dispatch_grid_environment import (
        DispatchGridEnvironment, EASY_CALLS, MEDIUM_CALLS, HARD_CALLS, compute_reward,
    )
except ImportError:
    try:
        from dispatch_grid.models import DispatchGridAction, DispatchGridObservation
        from dispatch_grid.server.dispatch_grid_environment import (
            DispatchGridEnvironment, EASY_CALLS, MEDIUM_CALLS, HARD_CALLS, compute_reward,
        )
    except ImportError:
        from ..models import DispatchGridAction, DispatchGridObservation
        from .dispatch_grid_environment import (
            DispatchGridEnvironment, EASY_CALLS, MEDIUM_CALLS, HARD_CALLS, compute_reward,
        )

from fastapi import Body, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Base app
# ---------------------------------------------------------------------------

app = create_app(
    DispatchGridEnvironment,
    DispatchGridAction,
    DispatchGridObservation,
    env_name="dispatch_grid",
    max_concurrent_envs=10,
)

# ---------------------------------------------------------------------------
# Fix: stateful HTTP /reset, /step, /state (OpenEnv default closes env each call)
# Clients must pass header X-Session-Id from the first /reset on every /step and /state.
# ---------------------------------------------------------------------------

SESSION_ID_HEADER: str = "X-Session-Id"

_MAX_HTTP_ENV_SESSIONS: int = 10
_http_env_lock: threading.Lock = threading.Lock()
_http_env_sessions: Dict[str, DispatchGridEnvironment] = {}
_http_env_session_order: list[str] = []


def _http_strip_stateless_openenv_routes() -> None:
    """Remove default create_app handlers that create+close a new env on every request."""
    keep: list = []
    remove_pairs = {("/reset", "POST"), ("/step", "POST"), ("/state", "GET")}
    for route in list(app.router.routes):
        if isinstance(route, APIRoute) and len(route.methods) == 1:
            meth = next(iter(route.methods))
            if (route.path, cast(str, meth)) in remove_pairs:
                continue
        keep.append(route)
    app.router.routes = keep  # type: ignore[assignment]


def _http_evict_oldest_session_if_full() -> None:
    while len(_http_env_sessions) >= _MAX_HTTP_ENV_SESSIONS and _http_env_session_order:
        old_id = _http_env_session_order.pop(0)
        old_env = _http_env_sessions.pop(old_id, None)
        if old_env is not None and hasattr(old_env, "close"):
            try:
                old_env.close()  # type: ignore[no-untyped-call]
            except Exception:  # pragma: no cover
                pass


def _http_get_or_create_session(
    request: Request, create_new_if_no_header: bool
) -> Tuple[str, DispatchGridEnvironment, bool]:
    """
    Return (session_id, env, created_new).
    If X-Session-Id is missing and create_new_if_no_header, allocate a new session.
    If X-Session-Id is present and unknown, start a new env under that id (client reconnect).
    """
    header_sid = request.headers.get("x-session-id")
    with _http_env_lock:
        if header_sid and header_sid in _http_env_sessions:
            return header_sid, _http_env_sessions[header_sid], False
        if not header_sid and not create_new_if_no_header:
            raise KeyError("missing session")
        _http_evict_oldest_session_if_full()
        new_id = header_sid or str(uuid4())
        env = DispatchGridEnvironment(task="easy")
        _http_env_sessions[new_id] = env
        _http_env_session_order.append(new_id)
        return new_id, env, True


async def _http_persist_reset(
    request: Request, response: Response, body: ResetRequest
) -> ResetResponse:
    sid, env, _is_new = _http_get_or_create_session(request, create_new_if_no_header=True)
    kwargs: Dict[str, Any] = body.model_dump(exclude_unset=True)
    observation = await asyncio.to_thread(env.reset, **kwargs)
    payload = serialize_observation(observation)
    out = ResetResponse(**payload)
    response.headers[SESSION_ID_HEADER] = sid
    return out


async def _http_persist_step(request: Request, body: StepRequest) -> StepResponse:
    with _http_env_lock:
        header_sid = request.headers.get("x-session-id")
        if not header_sid or header_sid not in _http_env_sessions:
            raise HTTPException(
                status_code=400,
                detail="Missing or unknown X-Session-Id. Call POST /reset first and reuse the returned session id.",
            )
        env = _http_env_sessions[header_sid]
    try:
        action = deserialize_action(body.action, DispatchGridAction)
    except Exception as exc:  # pydantic validation
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    observation = await asyncio.to_thread(env.step, action)
    return StepResponse(**serialize_observation(observation))


def _http_persist_get_state(request: Request) -> State:
    with _http_env_lock:
        header_sid = request.headers.get("x-session-id")
        if not header_sid or header_sid not in _http_env_sessions:
            raise HTTPException(
                status_code=400,
                detail="Missing or unknown X-Session-Id. Call POST /reset first and reuse the returned session id.",
            )
        env = _http_env_sessions[header_sid]
    st = env.state
    if isinstance(st, State):
        return st
    return State.model_validate(st)  # type: ignore[call-arg, arg-type, misc]


_http_strip_stateless_openenv_routes()

@app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
async def _dispatch_http_reset_persist(
    request: Request, response: Response, body: ResetRequest = Body(default_factory=ResetRequest)
) -> ResetResponse:
    return await _http_persist_reset(request, response, body=body)


@app.post("/step", response_model=StepResponse, tags=["Environment Control"])
async def _dispatch_http_step_persist(request: Request, body: StepRequest) -> StepResponse:
    return await _http_persist_step(request, body=body)


@app.get("/state", response_model=State, tags=["State Management"])
def _dispatch_http_state_persist(request: Request) -> State:
    return _http_persist_get_state(request)

# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------

TASK_DEFINITIONS = [
    {
        "task_id": "easy",
        "name": "Basic Emergency Dispatch",
        "difficulty": "Easy",
        "description": (
            "Single-type emergencies with clear descriptions. "
            "The agent dispatches only one unit type (ambulance, police, or fire)."
        ),
        "num_calls_per_episode": 4,
        "example_call": "Elderly woman collapsed at home, unconscious.",
        "example_correct_action": {
            "ambulance_units": 1, "police_units": 0, "fire_units": 0,
            "priority_level": 3, "backup_requested": False,
            "hospital_choice": "nearest", "coordination_level": "none", "ambulance_staging": "dispatch",
        },
    },
    {
        "task_id": "medium",
        "name": "Ambiguous & Multi-Type Dispatch",
        "difficulty": "Medium",
        "description": (
            "Ambiguous calls requiring multiple unit types simultaneously. "
            "Resource constraints and incomplete caller info add complexity."
        ),
        "num_calls_per_episode": 4,
        "example_call": "Car crash on highway — driver trapped, small engine fire visible.",
        "example_correct_action": {
            "ambulance_units": 1, "police_units": 1, "fire_units": 1,
            "priority_level": 4, "backup_requested": False,
            "hospital_choice": "nearest", "coordination_level": "none", "ambulance_staging": "dispatch",
        },
    },
    {
        "task_id": "hard",
        "name": "Multi-Hazard Cascading Emergencies",
        "difficulty": "Hard",
        "description": (
            "Complex multi-hazard incidents: explosions, active shooters, train derailments. "
            "Require large multi-agency response and backup."
        ),
        "num_calls_per_episode": 4,
        "example_call": "Gas explosion at apartment — casualties, fire spreading, residents trapped.",
        "example_correct_action": {
            "ambulance_units": 3, "police_units": 2, "fire_units": 3,
            "priority_level": 4, "backup_requested": True,
            "hospital_choice": "regional", "coordination_level": "mci_protocol", "ambulance_staging": "stage_nearby",
        },
    },
]


@app.get("/tasks")
async def list_tasks():
    return JSONResponse(content={
        "tasks": TASK_DEFINITIONS,
        "total_tasks": len(TASK_DEFINITIONS),
        "environment": "Emergency Response Dispatch System",
        "action_format": {
            "ambulance_units": "int 0-5",
            "police_units": "int 0-5",
            "fire_units": "int 0-5",
            "priority_level": "int 1-4",
            "backup_requested": "bool",
            "hospital_choice": "str: 'nearest' | 'regional' | 'auto'",
            "coordination_level": "str: 'none' | 'mutual_aid' | 'mci_protocol'",
            "ambulance_staging": "str: 'dispatch' | 'stage_nearby' | 'on_scene_hold'",
        },
    })


# ---------------------------------------------------------------------------
# /demo/compare — MVE outcome metrics (nearest vs capacity-aware)
# ---------------------------------------------------------------------------


@app.get("/demo/compare")
async def demo_compare(seed: int = 0, task: str = "hard"):
    """Return JSON metrics for side-by-side policies (outcome reward mode)."""
    import os
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    valid_tasks = {"easy", "medium", "hard", "crisis"}
    if task not in valid_tasks:
        return JSONResponse(
            status_code=400,
            content={"error": f"unknown task {task!r}", "valid_tasks": sorted(valid_tasks)},
        )
    try:
        from demo.run_demo import run_comparison_metrics
    except ImportError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    payload = run_comparison_metrics(seed=seed, task=task)
    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# /grader
# ---------------------------------------------------------------------------

ALL_CALLS = {c.call_id: c for c in EASY_CALLS + MEDIUM_CALLS + HARD_CALLS}


class GraderRequest(BaseModel):
    call_id: str
    ambulance_units: int = 0
    police_units: int = 0
    fire_units: int = 0
    priority_level: int
    backup_requested: bool = False
    hospital_choice: str = "auto"
    coordination_level: str = "none"
    ambulance_staging: str = "dispatch"


@app.post("/grader")
async def grade_action(request: GraderRequest):
    """Score a single dispatch action against ground truth. Returns 0.0–1.0."""
    call = ALL_CALLS.get(request.call_id)
    if call is None:
        return JSONResponse(status_code=404, content={
            "error": f"Call ID '{request.call_id}' not found.",
            "valid_call_ids": list(ALL_CALLS.keys()),
        })

    action = DispatchGridAction(
        ambulance_units=request.ambulance_units,
        police_units=request.police_units,
        fire_units=request.fire_units,
        priority_level=request.priority_level,
        backup_requested=request.backup_requested,
        hospital_choice=request.hospital_choice,
        coordination_level=request.coordination_level,
        ambulance_staging=request.ambulance_staging,
    )

    raw_reward, feedback = compute_reward(action, call)

    # Normalize: max possible = 0.10*3 + 0.20 + 0.10 = 0.60
    # Min possible ≈ -0.10*3 - 0.10 - 0.05 = -0.45
    max_r, min_r = 0.60, -0.45
    normalized = round((raw_reward - min_r) / (max_r - min_r), 4)
    normalized = max(0.0, min(1.0, normalized))

    return JSONResponse(content={
        "call_id": request.call_id,
        "incident_type": call.incident_type,
        "score": normalized,
        "raw_reward": round(raw_reward, 4),
        "feedback": feedback,
        "ground_truth": {
            "correct_dispatch": call.correct_dispatch,
            "correct_priority": call.correct_priority,
            "needs_backup": call.needs_backup,
        },
        "your_action": {
            "ambulance_units": request.ambulance_units,
            "police_units": request.police_units,
            "fire_units": request.fire_units,
            "priority_level": request.priority_level,
            "backup_requested": request.backup_requested,
        },
    })


# ---------------------------------------------------------------------------
# /baseline  — rule-based agent
# ---------------------------------------------------------------------------

def rule_based_agent(observation: dict) -> dict:
    """Keyword-based rule agent for baseline comparison."""
    desc = (
        observation.get("call_description", "") + " " +
        observation.get("incident_type", "")
    ).lower()
    severity = observation.get("severity", "moderate")

    severity_priority = {"critical": 4, "severe": 3, "moderate": 2, "minor": 1}
    priority = severity_priority.get(severity, 2)

    medical_kw = ["medical", "ambulance", "collapsed", "chest pain", "unconscious",
                  "breathing", "choking", "injured", "casualty", "diabetic", "bleeding", "hurt"]
    fire_kw = ["fire", "smoke", "burning", "explosion", "chemical", "hazmat", "gas", "flame"]
    police_kw = ["crime", "robbery", "burglary", "armed", "weapon", "assault",
                 "domestic", "shooter", "hostage", "theft", "fight", "stabbing"]

    needs_ambulance = any(k in desc for k in medical_kw)
    needs_fire = any(k in desc for k in fire_kw)
    needs_police = any(k in desc for k in police_kw)

    # Defaults if nothing matched
    if not needs_ambulance and not needs_fire and not needs_police:
        needs_ambulance = True

    ambulance_units = (1 if priority <= 2 else 2) if needs_ambulance else 0
    police_units = (2 if priority <= 2 else 3) if needs_police else 0
    fire_units = (1 if priority <= 2 else 2) if needs_fire else 0

    total_types = sum([needs_ambulance, needs_fire, needs_police])
    backup = priority == 4 and total_types >= 2

    return {
        "ambulance_units": ambulance_units,
        "police_units": police_units,
        "fire_units": fire_units,
        "priority_level": priority,
        "backup_requested": backup,
    }


@app.post("/baseline")
async def run_baseline():
    """Run rule-based baseline agent on all 3 tasks. Returns per-call scores and summary."""
    results = {}

    for task_name in ["easy", "medium", "hard"]:
        env = DispatchGridEnvironment(task=task_name)
        obs = env.reset()
        call_details = []
        rewards = []

        step = 0
        while not obs.done and step < 10:
            step += 1
            obs_dict = obs.model_dump()
            if obs_dict.get("call_id") == "DONE":
                break

            agent_action = rule_based_agent(obs_dict)
            action = DispatchGridAction(**agent_action)
            obs = env.step(action)

            call_details.append({
                "call_id": obs_dict.get("call_id", ""),
                "description": obs_dict.get("call_description", "")[:80] + "...",
                "agent_action": agent_action,
                "reward": obs.last_action_reward,
                "feedback": obs.last_action_feedback,
            })
            rewards.append(obs.last_action_reward)

        final_score = obs.cumulative_score
        n = len(call_details)
        max_score = n * (0.60 + 0.20)
        normalized = round(max(0.0, min(1.0, final_score / max(max_score, 1))), 4)

        results[task_name] = {
            "episode_score": round(final_score, 4),
            "normalized_score_0_to_1": normalized,
            "calls_handled": n,
            "per_call_rewards": rewards,
            "call_details": call_details,
        }

    return JSONResponse(content={
        "agent": "Rule-Based Keyword Baseline",
        "tasks": results,
        "summary": {
            "easy_score": results["easy"]["normalized_score_0_to_1"],
            "medium_score": results["medium"]["normalized_score_0_to_1"],
            "hard_score": results["hard"]["normalized_score_0_to_1"],
            "overall_avg": round(
                (results["easy"]["normalized_score_0_to_1"] +
                 results["medium"]["normalized_score_0_to_1"] +
                 results["hard"]["normalized_score_0_to_1"]) / 3, 4
            ),
        },
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()