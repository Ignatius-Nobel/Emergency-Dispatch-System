#!/usr/bin/env python3
"""
OpenEnv hackathon inference script — Emergency Response Dispatch System.

Runs an LLM agent against the DispatchGridEnvironment, emitting validator-exact
stdout lines. Runs one episode per difficulty (easy, medium, hard).
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from server.dispatch_grid_environment import DispatchGridEnvironment
from models import DispatchGridAction

# ── configuration ────────────────────────────────────────────────────────────

BENCHMARK = os.getenv("BENCHMARK", "dispatch_grid")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.5


def _episode_difficulties() -> list[str]:
    default_order = ("easy", "medium", "hard")
    raw = os.environ.get("OPENENV_DIFFICULTIES", "")
    if not raw.strip():
        return list(default_order)
    want = {x.strip().lower() for x in raw.split(",") if x.strip()}
    picked = [d for d in default_order if d in want]
    return picked if picked else list(default_order)


EPISODE_DIFFICULTIES: list[str] = _episode_difficulties()

# ── stdout helpers ───────────────────────────────────────────────────────────


def _emit_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _emit_step(
    step: int,
    action_str: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    done_s = "true" if done else "false"
    err_s = error[:200] if error is not None else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_s} error={err_s}",
        flush=True,
    )


def _emit_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    success_s = "true" if success else "false"
    rewards_s = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_s} steps={steps} "
        f"score={score:.3f} rewards={rewards_s}",
        flush=True,
    )


# ── prompt construction ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert emergency dispatch operator.
Given an emergency call, decide exactly how many units of each type to send.

Return ONLY a valid JSON object with these exact fields:
{
  "ambulance_units": <0-5>,
  "police_units": <0-5>,
  "fire_units": <0-5>,
  "priority_level": <1-4>,
  "backup_requested": <true|false>,
  "hospital_choice": <"nearest"|"regional"|"auto">,
  "coordination_level": <"none"|"mutual_aid"|"mci_protocol">,
  "ambulance_staging": <"dispatch"|"stage_nearby"|"on_scene_hold">
}

Return ONLY the JSON. No markdown, no extra text."""


def build_user_prompt(obs: dict) -> str:
    return f"""EMERGENCY CALL
==============
ID       : {obs.get('call_id')}
Type     : {obs.get('incident_type')} | Severity: {obs.get('severity')}
Details  : {obs.get('call_description')}
Location : {obs.get('location')}
Caller   : {obs.get('caller_info')}

Resources available:
  Ambulances : {obs.get('available_ambulances', 5)}
  Police     : {obs.get('available_police', 8)}
  Fire       : {obs.get('available_fire', 4)}

Hospital Status:
  {obs.get('hospital_a_name', 'Hospital A')}: ICU={obs.get('hospital_a_icu_beds', 0)}, General={obs.get('hospital_a_general_beds', 0)}, Trauma={obs.get('hospital_a_trauma_capable', False)}, Distance={obs.get('hospital_a_distance_minutes', 0)}min
  {obs.get('hospital_b_name', 'Hospital B')}: ICU={obs.get('hospital_b_icu_beds', 0)}, General={obs.get('hospital_b_general_beds', 0)}, Trauma={obs.get('hospital_b_trauma_capable', False)}, Distance={obs.get('hospital_b_distance_minutes', 0)}min
  Nearest: {obs.get('nearest_hospital', 'N/A')} | Recommended: {obs.get('recommended_hospital', 'N/A')}

Coordination State:
  District Reserve Units: {obs.get('district_reserve_units', 6)}
  MCI Protocol Active: {obs.get('mci_protocol_active', False)}

Calls remaining: {obs.get('calls_remaining', 0)}

Respond with JSON only."""


# ── model interaction ────────────────────────────────────────────────────────


def get_model_action(client: OpenAI, obs_dict: dict) -> dict:
    user_prompt = build_user_prompt(obs_dict)
    fallback = {
        "ambulance_units": 1,
        "police_units": 1,
        "fire_units": 0,
        "priority_level": 3,
        "backup_requested": False,
        "hospital_choice": "auto",
        "coordination_level": "none",
        "ambulance_staging": "dispatch",
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        d = json.loads(content)
        out = {
            "ambulance_units": max(0, min(5, int(d.get("ambulance_units", fallback["ambulance_units"])))),
            "police_units": max(0, min(5, int(d.get("police_units", fallback["police_units"])))),
            "fire_units": max(0, min(5, int(d.get("fire_units", fallback["fire_units"])))),
            "priority_level": max(1, min(4, int(d.get("priority_level", fallback["priority_level"])))),
            "backup_requested": bool(d.get("backup_requested", fallback["backup_requested"])),
            "hospital_choice": d.get("hospital_choice", "auto"),
            "coordination_level": d.get("coordination_level", "none"),
            "ambulance_staging": d.get("ambulance_staging", "dispatch"),
        }

        if out["ambulance_units"] == 0 and out["police_units"] == 0 and out["fire_units"] == 0:
            out["ambulance_units"] = 1

        return out
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr)
        return fallback


# ── main loop ────────────────────────────────────────────────────────────────


def run_one_episode(
    env: DispatchGridEnvironment,
    client: OpenAI,
    difficulty: str,
) -> None:
    """One full episode: [START] ... [STEP]* ... [END] for a single difficulty."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0

    _emit_start(difficulty)

    try:
        obs = env.reset(seed=0, difficulty=difficulty)
        obs_dict = obs.model_dump()

        for step in range(1, MAX_STEPS + 1):
            if obs_dict.get("call_id") == "DONE":
                break

            action_data = get_model_action(client, obs_dict)
            action_str = json.dumps(action_data, separators=(",", ":"))
            action = DispatchGridAction(**action_data)

            obs = env.step(action)
            obs_dict = obs.model_dump()

            reward = obs.last_action_reward or 0.0
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step

            _emit_step(step=step, action_str=action_str, reward=reward, done=done, error=error)

            if done:
                break

        final_raw_score = obs.cumulative_score
        n = len(rewards)
        max_possible_score = (n * 0.60) + 0.20 if n > 0 else 1.0
        score = final_raw_score / max_possible_score
        score = min(max(score, 0.0), 1.0)

    except Exception:
        error_msg = traceback.format_exc().splitlines()[-1][:200]
        if not rewards:
            _emit_step(1, "{}", 0.0, True, error_msg)
            steps_taken = max(steps_taken, 1)

    success = score >= SUCCESS_SCORE_THRESHOLD
    _emit_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "not-needed")
    env = DispatchGridEnvironment()

    for difficulty in EPISODE_DIFFICULTIES:
        run_one_episode(env, client, difficulty)


if __name__ == "__main__":
    main()
