#!/usr/bin/env python3
"""
Inference Script — Emergency Response Dispatch System (OpenAI Client)
"""

import asyncio
import json
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

from client import DispatchGridEnv
from models import DispatchGridAction


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
TASK_NAME = os.getenv("DISPATCH_GRID_TASK", "easy")
BENCHMARK = os.getenv("DISPATCH_GRID_BENCHMARK", "dispatch_grid")

# Each episode has 4 calls. Score is maxed at ((n * 0.60) + 0.20)
MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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
        "ambulance_staging": "dispatch"
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
        # Ensure all required fields exist
        out = {
            "ambulance_units": max(0, min(5, int(d.get("ambulance_units", fallback["ambulance_units"])))),
            "police_units": max(0, min(5, int(d.get("police_units", fallback["police_units"])))),
            "fire_units": max(0, min(5, int(d.get("fire_units", fallback["fire_units"])))),
            "priority_level": max(1, min(4, int(d.get("priority_level", fallback["priority_level"])))),
            "backup_requested": bool(d.get("backup_requested", fallback["backup_requested"])),
            "hospital_choice": d.get("hospital_choice", "auto"),
            "coordination_level": d.get("coordination_level", "none"),
            "ambulance_staging": d.get("ambulance_staging", "dispatch")
        }

        # Guard: at least one unit must be > 0
        if out["ambulance_units"] == 0 and out["police_units"] == 0 and out["fire_units"] == 0:
            out["ambulance_units"] = 1

        return out
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr)
        return fallback


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "not-needed")

    if LOCAL_IMAGE_NAME:
        env = await DispatchGridEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        # Fallback to local running server default standard configuration
        env = DispatchGridEnv(base_url="http://localhost:8000")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
                
            obs_dict = obs.model_dump()
            if obs_dict.get("call_id") == "DONE":
                break

            action_data = get_model_action(client, obs_dict)
            action_str = json.dumps(action_data).replace(" ", "")
            action = DispatchGridAction(**action_data)

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        final_raw_score = obs.cumulative_score
        n = len(rewards)
        # Max score is 0.60 per call + 0.20 completion bonus
        max_possible_score = (n * 0.60) + 0.20 if n > 0 else 1.0
        
        score = final_raw_score / max_possible_score
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", file=sys.stderr)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
