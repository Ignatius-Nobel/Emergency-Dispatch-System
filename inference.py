#!/usr/bin/env python3
"""
Inference Script — Emergency Response Dispatch System (OpenAI Client)

Required Environment Variables:
    API_BASE_URL  - The LLM API endpoint (e.g., https://api.groq.com/openai/v1)
    MODEL_NAME    - The model identifier (e.g., llama-3.1-8b-instant)
    HF_TOKEN      - Hugging Face token (for model access if needed)

Setup:
    1. Set environment variables:
       PowerShell: $env:API_BASE_URL="https://api.groq.com/openai/v1"
                   $env:MODEL_NAME="llama-3.1-8b-instant"
                   $env:HF_TOKEN="your_token_here"
    2. pip install openai
    3. python inference.py
"""

import argparse
import json
import os
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

USE_LOCAL = False
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.dispatch_grid_environment import DispatchGridEnvironment
    from models import DispatchGridAction
    USE_LOCAL = True
    print("✅ LOCAL mode")
except ImportError:
    print("ℹ️  HTTP mode (requires --base-url)")

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
  "ambulance_staging": <"dispatch"|"stage_nearby"|"on_scene_hold">,
  "reasoning": "<brief explanation>"
}

Rules:
- ambulance_units: for medical emergencies, injuries, casualties
- police_units: for crime, violence, armed threats, hostage, shooter
- fire_units: for fire, smoke, explosion, chemical/hazmat, gas leak
- Set to 0 if that unit type is NOT needed
- At least one unit type must be > 0
- priority: 4=critical/life-threatening, 3=serious, 2=moderate, 1=minor
- backup_requested: true only for large-scale multi-casualty or major incidents
- hospital_choice: "nearest" for closest, "regional" for trauma-capable/specialized, "auto" for system default
- coordination_level: "none" for routine, "mutual_aid" for multi-incident support, "mci_protocol" for mass casualty events
- ambulance_staging: "dispatch" send directly, "stage_nearby" hold close, "on_scene_hold" wait at scene
- Scale units to severity: minor=1, moderate=2, serious=3, critical=4-5

Return ONLY the JSON. No markdown, no extra text."""


def call_lllm(client, obs: dict, model: str) -> dict:
    """Call the LLM using OpenAI client and parse response."""
    msg = f"""EMERGENCY CALL
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

Resource Replenishment:
  Ambulances returning in: {obs.get('ambulances_returning_in', 0)} calls
  Police returning in: {obs.get('police_returning_in', 0)} calls
  Fire returning in: {obs.get('fire_returning_in', 0)} calls

Calls remaining: {obs.get('calls_remaining', 0)}

Respond with JSON only."""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            d = json.loads(content)
            # Validate & clamp
            for f in ["ambulance_units", "police_units", "fire_units", "priority_level", "backup_requested"]:
                if f not in d:
                    raise ValueError(f"Missing: {f}")
            d["ambulance_units"] = max(0, min(5, int(d.get("ambulance_units", 0))))
            d["police_units"] = max(0, min(5, int(d.get("police_units", 0))))
            d["fire_units"] = max(0, min(5, int(d.get("fire_units", 0))))
            d["priority_level"] = max(1, min(4, int(d.get("priority_level", 1))))
            d["backup_requested"] = bool(d.get("backup_requested", False))
            # New fields with defaults
            d["hospital_choice"] = d.get("hospital_choice", "auto")
            if d["hospital_choice"] not in ("nearest", "regional", "auto"):
                d["hospital_choice"] = "auto"
            d["coordination_level"] = d.get("coordination_level", "none")
            if d["coordination_level"] not in ("none", "mutual_aid", "mci_protocol"):
                d["coordination_level"] = "none"
            d["ambulance_staging"] = d.get("ambulance_staging", "dispatch")
            if d["ambulance_staging"] not in ("dispatch", "stage_nearby", "on_scene_hold"):
                d["ambulance_staging"] = "dispatch"
            # Guard: at least one unit must be > 0
            if d["ambulance_units"] == 0 and d["police_units"] == 0 and d["fire_units"] == 0:
                d["ambulance_units"] = 1
            return d
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠️  Parse error attempt {attempt+1}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"  ❌ LLM error attempt {attempt+1}: {e}")
            time.sleep(2)

    print("  ⚠️  Using fallback action")
    return {
        "ambulance_units": 1,
        "police_units": 1,
        "fire_units": 0,
        "priority_level": 3,
        "backup_requested": False,
        "reasoning": "fallback",
    }


def run_local(task: str, client, model: str, verbose=True) -> dict:
    """Run inference using local environment import."""
    env = DispatchGridEnvironment(task=task)
    obs = env.reset()
    data, step = [], 0

    if verbose:
        print(f"\n{'='*60}\n  TASK: {task.upper()}\n{'='*60}")

    while not obs.done and step < 10:
        step += 1
        od = obs.model_dump()
        if od.get("call_id") == "DONE":
            break

        if verbose:
            print(f"\n📞 Call {step}/{od['total_calls']}: [{od['call_id']}]")
            print(f"   {od['call_description'][:100]}...")
            print(f"   Severity: {od['severity']} | Type: {od['incident_type']}")
            print(f"   Resources — 🚑{od['available_ambulances']} 🚔{od['available_police']} 🚒{od['available_fire']}")

        a = call_lllm(client, od, model)

        if verbose:
            print(f"   🤖 ambulance={a['ambulance_units']} | police={a['police_units']} | "
                  f"fire={a['fire_units']} | priority={a['priority_level']} | backup={a['backup_requested']}")
            if a.get("reasoning"):
                print(f"   💭 {a['reasoning'][:80]}")

        action = DispatchGridAction(
            ambulance_units=a["ambulance_units"],
            police_units=a["police_units"],
            fire_units=a["fire_units"],
            priority_level=a["priority_level"],
            backup_requested=a["backup_requested"],
            hospital_choice=a.get("hospital_choice", "auto"),
            coordination_level=a.get("coordination_level", "none"),
            ambulance_staging=a.get("ambulance_staging", "dispatch"),
        )
        obs = env.step(action)

        if verbose:
            print(f"   📊 Reward: {obs.last_action_reward:+.3f} | {obs.last_action_feedback[:100]}")
            print(f"   Resources after — 🚑{obs.available_ambulances} 🚔{obs.available_police} 🚒{obs.available_fire}")

        data.append({
            "call_id": od["call_id"],
            "incident_type": od["incident_type"],
            "severity": od["severity"],
            "action": a,
            "reward": obs.last_action_reward,
            "feedback": obs.last_action_feedback,
        })

    final = obs.cumulative_score
    n = len(data)
    # FIX: Episode bonus (+0.20) is added ONCE, not per call
    # Correct max = n * 0.60 + 0.20 (not n * (0.60 + 0.20))
    max_s = n * 0.60 + 0.20
    norm = round(max(0.0, min(1.0, final / max(max_s, 1))), 4)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  ✅ Raw: {final:.4f} | Normalized: {norm:.4f} ({norm*100:.1f}%)")
        print(f"{'─'*60}")

    return {
        "task": task,
        "calls_handled": n,
        "raw_score": round(final, 4),
        "normalized_score": norm,
        "per_call": data,
    }


def run_http(task: str, base_url: str, client, model: str, verbose=True) -> dict:
    """Run inference using HTTP endpoint."""
    import requests

    resp = requests.post(f"{base_url}/reset", json={"task": task}, timeout=10)
    resp.raise_for_status()
    od = resp.json().get("observation", {})

    data, step, done = [], 0, False
    if verbose:
        print(f"\n{'='*60}\n  TASK: {task.upper()} | {base_url}\n{'='*60}")

    while not done and step < 10:
        step += 1
        if od.get("call_id") == "DONE":
            break
        if verbose:
            print(f"\n📞 Call {step}: {od.get('call_id')}")
            print(f"   {str(od.get('call_description',''))[:100]}...")

        a = call_lllm(client, od, model)
        if verbose:
            print(f"   🤖 ambulance={a['ambulance_units']} police={a['police_units']} fire={a['fire_units']}")

        r = requests.post(
            f"{base_url}/step",
            json={
                "ambulance_units": a["ambulance_units"],
                "police_units": a["police_units"],
                "fire_units": a["fire_units"],
                "priority_level": a["priority_level"],
                "backup_requested": a["backup_requested"],
                "hospital_choice": a.get("hospital_choice", "auto"),
                "coordination_level": a.get("coordination_level", "none"),
                "ambulance_staging": a.get("ambulance_staging", "dispatch"),
            },
            timeout=10,
        )
        r.raise_for_status()
        result = r.json()
        od = result.get("observation", {})
        done = result.get("done", False)
        reward = result.get("reward", 0.0)
        if verbose:
            print(f"   📊 Reward: {reward:+.3f}")
        data.append({"call_id": od.get("call_id", ""), "action": a, "reward": reward})

    final = od.get("cumulative_score", 0.0)
    n = len(data)
    # FIX: Correct normalization formula
    max_s = n * 0.60 + 0.20
    norm = round(max(0.0, min(1.0, final / max(max_s, 1))), 4)
    if verbose:
        print(f"\n  ✅ Score: {norm:.4f} ({norm*100:.1f}%)")
    return {
        "task": task,
        "calls_handled": n,
        "raw_score": round(final, 4),
        "normalized_score": norm,
        "per_call": data,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["easy", "medium", "hard", "crisis"],
        default=["easy", "medium", "hard"],
    )
    p.add_argument("--base-url", default=None)
    p.add_argument(
        "--model",
        default=None,
        help="Model name from MODEL_NAME env var",
    )
    p.add_argument("--output", default="inference_results.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # Required environment variables (per hackathon requirements)
    api_base_url = os.environ.get("API_BASE_URL")
    model_name = args.model or os.environ.get("MODEL_NAME")
    hf_token = os.environ.get("HF_TOKEN")

    if not api_base_url:
        print("\nERROR: API_BASE_URL not set.")
        print("  PowerShell: $env:API_BASE_URL=\"https://api.groq.com/openai/v1\"")
        sys.exit(1)

    if not model_name:
        print("\nERROR: MODEL_NAME not set.")
        print("  PowerShell: $env:MODEL_NAME=\"llama-3.1-8b-instant\"")
        sys.exit(1)

    if not hf_token:
        print("\nWARNING: HF_TOKEN not set (may be required for some models)")
        print("  PowerShell: $env:HF_TOKEN=\"your_token_here\"")

    # Initialize OpenAI client
    openai_client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token or "not-needed",
    )
    verbose = not args.quiet

    print(f"\n🚨 Emergency Dispatch Inference | Model: {model_name} | Tasks: {', '.join(args.tasks)}")
    print(f"   API Base: {api_base_url}")

    results = {}
    for task in args.tasks:
        if USE_LOCAL and not args.base_url:
            results[task] = run_local(task, openai_client, model_name, verbose)
        else:
            if not args.base_url:
                print("ERROR: no --base-url and local import failed")
                sys.exit(1)
            results[task] = run_http(task, args.base_url, openai_client, model_name, verbose)

    print("\n" + "="*60)
    print("  📊 FINAL RESULTS")
    print("="*60)
    scores = []
    for task, res in results.items():
        s = res["normalized_score"]
        scores.append(s)
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  {task.upper():8s}: [{bar}] {s:.4f} ({s*100:.1f}%)")
    overall = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  OVERALL AVG : {overall:.4f} ({overall*100:.1f}%)")
    print("="*60)

    out = {
        "agent": f"OpenAI {model_name}",
        "tasks": results,
        "summary": {t: results[t]["normalized_score"] for t in results},
        "overall_avg": round(overall, 4),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()
