#!/usr/bin/env python3
"""
Baseline Agent — Emergency Response Dispatch System (Groq)

Setup:
    1. Sign up FREE at https://console.groq.com → get API key
    2. PowerShell: $env:GROQ_API_KEY="gsk_your_key_here"
    3. pip install groq
    4. python baseline_agent.py
"""

import argparse, json, os, sys, time

try:
    from groq import Groq
except ImportError:
    print("ERROR: pip install groq")
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
- Scale units to severity: minor=1, moderate=2, serious=3, critical=4-5

Return ONLY the JSON. No markdown, no extra text."""


def call_groq(client, obs: dict, model: str) -> dict:
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
                temperature=0.1, max_tokens=300,
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
            d["ambulance_units"] = max(0, min(5, int(d["ambulance_units"])))
            d["police_units"]    = max(0, min(5, int(d["police_units"])))
            d["fire_units"]      = max(0, min(5, int(d["fire_units"])))
            d["priority_level"]  = max(1, min(4, int(d["priority_level"])))
            d["backup_requested"] = bool(d["backup_requested"])
            # Guard: at least one unit must be > 0
            if d["ambulance_units"] == 0 and d["police_units"] == 0 and d["fire_units"] == 0:
                d["ambulance_units"] = 1
            return d
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠️  Parse error attempt {attempt+1}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"  ❌ Groq error attempt {attempt+1}: {e}")
            time.sleep(2)

    print("  ⚠️  Using fallback action")
    return {"ambulance_units": 1, "police_units": 1, "fire_units": 0,
            "priority_level": 3, "backup_requested": False, "reasoning": "fallback"}


def run_local(task: str, client, model: str, verbose=True) -> dict:
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

        a = call_groq(client, od, model)

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
    max_s = n * (0.60 + 0.20)
    norm = round(max(0.0, min(1.0, final / max(max_s, 1))), 4)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  ✅ Raw: {final:.4f} | Normalized: {norm:.4f} ({norm*100:.1f}%)")
        print(f"{'─'*60}")

    return {"task": task, "calls_handled": n, "raw_score": round(final, 4),
            "normalized_score": norm, "per_call": data}


def run_http(task: str, base_url: str, client, model: str, verbose=True) -> dict:
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

        a = call_groq(client, od, model)
        if verbose:
            print(f"   🤖 ambulance={a['ambulance_units']} police={a['police_units']} fire={a['fire_units']}")

        r = requests.post(f"{base_url}/step", json={
            "ambulance_units": a["ambulance_units"],
            "police_units": a["police_units"],
            "fire_units": a["fire_units"],
            "priority_level": a["priority_level"],
            "backup_requested": a["backup_requested"],
        }, timeout=10)
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
    norm = round(max(0.0, min(1.0, final / max(n, 1))), 4)
    if verbose:
        print(f"\n  ✅ Score: {norm:.4f} ({norm*100:.1f}%)")
    return {"task": task, "calls_handled": n, "raw_score": round(final, 4),
            "normalized_score": norm, "per_call": data}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", choices=["easy","medium","hard"],
                   default=["easy","medium","hard"])
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default="moonshotai/kimi-k2-instruct",##"llama-3.1-8b-instant",
                   help="Groq model. Options: llama-3.1-8b-instant, llama-3.3-70b-versatile")
    p.add_argument("--output", default="baseline_results.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\nERROR: GROQ_API_KEY not set.")
        print("  PowerShell: $env:GROQ_API_KEY=\"gsk_your_key_here\"")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)
    verbose = not args.quiet

    print(f"\n🚨 Emergency Dispatch Baseline | Model: {args.model} | Tasks: {', '.join(args.tasks)}")

    results = {}
    for task in args.tasks:
        if USE_LOCAL and not args.base_url:
            results[task] = run_local(task, groq_client, args.model, verbose)
        else:
            if not args.base_url:
                print("ERROR: no --base-url and local import failed")
                sys.exit(1)
            results[task] = run_http(task, args.base_url, groq_client, args.model, verbose)

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

    out = {"agent": f"Groq {args.model}", "tasks": results,
           "summary": {t: results[t]["normalized_score"] for t in results},
           "overall_avg": round(overall, 4)}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()