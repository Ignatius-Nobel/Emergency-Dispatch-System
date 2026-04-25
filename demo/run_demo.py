#!/usr/bin/env python3
"""Side-by-side demo: nearest vs capacity-aware under outcome reward."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List

# Project paths
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SERVER = os.path.join(_ROOT, "server")
if _SERVER not in sys.path:
    sys.path.insert(0, _SERVER)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ys = sorted(values)
    idx = int(round(0.95 * (len(ys) - 1)))
    return ys[idx]


def _rollout(
    task: str,
    seed: int,
    policy_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    narrate: bool = False,
) -> Dict[str, Any]:
    from dispatch_grid_environment import DispatchGridEnvironment
    from models import DispatchGridAction

    env = DispatchGridEnvironment(task=task)
    obs = env.reset(seed=seed, difficulty=task, reward_mode="outcome")
    etas: List[float] = []
    harms: List[float] = []
    diversions = 0
    reward_sum = 0.0
    steps = 0
    while True:
        od = obs.model_dump()
        if od.get("done") or od.get("call_id") == "DONE":
            break
        ad = policy_fn(od)
        act = DispatchGridAction(**ad)
        if narrate:
            print(
                f"  [{od.get('call_id')}] zone={od.get('current_zone')} "
                f"traffic={od.get('traffic_regime')} -> "
                f"amb={ad['ambulance_units']} pol={ad['police_units']} fire={ad['fire_units']} "
                f"hospital={ad['hospital_choice']}"
            )
        obs = env.step(act)
        od2 = obs.model_dump()
        reward_sum += float(od2.get("last_action_reward", 0.0))
        etas.append(float(od2.get("last_eta_minutes", 0.0)))
        harms.append(float(od2.get("last_harm", 0.0)))
        if od2.get("last_diverted"):
            diversions += 1
        steps += 1
        if od2.get("done"):
            break
        if steps > 30:
            break

    eta_nz = [e for e in etas if e > 0.01]
    mean_eta = sum(eta_nz) / len(eta_nz) if eta_nz else 0.0
    return {
        "mean_eta_minutes": round(mean_eta, 3),
        "p95_eta_minutes": round(_p95(eta_nz), 3),
        "diversion_count": diversions,
        "total_harm": round(sum(harms), 4),
        "episode_reward_sum": round(reward_sum, 4),
        "steps": steps,
    }


def run_comparison_metrics(seed: int = 0, task: str = "hard") -> Dict[str, Any]:
    """Used by demo CLI and GET /demo/compare."""
    from demo.policies import capacity_aware_policy, nearest_policy

    nearest = _rollout(task, seed, nearest_policy, narrate=False)
    smart = _rollout(task, seed, capacity_aware_policy, narrate=False)
    return {
        "seed": seed,
        "task": task,
        "reward_mode": "outcome",
        "nearest_policy": nearest,
        "capacity_aware_policy": smart,
    }


def _save_chart(metrics: Dict[str, Any], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_metrics = ("mean_eta_minutes", "p95_eta_minutes", "diversion_count", "total_harm")
    labels = ("Mean ETA", "P95 ETA", "Diversions", "Total harm")
    near = metrics["nearest_policy"]
    sm = metrics["capacity_aware_policy"]
    x = range(len(n_metrics))
    width = 0.35
    v0 = [near[k] for k in n_metrics]
    v1 = [sm[k] for k in n_metrics]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - width / 2 for i in x], v0, width, label="nearest")
    ax.bar([i + width / 2 for i in x], v1, width, label="capacity_aware")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title(f"Dispatch MVE — seed={metrics['seed']} task={metrics['task']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="hard", choices=["easy", "medium", "hard", "crisis"])
    parser.add_argument("--narrate", action="store_true")
    args = parser.parse_args()

    from demo.policies import capacity_aware_policy, nearest_policy

    print(f"[demo] task={args.task} seed={args.seed} reward_mode=outcome")
    print("--- nearest_policy ---")
    _rollout(args.task, args.seed, nearest_policy, narrate=args.narrate)
    print("--- capacity_aware_policy ---")
    _rollout(args.task, args.seed, capacity_aware_policy, narrate=args.narrate)

    metrics = run_comparison_metrics(seed=args.seed, task=args.task)
    demo_dir = os.path.join(_ROOT, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    json_path = os.path.join(demo_dir, "metrics.json")
    png_path = os.path.join(demo_dir, "comparison.png")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[demo] wrote {json_path}")
    try:
        _save_chart(metrics, png_path)
        print(f"[demo] wrote {png_path}")
    except ModuleNotFoundError:
        print(
            "[demo] matplotlib not installed; skipped comparison.png "
            "(install with: pip install matplotlib or uv sync --extra dev)"
        )


if __name__ == "__main__":
    main()
