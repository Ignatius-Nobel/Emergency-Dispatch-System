# Fix tracker

Update this file after every completed task. Format: `[DONE]` or `[PENDING]`.

| ID      | Status  | File(s) changed                          | Notes |
|---------|---------|------------------------------------------|-------|
| FIX-1   | [DONE]  | server/dispatch_grid_environment.py      | Completion bonus: cumulative once, reward clamped; no double-count vs sum of step rewards |
| FIX-2   | [DONE]  | server/dispatch_grid_environment.py, models.py, server/models.py | EmergencyCall already had defaults in `models.py`; `getattr` replaced with direct attributes; `server/models.py` re-exports root models for `sys.path` = server |
| FIX-3   | [DONE]  | server/dispatch_grid_environment.py      | `MAX_STEPS_PER_EPISODE=20`, step-limit return, action validation with -0.2 on invalid |
| FIX-4   | [DONE]  | server/dispatch_grid_environment.py      | Over-dispatch scaled: `adjusted = max(-0.05, 0.075 - (excess-1)*0.02)` |
| TEST-1  | [DONE]  | tests/test_env.py                        | 16 passed (pytest) |
| TRAIN-1 | [DONE]  | training/train_dispatch.py             | GRPO training script + reward curve + comparison.json |
| NB-1    | [DONE]  | training/dispatch_rl_training.ipynb      | Colab-oriented notebook (8 cells) |
| MVE-1   | [DONE]  | models.py, server/dispatch_grid_environment.py, demo/*, server/app.py, client.py, tests/test_env_v2.py, pyproject.toml | Lean dynamics: zone ETA, fake traffic, ICU/general bed consumption + diversion; optional `reward_mode=outcome`; scripted SURGE call for hard/crisis; `GET /demo/compare`; demo CLI + tests |
| HTTP-1  | [DONE]  | server/app.py, openenv_http_session.py, training/train_dispatch.py, baseline_groq.py, README.md | Stateful REST: `X-Session-Id` for `/reset`+`/step`+`/state` (replaces stateless openenv create_app handlers); helpers + training/baseline use wrapped `action` + session header |

**Note:** The FIX-4 inline verification snippet in `tasks/FIX-4-over-dispatch-reward.md` uses `r - 0.20` as a proxy for “unit-only” reward; full `compute_reward` also scores coordination, hospital, and staging, so that snippet does not assert on the full total. Automated coverage for FIX-4 is in `tests/test_env.py` (`test_over_dispatch_*`).
