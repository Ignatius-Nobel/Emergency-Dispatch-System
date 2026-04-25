# Emergency Dispatch System — Claude Code Project Context

## What this project is
An OpenEnv RL environment (`server/dispatch_grid_environment.py`) for a hackathon.
An AI agent acts as an emergency dispatcher allocating ambulance / police / fire units.
The project needs fixes and new deliverables to meet judging criteria.

## Repo layout
```
server/
  dispatch_grid_environment.py   ← main env file (all fixes go here)
  models.py                      ← EmergencyCall, DispatchGridAction, DispatchGridObservation
tests/
  test_env.py                    ← unit tests (create if missing)
training/
  dispatch_rl_training.ipynb     ← Colab training notebook (create if missing)
  train_dispatch.py              ← standalone training script (create if missing)
FIXES.md                         ← tracks which fixes have been applied
```

## Coding rules
- Python 3.10+. Type hints on all new functions.
- Never remove existing calls — only patch the specific lines described in each task.
- After every file edit, run `python -m pytest tests/ -q` and confirm it passes.
- Commit message format: `fix: <short description>` or `feat: <short description>`.
- Do not auto-format the entire file — only touch the lines required by the task.

## Task execution order
Always work tasks in this sequence. Do NOT skip ahead.
1. FIX-1  — double-count bug            (server/dispatch_grid_environment.py)
2. FIX-2  — EmergencyCall model fields  (server/models.py)
3. FIX-3  — reward hacking safeguards   (server/dispatch_grid_environment.py)
4. FIX-4  — over-dispatch reward curve  (server/dispatch_grid_environment.py)
5. TEST-1 — unit test suite             (tests/test_env.py)
6. TRAIN-1 — training script            (training/train_dispatch.py)
7. NB-1   — Colab notebook              (training/dispatch_rl_training.ipynb)

Mark each task DONE in FIXES.md immediately after completing it.

## Definition of done for each task
A task is done when:
- [ ] Code change is applied exactly as specified in the task file
- [ ] `python -m pytest tests/ -q` exits 0
- [ ] FIXES.md is updated with status DONE and a one-line summary
