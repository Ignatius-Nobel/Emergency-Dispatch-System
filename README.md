# Emergency Response Dispatch System

## Introduction

This project is an **OpenEnv**-based simulation where an agent plays the role of an **emergency dispatcher**: it reads incoming call descriptions and chooses how to allocate **ambulance**, **police**, and **fire** resources, with priorities and optional backup, under the same kind of pressure as a simplified real dispatch desk. The stack includes a FastAPI server, `openenv-core` client types, optional **reinforcement learning** training (Unsloth + Hugging Face TRL / GRPO), and a path to run the same environment in **Docker** or on **Hugging Face Spaces**. The sections below describe the **problem** we model, how the system is **structured**, and how to **run, train, and deploy** it. **External links** (live demo, source, Colab) are collected at the end under [Resources](#resources).

## Problem

Real dispatch is constrained: wrong unit types, bad priority, or missing backup can worsen outcomes, while over-committing resources leaves the next call uncovered. This repository encodes that tradeoff in a **simulated** grid. Each **episode** is a short sequence of calls; the agent must output a legal dispatch (unit counts, priority, backup) **within a finite resource budget** and across **task difficulties** (from clear single-type incidents to ambiguous and multi-hazard situations).

The goal of the project is a **reproducible** environment: same HTTP API in the lab, in the cloud, or on a public Space, so baselines, trained policies, and human-readable feedback stay comparable. Details of the **rubric-based reward** and optional **dynamics** (e.g. traffic, hospital capacity) appear in the reference sections below.

## How it works

### Environment and interface

The server follows an **OpenEnv**-style contract: `POST /reset` and `POST /step` with session tracking via `X-Session-Id`, so training scripts, the web UI, and third-party clients share one protocol. A **hosted** build is available on Hugging Face; URLs are in [Resources](#resources). Reward defaults to the rubric described under [Reward Structure](#reward-structure); extended demos (traffic, bed capacity, `reward_mode="outcome"`) are documented in the [Demo](#demo-mve--traffic-eta-hospital-beds-outcome-reward) section.

### Training

We ship a **Unsloth** + **Hugging Face TRL** pipeline using **GRPO** in [`training/train_dispatch.py`](training/train_dispatch.py). The script starts a **local** OpenEnv-compatible FastAPI server in a background thread and trains against it with [`openenv_http_session.py`](openenv_http_session.py) (same process as the trainer, which is convenient in **Google Colab** so each environment step is not sent over the public internet). An end-to-end Colab path is in [`training/dispatch_rl_training.ipynb`](training/dispatch_rl_training.ipynb) (install cell clones the repository and runs the training script).

### Training outputs

After a full run, the training script writes **artifacts** under `training/`:

- `reward_curve.png` — mean reward vs training step  
- `loss_curve.png` — training loss vs step  
- `comparison.json` — baseline (random) vs trained mean episode score and improvement  

Generate them with `python training/train_dispatch.py` (or the Colab notebook). Plots are shown here when the files exist in the repo:

![Reward curve](training/reward_curve.png)

![Loss curve](training/loss_curve.png)

*If the images do not load, the PNGs are not in your tree yet—run training to create them.*

## Quick Start

The simplest way to use the Dispatch Grid environment is through the `DispatchGridEnv` class:

```python
from dispatch_grid import DispatchGridAction, DispatchGridEnv

try:
    # Create environment from Docker image
    env = DispatchGridEnv.from_docker_image("dispatch_grid-env:latest")

    # Reset to start a new episode
    result = env.reset()
    obs = result.observation
    print(f"Call ID: {obs.call_id}")
    print(f"Description: {obs.call_description}")
    print(f"Severity: {obs.severity}")

    # Take an action
    action = DispatchGridAction(
        ambulance_units=1,
        police_units=0,
        fire_units=0,
        priority_level=3,
        backup_requested=False,
    )
    result = env.step(action)
    print(f"Reward: {result.reward}")
    print(f"Feedback: {result.observation.last_action_feedback}")

finally:
    # Always clean up
    env.close()
```

That's it! The `DispatchGridEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t dispatch_grid-env:latest -f server/Dockerfile .
```

## Demo (MVE — traffic, ETA, hospital beds, outcome reward)

Run a side-by-side comparison of **nearest** vs **capacity-aware** routing under `reward_mode="outcome"` (same seed, same episode). Writes `demo/metrics.json`; writes `demo/comparison.png` if `matplotlib` is installed (`uv sync --extra dev` or `pip install matplotlib`).

```bash
python3 demo/run_demo.py --seed 0 --task hard
```

With the HTTP server running, JSON metrics are also available at `GET /demo/compare?seed=0&task=hard`.

**Raw HTTP (`requests`, curl):** `POST /reset` returns a header `X-Session-Id`. Send that header on every `POST /step` and `GET /state` for the same episode. Step body must be `{"action": { ... }}`. Helpers: [`openenv_http_session.py`](openenv_http_session.py). WebSocket clients (e.g. `DispatchGridEnv().sync()`) are unchanged.

**Talking points (2–3 minutes):**

1. Fake traffic regimes (`light` / `normal` / `heavy`) scale travel time; observations expose `traffic_regime` and `last_eta_minutes`.
2. Medical calls consume **ICU** or **general** beds; lack of capacity triggers **diversion** (+8 minutes) and `last_overflow_penalty`, surfaced in the observation.
3. Default `reward_mode="rubric"` is unchanged for OpenEnv graders; use `reset(..., reward_mode="outcome")` for the harm-based proxy used in the demo.

## Action Space

**DispatchGridAction** - Decide how to respond to each emergency call:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `ambulance_units` | int | 0-5 | Number of ambulance units (medical emergencies) |
| `police_units` | int | 0-5 | Number of police units (crime, violence, threats) |
| `fire_units` | int | 0-5 | Number of fire units (fire, hazmat, rescue) |
| `priority_level` | int | 1-4 | Response priority (1=low, 4=critical) |
| `backup_requested` | bool | true/false | Request backup from neighboring districts |

**Valid dispatch combinations:**
- Single type: ambulance only, police only, or fire only
- Two types: ambulance+police, ambulance+fire, police+fire
- All types: ambulance+police+fire (for major incidents)

## Observation Space

**DispatchGridObservation** - Information available for each decision:

| Field | Type | Description |
|-------|------|-------------|
| `call_id` | str | Unique identifier for the emergency call |
| `incident_type` | str | Category: medical, fire, crime, accident, multi-hazard |
| `call_description` | str | Detailed description of the emergency |
| `location` | str | Address/location of incident |
| `caller_info` | str | Information about the caller |
| `severity` | str | Observed severity: minor/moderate/severe/critical |
| `calls_handled` | int | Number of calls handled in current episode |
| `total_calls` | int | Total calls in this episode (4) |
| `calls_remaining` | int | Calls remaining in episode |
| `cumulative_score` | float | Running total score |
| `last_action_reward` | float | Reward from previous action |
| `last_action_feedback` | str | Human-readable feedback on previous action |
| `available_ambulances` | int | Remaining ambulance units (starts at 5) |
| `available_police` | int | Remaining police units (starts at 8) |
| `available_fire` | int | Remaining fire units (starts at 4) |
| `avg_response_time_minutes` | float | Average response time across handled calls |

## Reward Structure

The reward function evaluates dispatch decisions across multiple dimensions:

### Per-Call Reward (max 0.60)

| Component | Max | Scoring |
|-----------|-----|---------|
| **Unit Type Accuracy** (×3) | 0.30 | +0.10 per correct type with sufficient units<br>+0.05 per correct type but too few units<br>-0.10 per wrong type sent<br>-0.10 per needed type not sent |
| **Priority Level** | 0.20 | +0.20 exact match<br>+0.10 off by 1<br>-0.10 off by 2+ |
| **Backup Decision** | 0.10 | +0.10 correct decision<br>-0.05 wrong decision |

### Episode Bonus

| Component | Value | Condition |
|-----------|-------|-----------|
| **Completion Bonus** | +0.20 | Awarded once when all 4 calls are handled |

**Maximum possible score per episode:** 4 × 0.60 + 0.20 = 2.60

## Task Difficulties

The environment provides three difficulty levels:

### Easy: Basic Emergency Dispatch
- Single-type emergencies with clear descriptions
- Only one unit type needed per call
- No backup requests required
- 8 possible calls, 4 sampled per episode

### Medium: Ambiguous & Multi-Type Dispatch
- Calls requiring multiple unit types simultaneously
- Resource constraints and incomplete information
- Some calls require backup
- 6 possible calls, 4 sampled per episode

### Hard: Multi-Hazard Cascading Emergencies
- Complex incidents: explosions, active shooters, disasters
- All unit types required
- Backup always needed
- 4 possible calls, all used per episode

## Running Inference

The environment includes a baseline inference script using OpenAI-compatible APIs:

```bash
# Set required environment variables
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_token_here"

# Run inference on all tasks
python inference.py --tasks easy medium hard

# Run with custom output
python inference.py --output results.json --quiet
```

## Deploying to Hugging Face Spaces

A **public hosted instance** of the app and its Space page are listed under [Resources](#resources). To publish or update the Space from your own checkout, use `openenv push` as below.

You can deploy your environment to Hugging Face Spaces:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

### Prerequisites
- Authenticate with Hugging Face: `huggingface-cli login`

### Options
- `--directory`, `-d`: Directory containing the OpenEnv environment
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name'
- `--base-image`, `-b`: Base Docker image to use
- `--private`: Deploy as private space

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint

## Advanced Usage

### Connecting to an Existing Server

```python
from dispatch_grid import DispatchGridEnv

# Connect to existing server
env = DispatchGridEnv(base_url="http://localhost:8000")

# Use as normal
result = env.reset()
result = env.step(DispatchGridAction(
    ambulance_units=1,
    police_units=0,
    fire_units=0,
    priority_level=3,
    backup_requested=False,
))
```

Note: When connecting to an existing server, `env.close()` will NOT stop the server.

### Context Manager Usage

```python
from dispatch_grid import DispatchGridAction, DispatchGridEnv

with DispatchGridEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    for msg in ["Call 1", "Call 2", "Call 3"]:
        result = env.step(DispatchGridAction(
            ambulance_units=1,
            police_units=0,
            fire_units=0,
            priority_level=3,
            backup_requested=False,
        ))
```

## Project Structure

The repository is laid out at the project root. When you `pip install` the package, [pyproject.toml](pyproject.toml) maps the installable name **`dispatch_grid`** to this tree (`dispatch_grid` → `.`, `dispatch_grid.server` → `server/`), so `from dispatch_grid import DispatchGridAction, DispatchGridEnv` works as in the examples.

```
.
├── openenv.yaml                # OpenEnv manifest
├── pyproject.toml
├── models.py                    # Pydantic action / observation types
├── client.py                    # DispatchGridEnv (OpenEnv HTTP client)
├── openenv_http_session.py      # REST helpers (session id, reset/step)
├── inference.py                 # OpenAI-compatible LLM inference
├── test.py / demo/              # Optional connectivity & routing demos
├── server/
│   ├── app.py                   # FastAPI app, routes, /demo, graders hooks
│   ├── dispatch_grid_environment.py
│   ├── Dockerfile
│   └── requirements.txt
├── training/
│   ├── train_dispatch.py        # Unsloth + TRL GRPO training
│   └── dispatch_rl_training.ipynb
├── graders/                     # OpenEnv task grader classes
└── tests/                       # pytest
```

## Development & Testing

### Direct Environment Testing

```bash
# From project root
PYTHONPATH=src python3 server/dispatch_grid_environment.py
```

### Running Locally

```bash
uvicorn server.app:app --reload
```

### Validation

```bash
# Build the environment
openenv build

# Validate against OpenEnv spec
openenv validate --verbose
```

## Resources

Links outside this repository:

| | |
|--|--|
| **Live app (Hugging Face Space)** | [https://ignatius-nobel-Emergency-Dispatch-System.hf.space](https://ignatius-nobel-Emergency-Dispatch-System.hf.space) |
| **Hugging Face Space (project page)** | [https://huggingface.co/spaces/ignatius-nobel/Emergency-Dispatch-System](https://huggingface.co/spaces/ignatius-nobel/Emergency-Dispatch-System) |
| **Source (GitHub)** | [https://github.com/Ignatius-Nobel/Emergency-Dispatch-System](https://github.com/Ignatius-Nobel/Emergency-Dispatch-System) |
| **Training in Colab** | [Open in Colab](https://colab.research.google.com/github/Ignatius-Nobel/Emergency-Dispatch-System/blob/main/training/dispatch_rl_training.ipynb) — `training/dispatch_rl_training.ipynb` |
| **Written overview (Hugging Face)** | Draft: [`docs/hackathon_blog.md`](docs/hackathon_blog.md) — *after you publish a post, add its public URL here* |
| **Long-form: RL in emergency dispatch** | [`docs/blog_rl_emergency_dispatch.md`](docs/blog_rl_emergency_dispatch.md) |
| **Video walkthrough (optional)** | *Add a public YouTube or other link when available* |
