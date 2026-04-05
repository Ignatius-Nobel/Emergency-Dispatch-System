---
title: Emergency Response Dispatch System
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - emergency-response
  - rl-environment
---

# Emergency Response Dispatch System

A realistic simulation environment for training AI agents in emergency dispatch decision-making. The agent receives emergency calls and must decide which resources to dispatch (ambulance, police, fire), at what priority level, and whether to request backup.

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

```
dispatch_grid/
├── __init__.py                 # Module exports
├── README.md                   # This file
├── openenv.yaml                # OpenEnv manifest
├── pyproject.toml              # Project metadata and dependencies
├── inference.py                # LLM inference script
├── client.py                   # DispatchGridEnv client
├── models.py                   # Action and Observation models
└── server/
    ├── __init__.py             # Server module exports
    ├── dispatch_grid_environment.py  # Core environment logic
    ├── app.py                  # FastAPI application
    └── Dockerfile              # Container image definition
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
