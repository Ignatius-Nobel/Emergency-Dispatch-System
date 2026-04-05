# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Emergency Response Dispatch System — an OpenEnv RL environment where an AI agent acts as an emergency dispatcher, receiving emergency calls and deciding which resources to deploy (ambulance, police, fire) with appropriate priority levels and coordination decisions.

## Quick Start

```bash
# Build Docker image
docker build -t dispatch_grid-env:latest -f server/Dockerfile .

# Run server locally
uvicorn server.app:app --reload

# Run inference
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_token"
python inference.py --tasks easy medium hard crisis
```

## OpenEnv Commands

```bash
# Build and validate
openenv build
openenv validate --verbose

# Deploy to Hugging Face Spaces
openenv push --namespace <org> --private
```

## Architecture

```
dispatch_grid/
├── models.py                     # Pydantic Action/Observation schemas
├── client.py                     # EnvClient subclass for HTTP/WebSocket comms
├── inference.py                  # LLM inference via OpenAI-compatible APIs
├── server/
│   ├── app.py                    # FastAPI server (/reset, /step, /tasks, /grader, /baseline)
│   └── dispatch_grid_environment.py  # Core environment logic + call database
```

## Key Components

**Action Space** (`models.py:16`):
- `ambulance_units`, `police_units`, `fire_units` (0-5)
- `priority_level` (1-4)
- `hospital_choice`: "nearest" | "regional" | "auto"
- `coordination_level`: "none" | "mutual_aid" | "mci_protocol"
- `ambulance_staging`: "dispatch" | "stage_nearby" | "on_scene_hold"

**Observation Space** (`models.py:122`):
- Call info (ID, type, description, severity, location)
- Resource counts (ambulances, police, fire units available)
- Hospital status (beds, trauma capability, distance)
- Episode state (calls handled, cumulative score, feedback)

**Tasks** (`server/app.py:68`):
- `easy`: Single-type emergencies (8 calls → 4 sampled)
- `medium`: Multi-type, ambiguous calls (6 calls → 4 sampled)
- `hard`: Multi-hazard cascading incidents (4 calls, all used)
- `crisis`: Resource-strained scenarios with hospital capacity constraints

**Reward** (`server/dispatch_grid_environment.py:400`):
- Unit type accuracy (×3 types): max 0.30
- Priority level match: max 0.20
- Coordination decision: max 0.15
- Hospital routing: max 0.15
- Staging appropriateness: max 0.10
- Episode completion bonus: +0.20
- Max per episode: 4 × 0.90 + 0.20 = 3.80

## Development

```bash
# Direct environment testing
PYTHONPATH=src python3 server/dispatch_grid_environment.py

# Install dev dependencies
pip install -e ".[dev]"
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM endpoint (e.g., Groq, OpenAI) |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face token for deployment |
