# Dispatch Grid Environment - Enhancement & Gap Fix Plan

## Context

This plan consolidates two critical workstreams for the dispatch_grid OpenEnv environment:

### Workstream 1: Gap Analysis Fixes (Critical - Disqualifying if not fixed)
From `docs/GAP_ANALYSIS.md`, the hackathon submission has 4 disqualifying gaps:
1. **Wrong script name**: `baseline_groq.py` exists, but hackathon requires `inference.py` at root
2. **Wrong LLM client**: Uses `groq` library, must use OpenAI client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars
3. **README describes wrong environment**: Currently documents "echo environment" with `echoed_message`, `message_length`, not the dispatch system
4. **Missing required env vars**: No reference to `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` anywhere in codebase

### Workstream 2: Feature Upgrades (Scoring impact)
From `docs/FEATURE_UPGRADE.md`, three tightly integrated enhancements:
1. **Hospital Bed Tracking**: Agent must choose between hospitals based on capacity/distance
2. **Coordination Tiers**: Replace binary `backup_requested` with 3-tier system (none/mutual_aid/mci_protocol)
3. **Resource Replenishment**: Units return over time, adding sequential pressure

These enhancements touch action, observation, and reward together, creating genuine sequential decision problems.

---

## Implementation Plan

### Phase 1: Critical Gap Fixes (Do First - Disqualifying if skipped)

#### 1.1 Create `inference.py` at root
**File**: `/Users/ignatius/Desktop/dispatch_grid/inference.py`

Replace `baseline_groq.py` with OpenAI client-based script:
- Use `openai.OpenAI()` with `base_url=os.environ.get("API_BASE_URL")`
- Read `MODEL_NAME` from env vars
- Read `HF_TOKEN` from env vars (for Hugging Face integration if needed)
- Keep same CLI interface (`--tasks`, `--base-url`, `--output`, `--quiet`)
- Fix normalization bug: max = `n * 0.60 + 0.20` (episode bonus once, not per-call)

#### 1.2 Rewrite README.md
**File**: `/Users/ignatius/Desktop/dispatch_grid/README.md`

Complete rewrite to describe actual dispatch environment:
- Remove all "echo" references
- Document actual action space: `ambulance_units`, `police_units`, `fire_units`, `priority_level`, `backup_requested`
- Document actual observation space: `call_id`, `incident_type`, `call_description`, `severity`, resource counts, etc.
- Document actual reward structure (0.60 max per call + 0.20 episode bonus)
- Update quickstart examples with correct dispatch actions
- Remove Hugging Face deployment section if not needed

#### 1.3 Update `pyproject.toml`
**File**: `/Users/ignatius/Desktop/dispatch_grid/pyproject.toml`

- Replace `groq` dependency with `openai`
- Ensure `openai>=1.0.0` in dependencies

---

### Phase 2: Feature Upgrades (Core Enhancement Work)

#### 2.1 Update `models.py`
**File**: `/Users/ignatius/Desktop/dispatch_grid/models.py`

**DispatchGridAction** - Add 3 new fields:
```python
hospital_choice: str  # "nearest" | "regional" | "auto"
coordination_level: str  # "none" | "mutual_aid" | "mci_protocol"
ambulance_staging: str  # "dispatch" | "stage_nearby" | "on_scene_hold"
```

**DispatchGridObservation** - Add hospital + coordination state:
```python
# Hospital tracking
hospital_a_name: str
hospital_a_icu_beds: int
hospital_a_general_beds: int
hospital_a_trauma_capable: bool
hospital_a_distance_minutes: int

hospital_b_name: str
hospital_b_icu_beds: int
hospital_b_general_beds: int
hospital_b_trauma_capable: bool
hospital_b_distance_minutes: int

nearest_hospital: str  # "A" or "B"
recommended_hospital: str  # system recommendation

# Coordination state
district_reserve_units: int  # shared mutual-aid pool (starts at 6)
mci_protocol_active: bool
coordination_cost_remaining: int

# Resource replenishment
ambulances_returning_in: int  # calls until units return
police_returning_in: int
fire_returning_in: int
```

#### 2.2 Update `dispatch_grid_environment.py`
**File**: `/Users/ignatius/Desktop/dispatch_grid/server/dispatch_grid_environment.py`

**Hospital State Management**:
- Add hospital state dicts per task difficulty (Easy/Medium/Hard/Crisis)
- Track bed availability per episode
- Implement hospital choice logic in `step()`

**Coordination System**:
- Replace `needs_backup: bool` with `coordination_tier: str` in EmergencyCall
- Track `district_reserve_units` depletion
- Implement MCI protocol state

**Replenishment Logic**:
- Track units returning after N calls
- Replenish resources on return
- Expose `*_returning_in` counters in observation

**New Task Difficulty Configs**:
```python
# Easy - beds plentiful, coordination never needed
hospital_a = {"icu": 8, "general": 25, "trauma": True, "distance": 5}
hospital_b = {"icu": 6, "general": 20, "trauma": False, "distance": 12}
district_reserve = 6
replenishment_rate = 2

# Medium - one hospital strained, mutual_aid occasionally correct
hospital_a = {"icu": 2, "general": 8, "trauma": True, "distance": 5}
hospital_b = {"icu": 5, "general": 18, "trauma": True, "distance": 15}
district_reserve = 3
replenishment_rate = 3

# Hard - both strained, mci_protocol correct for cascading
hospital_a = {"icu": 1, "general": 3, "trauma": True, "distance": 5}
hospital_b = {"icu": 0, "general": 6, "trauma": False, "distance": 20}
district_reserve = 2
replenishment_rate = 4

# Crisis (NEW 4th task) - both full, reserve=0, cascade spawns
hospital_a = {"icu": 0, "general": 1, "trauma": True, "distance": 5}
hospital_b = {"icu": 0, "general": 0, "trauma": True, "distance": 25}
district_reserve = 0
replenishment_rate = 5
```

#### 2.3 Update `compute_reward()` function
**File**: `/Users/ignatius/Desktop/dispatch_grid/server/dispatch_grid_environment.py`

**New reward structure** (0.90 max vs current 0.60):
```python
Unit type accuracy × 3      → max 0.30  (unchanged)
Priority level              → max 0.20  (unchanged)
Coordination decision       → max 0.15  (replaces binary backup)
Hospital routing            → max 0.15  (new)
Staging appropriateness     → max 0.10  (new)
```

**Coordination scoring**:
- `needs_mci=True, chose mci_protocol` → +0.15
- `needs_mci=False, chose mutual_aid` → +0.10 (partial credit)
- `needs_mci=False, chose none` → +0.15
- `chose mci_protocol when not needed` → -0.10 (wastes reserve)
- `chose none when mutual_aid needed` → -0.08
- `chose none when mci needed` → -0.15

**Hospital routing scoring**:
- `correct_hospital == "regional", chose "regional"` → +0.15
- `correct_hospital == "nearest", chose "nearest"` → +0.15
- `correct_hospital == "regional", chose "nearest"` → -0.10 (sent critical to full hospital)
- `correct_hospital == "nearest", chose "regional"` → -0.05 (unnecessary delay)
- `chose "auto"` → +0.08 (safe fallback, never optimal)

#### 2.4 Add New Emergency Calls
**File**: `/Users/ignatius/Desktop/dispatch_grid/server/dispatch_grid_environment.py`
**Hospital routing calls**:
- Severe burns patient — needs trauma center, nearest is not trauma-capable
- Multi-organ trauma — ICU required, Hospital A has 0 ICU beds
- Minor laceration during busy period — nearest strained, patient doesn't need ICU

**Coordination calls**:
- Stadium crowd crush — 40+ casualties, MCI clearly needed
- Two-block radius domestic disturbances (3 simultaneous) — mutual_aid appropriate
- Single armed robbery — coordination should be `none`, penalize over-escalation

#### 2.5 Update `openenv.yaml`
**File**: `/Users/ignatius/Desktop/dispatch_grid/openenv.yaml`

- Add 4th task: `crisis`
- Update reward components to reflect new structure
- Update task descriptions

#### 2.6 Update `server/app.py`
**File**: `/Users/ignatius/Desktop/dispatch_grid/server/app.py`

- Add `crisis` task to `TASK_DEFINITIONS`
- Update `/tasks` endpoint with new action/observation fields
- Update `/grader` to handle new fields
- Update `/baseline` rule-based agent with hospital/coordination logic

#### 2.7 Update `client.py`
**File**: `/Users/ignatius/Desktop/dispatch_grid/client.py`

- Update `_step_payload()` to include new action fields
- Update `_parse_result()` to parse new observation fields

#### 2.8 Update `inference.py` prompt
**File**: `/Users/ignatius/Desktop/dispatch_grid/inference.py`

- Update system prompt with new action fields
- Update JSON schema for LLM response
- Add hospital choice + coordination level to prompt

---

### Phase 3: Validation & Testing

#### 3.1 Fix Normalization Bugs
**Files**: `baseline_groq.py` (for reference), `server/app.py`, `inference.py`

Ensure consistent normalization:
- Max per call = 0.60 (old) or 0.90 (new)
- Episode bonus = +0.20 once per episode, NOT per call
- Correct formula: `max_score = n * per_call_max + episode_bonus`

#### 3.2 Run Pre-Validation
```bash
# Verify in-repo imports
PYTHONPATH=src:envs uv run python -c "from server.dispatch_grid_environment import DispatchGridEnvironment"

# Build and validate
openenv build
openenv validate --verbose

# Smoke test server
uvicorn server.app:app --port 8000 &
curl http://localhost:8000/health
openenv validate --url http://localhost:8000
```

#### 3.3 Test Full Episode End-to-End
```bash
# Run inference script
API_BASE_URL=http://localhost:8000 MODEL_NAME=test HF_TOKEN=test python inference.py --tasks easy medium hard crisis
```

---

## Critical Files to Modify

| Priority | File | Changes |
|----------|------|---------|
| 🔴 | `inference.py` (CREATE NEW) | OpenAI client, env vars, fix normalization |
| 🔴 | `README.md` | Complete rewrite for dispatch system |
| 🔴 | `pyproject.toml` | Replace groq → openai dependency |
| 🔴 | `models.py` | Add 3 action fields + 14 observation fields |
| 🔴 | `dispatch_grid_environment.py` | Hospital state, coordination, replenishment, new tasks |
| 🟠 | `openenv.yaml` | Add crisis task, update reward docs |
| 🟠 | `server/app.py` | Update /tasks, /grader, /baseline endpoints |
| 🟠 | `client.py` | Update serialization for new fields |

---

## Assumptions

1. **Groq API compatibility**: The hackathon allows using Groq's OpenAI-compatible endpoint via `API_BASE_URL` - the key requirement is using OpenAI client, not the specific provider
2. **Backwards compatibility**: Existing tests/calls should still work with default values for new fields
3. **Episode length**: Still 4 calls per episode for all tasks including Crisis
4. **Crisis task**: May include cascade spawns (2 of 6 calls spawned from earlier decisions) - needs clarification

---

## Verification

After implementation:
1. `openenv build` passes without errors
2. `openenv validate --verbose` passes
3. All 4 tasks (easy/medium/hard/crisis) runnable via `inference.py`
4. README accurately describes action/observation spaces
5. Normalization consistent across `/grader`, `/baseline`, and `inference.py`

                                                                                                         