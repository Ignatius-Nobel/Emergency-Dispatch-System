Perfect — all three spaces touched, strong backend team, and all criteria matter equally. That's actually the ideal profile for these enhancements. Let me give you a concrete, prioritized implementation plan.

---

## Master Enhancement Plan

### What We're Building

Three tightly integrated enhancements that touch action, observation, and reward together:

```
Hospital Bed Tracking  ──┐
                          ├──► New Observation Fields
Coordination Tiers     ──┤    New Action Fields  
                          ├──► New Reward Components
Resource Replenishment ──┘    New Task Difficulty Layer
```

---

## 1. Action Space Changes

Your current action has 5 fields. The enhanced version adds 3 more:

```python
# CURRENT
ambulance_units: int       # 0-5
police_units: int          # 0-5
fire_units: int            # 0-5
priority_level: int        # 1-4
backup_requested: bool     # True/False

# ENHANCED — additions
hospital_choice: str       # "nearest" | "regional" | "auto"
coordination_level: str    # "none" | "mutual_aid" | "mci_protocol"
ambulance_staging: str     # "dispatch" | "stage_nearby" | "on_scene_hold"
```

**Why these three specifically:**

- `hospital_choice` — directly powers idea #1. Agent must weigh bed availability vs. distance.
- `coordination_level` — replaces the binary `backup_requested` with a 3-tier system that has different cost/benefit tradeoffs.
- `ambulance_staging` — adds a tactical depth layer. Staging nearby without full commitment is a real dispatcher tactic for uncertain scenes (active shooters, chemical unknowns).

---

## 2. Observation Space Changes

```python
# HOSPITAL BED TRACKING — Idea #1
hospital_a_name: str              # "City General"
hospital_a_icu_beds: int          # 0-10
hospital_a_general_beds: int      # 0-30
hospital_a_trauma_capable: bool
hospital_a_distance_minutes: int  # travel time in minutes

hospital_b_name: str              # "Regional Medical Center"
hospital_b_icu_beds: int
hospital_b_general_beds: int
hospital_b_trauma_capable: bool
hospital_b_distance_minutes: int

nearest_hospital: str             # "A" or "B"
recommended_hospital: str         # system recommendation based on capacity + severity

# COORDINATION STATE — Idea #2
district_reserve_units: int       # shared mutual-aid pool, depletes on use (starts at 6)
mci_protocol_active: bool         # True if MCI has been declared this episode
coordination_cost_remaining: int  # how many more mutual_aid requests are viable

# RESOURCE REPLENISHMENT (Enhancement #3)
ambulances_returning_in: int      # units returning after current call (simulated)
police_returning_in: int
fire_returning_in: int
```

**The key insight:** The agent now has to reason across three dimensions simultaneously — what's the incident, what resources do I have, and where can I send patients. This is a genuine sequential decision problem, not just per-call classification.

---

## 3. Reward Function Changes

The current reward max per call is **0.60**. The enhanced version raises it to **0.90**, rewarding smarter decisions:

```
CURRENT components (0.60 max):
  Unit type accuracy    × 3    → max 0.30
  Priority level               → max 0.20
  Backup decision              → max 0.10

ENHANCED components (0.90 max):
  Unit type accuracy    × 3    → max 0.30  (unchanged)
  Priority level               → max 0.20  (unchanged)
  Coordination decision        → max 0.15  (replaces binary backup)
  Hospital routing             → max 0.15  (new)
  Staging appropriateness      → max 0.10  (new)
```

**Coordination decision scoring (replaces `backup_requested`):**
```
needs_mci=True,  chose mci_protocol  → +0.15
needs_mci=False, chose mutual_aid    → +0.10  (partial — overkill but not wrong)
needs_mci=False, chose none          → +0.15
chose mci_protocol when not needed   → -0.10  (wastes district reserve)
chose none when mutual_aid needed    → -0.08
chose none when mci needed           → -0.15
```

**Hospital routing scoring:**
```
correct_hospital == "regional", chose "regional"  → +0.15
correct_hospital == "nearest",  chose "nearest"   → +0.15
correct_hospital == "regional", chose "nearest"   → -0.10  (sent critical to full hospital)
correct_hospital == "nearest",  chose "regional"  → -0.05  (unnecessary delay)
chose "auto"                                      → +0.08  (always safe, never optimal)
```

This scoring design is deliberate — `auto` is always available as a safe fallback, but an agent that learns the actual hospital states will consistently outscore one that defaults to auto.

---

## 4. Task Difficulty Redesign

The three existing tasks need richer initial states:

**Easy** — beds plentiful, coordination never needed, replenishment fast
```python
hospital_a = {"icu": 8, "general": 25, "trauma": True,  "distance": 5}
hospital_b = {"icu": 6, "general": 20, "trauma": False, "distance": 12}
district_reserve = 6
replenishment_rate = 2  # units return every 2 calls
```

**Medium** — one hospital strained, mutual_aid occasionally correct
```python
hospital_a = {"icu": 2, "general": 8,  "trauma": True,  "distance": 5}
hospital_b = {"icu": 5, "general": 18, "trauma": True,  "distance": 15}
district_reserve = 3
replenishment_rate = 3
```

**Hard** — both hospitals strained, mci_protocol correct for cascading calls
```python
hospital_a = {"icu": 1, "general": 3,  "trauma": True,  "distance": 5}
hospital_b = {"icu": 0, "general": 6,  "trauma": False, "distance": 20}
district_reserve = 2
replenishment_rate = 4  # slow replenishment increases pressure
```

**New: "Crisis" task (4th task)** — both hospitals full, district reserve = 0, cascading incidents
```python
hospital_a = {"icu": 0, "general": 1,  "trauma": True,  "distance": 5}
hospital_b = {"icu": 0, "general": 0,  "trauma": True,  "distance": 25}
district_reserve = 0
replenishment_rate = 5
# + 2 of the 6 calls are cascade spawns from earlier decisions
```

Adding a 4th task also strengthens the submission since the requirement is "minimum 3" — going to 4 signals depth.

---

## 5. New Emergency Calls to Add

Your current call database needs entries that specifically test the new mechanics:

**Hospital routing calls** (where choice of hospital matters):
- Severe burns patient — needs trauma center, nearest hospital is not trauma-capable
- Multi-organ trauma — ICU required, Hospital A has 0 ICU beds
- Minor laceration during a busy period — nearest is strained, but this patient doesn't need ICU

**Coordination calls** (where tier matters):
- Stadium crowd crush — 40+ potential casualties, MCI clearly needed
- Two-block radius domestic disturbances (3 simultaneous) — mutual aid appropriate, not MCI
- Single armed robbery — coordination_level should be none, agent penalized for over-escalating

---

## 6. Implementation Sequence

Given your backend strength and the deadline, here's the exact order to tackle this:

```
Day 1-2:  Fix critical gaps (inference.py, OpenAI client, README)
Day 3:    Update models.py — new action + observation fields
Day 4:    Update dispatch_grid_environment.py — hospital state + replenishment logic
Day 5:    Update compute_reward() — new scoring components
Day 6:    Add new emergency calls to the database
Day 7:    Add crisis task, update openenv.yaml
Day 8:    Run pre-validation script, test full episode end-to-end
```

---

## How This Scores Against Every Judging Criterion

| Criterion | Current state | After enhancements |
|---|---|---|
| **Runtime correctness** | Passes | Passes — all additions are pure Python, no new infra |
| **Interface compliance** | Passes | Passes — still standard OpenEnv step/reset/state |
| **Task design** | 3 tasks, clear | 4 tasks, hospital + coordination state makes each call genuinely contextual |
| **Grading logic** | Reward makes sense | Richer reward with 5 components, each independently justifiable |
| **Real-world simulation** | Good | Excellent — hospital capacity and mutual-aid tiers are direct real-world mechanics |

---

Ready to start implementing? I'd suggest beginning with `models.py` since every other file depends on it — want me to write the updated version first?