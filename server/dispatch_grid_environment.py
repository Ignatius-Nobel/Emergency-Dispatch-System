# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Emergency Response Dispatch System Environment.

Agent acts as an emergency dispatcher:
- Receives incoming emergency calls
- Decides exactly how many ambulance / police / fire units to send
- Gets rewarded based on correctness of each unit type + priority + backup

Tasks:
  easy   - Single-type emergencies, clear descriptions
  medium - Ambiguous calls, multiple unit types needed, resource constraints
  hard   - Multi-hazard cascading incidents, all unit types required
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import DispatchGridAction, DispatchGridObservation, EmergencyCall
except ImportError:
    try:
        from dispatch_grid.models import DispatchGridAction, DispatchGridObservation, EmergencyCall
    except ImportError:
        from ..models import DispatchGridAction, DispatchGridObservation, EmergencyCall


# ---------------------------------------------------------------------------
# Emergency Call Database
# correct_dispatch = {"ambulance": min_needed, "police": min_needed, "fire": min_needed}
# 0 means that unit type is NOT needed for this call
# ---------------------------------------------------------------------------

EASY_CALLS = [
    EmergencyCall(
        call_id="E001",
        incident_type="medical",
        description="Elderly woman, 78, collapsed at home. Unconscious, breathing but unresponsive. Neighbor called 911.",
        location="42 Maple Street, Riverside",
        caller_info="Concerned neighbor, calm",
        severity="severe",
        correct_dispatch={"ambulance": 1, "police": 0, "fire": 0},
        correct_priority=3,
        needs_backup=False,
    ),
    EmergencyCall(
        call_id="E002",
        incident_type="fire",
        description="Kitchen fire in apartment. Smoke visible from windows. Residents evacuating.",
        location="Sunset Apartments, Block C, Unit 14",
        caller_info="Building manager, panicked",
        severity="moderate",
        correct_dispatch={"ambulance": 0, "police": 0, "fire": 2},
        correct_priority=3,
        needs_backup=False,
    ),
    EmergencyCall(
        call_id="E003",
        incident_type="crime",
        description="Burglary in progress. Neighbor saw someone break window and enter house. Owners not home.",
        location="88 Oak Avenue",
        caller_info="Neighbor, whispering",
        severity="moderate",
        correct_dispatch={"ambulance": 0, "police": 2, "fire": 0},
        correct_priority=3,
        needs_backup=False,
    ),
    EmergencyCall(
        call_id="E004",
        incident_type="medical",
        description="25-year-old male, chest pain and difficulty breathing. No prior conditions reported.",
        location="Downtown Coffee Shop, 5th & Main",
        caller_info="Bystander, urgent",
        severity="severe",
        correct_dispatch={"ambulance": 1, "police": 0, "fire": 0},
        correct_priority=4,
        needs_backup=False,
    ),
    EmergencyCall(
        call_id="E005",
        incident_type="fire",
        description="Dumpster fire behind restaurant. Small but spreading toward wooden fence.",
        location="Rear alley, 220 Harbor Blvd",
        caller_info="Restaurant employee",
        severity="minor",
        correct_dispatch={"ambulance": 0, "police": 0, "fire": 1},
        correct_priority=2,
        needs_backup=False,
    ),
    EmergencyCall(
        call_id="E006",
        incident_type="crime",
        description="Domestic disturbance. Loud arguing and sounds of objects breaking reported by neighbors.",
        location="301 Pine Street, Apt 4B",
        caller_info="Downstairs neighbor",
        severity="moderate",
        correct_dispatch={"ambulance": 0, "police": 2, "fire": 0},
        correct_priority=2,
        needs_backup=False,
    ),
    EmergencyCall(
        call_id="E007",
        incident_type="medical",
        description="Child, approximately 6 years old, choking. Parent is attempting Heimlich but unsuccessful.",
        location="Greenfield Elementary School parking lot",
        caller_info="Parent, extremely distressed",
        severity="critical",
        correct_dispatch={"ambulance": 1, "police": 0, "fire": 0},
        correct_priority=4,
        needs_backup=False,
    ),
    EmergencyCall(
        call_id="E008",
        incident_type="crime",
        description="Armed robbery at convenience store. Suspect fled on foot. No injuries reported.",
        location="QuickMart, 9th and Central",
        caller_info="Store clerk, shaking",
        severity="severe",
        correct_dispatch={"ambulance": 0, "police": 3, "fire": 0},
        correct_priority=3,
        needs_backup=False,
    ),
]

MEDIUM_CALLS = [
    EmergencyCall(
        call_id="M001",
        incident_type="accident",
        description="Two-car collision on highway. One driver appears injured and trapped. Minor fire from engine.",
        location="Highway 7, mile marker 42",
        caller_info="Passing motorist",
        severity="severe",
        correct_dispatch={"ambulance": 1, "police": 1, "fire": 1},
        correct_priority=4,
        needs_backup=False,
    ),
    EmergencyCall(
        call_id="M002",
        incident_type="medical",
        description="Man behaving erratically on street, threatening passers-by, appears to be in medical distress (possibly diabetic shock or mental health crisis).",
        location="City Park near fountain",
        caller_info="Park visitor",
        severity="moderate",
        correct_dispatch={"ambulance": 1, "police": 2, "fire": 0},
        correct_priority=3,
        needs_backup=False,
    ),
    EmergencyCall(
        call_id="M003",
        incident_type="fire",
        description="Chemical smell reported from industrial warehouse. Workers evacuating. Unknown substance.",
        location="Westport Industrial Zone, Building 7",
        caller_info="Warehouse foreman",
        severity="severe",
        correct_dispatch={"ambulance": 1, "police": 0, "fire": 3},
        correct_priority=4,
        needs_backup=True,
    ),
    EmergencyCall(
        call_id="M004",
        incident_type="crime",
        description="Bank alarm triggered. Cameras show 3 armed individuals inside. Customers may be hostages.",
        location="First National Bank, Commerce Street",
        caller_info="Silent alarm, no caller",
        severity="critical",
        correct_dispatch={"ambulance": 1, "police": 5, "fire": 0},
        correct_priority=4,
        needs_backup=True,
    ),
    EmergencyCall(
        call_id="M005",
        incident_type="accident",
        description="Motorcycle vs truck. Rider down, not moving. Truck driver uninjured but in shock.",
        location="Intersection of Route 9 and Park Lane",
        caller_info="Truck driver, distressed",
        severity="severe",
        correct_dispatch={"ambulance": 2, "police": 1, "fire": 0},
        correct_priority=4,
        needs_backup=False,
    ),
    EmergencyCall(
        call_id="M006",
        incident_type="medical",
        description="Multiple people feeling dizzy and nauseous in an office building. Possible carbon monoxide leak.",
        location="Meridian Tower, 14th Floor",
        caller_info="Office worker, coughing",
        severity="severe",
        correct_dispatch={"ambulance": 2, "police": 0, "fire": 2},
        correct_priority=4,
        needs_backup=False,
    ),
]

HARD_CALLS = [
    EmergencyCall(
        call_id="H001",
        incident_type="multi-hazard",
        description="Gas explosion at apartment complex. Multiple casualties reported. Structural damage. Fire spreading. Residents trapped on upper floors. Secondary explosion risk.",
        location="Riverfront Apartments, 100 Waterside Drive",
        caller_info="Multiple callers simultaneously",
        severity="critical",
        correct_dispatch={"ambulance": 3, "police": 2, "fire": 3},
        correct_priority=4,
        needs_backup=True,
    ),
    EmergencyCall(
        call_id="H002",
        incident_type="multi-hazard",
        description="Active shooter reported at shopping mall. Multiple shots fired, unknown number of casualties. Suspect still at large inside. Crowds panicking and fleeing.",
        location="Westfield Mall, Central Atrium",
        caller_info="Multiple terrified callers",
        severity="critical",
        correct_dispatch={"ambulance": 2, "police": 5, "fire": 0},
        correct_priority=4,
        needs_backup=True,
    ),
    EmergencyCall(
        call_id="H003",
        incident_type="multi-hazard",
        description="Train derailment near residential area. Multiple cars off track. Hazmat car involved — unknown chemical leak. Injuries reported. Road access blocked.",
        location="Rail Yard, Southern District",
        caller_info="Train operator via radio",
        severity="critical",
        correct_dispatch={"ambulance": 3, "police": 2, "fire": 3},
        correct_priority=4,
        needs_backup=True,
    ),
    EmergencyCall(
        call_id="H004",
        incident_type="multi-hazard",
        description="Large-scale bar fight spilled into street. Multiple injuries including stab wound victim. Fire from knocked-over heater in bar. Crowds blocking access.",
        location="The Iron Horse Bar, Downtown",
        caller_info="Bar staff, chaotic background noise",
        severity="critical",
        correct_dispatch={"ambulance": 2, "police": 3, "fire": 1},
        correct_priority=4,
        needs_backup=True,
    ),
]

# Crisis calls - both hospitals full, reserve=0, cascade spawns
CRISIS_CALLS = [
    EmergencyCall(
        call_id="C001",
        incident_type="multi-hazard",
        description="Building collapse during construction. Multiple workers trapped. Dust cloud visible. Secondary collapse risk.",
        location="Downtown Construction Site, Block 7",
        caller_info="Site foreman, panicked",
        severity="critical",
        correct_dispatch={"ambulance": 3, "police": 2, "fire": 3},
        correct_priority=4,
        needs_backup=True,
        coordination_tier="mci_protocol",
        correct_hospital="regional",
        correct_staging="stage_nearby",
    ),
    EmergencyCall(
        call_id="C002",
        incident_type="medical",
        description="Pediatric cardiac arrest. School nurse performing CPR. Needs immediate advanced life support.",
        location="Lincoln Middle School, Gymnasium",
        caller_info="School principal, distressed",
        severity="critical",
        correct_dispatch={"ambulance": 2, "police": 0, "fire": 1},
        correct_priority=4,
        needs_backup=False,
        coordination_tier="none",
        correct_hospital="nearest",
        correct_staging="dispatch",
    ),
    EmergencyCall(
        call_id="C003",
        incident_type="multi-hazard",
        description="Highway pileup - 12 vehicles in chain reaction. Fog conditions. Multiple critical injuries. Road blocked.",
        location="Interstate 95, Mile Marker 73",
        caller_info="Multiple callers via 911",
        severity="critical",
        correct_dispatch={"ambulance": 4, "police": 3, "fire": 2},
        correct_priority=4,
        needs_backup=True,
        coordination_tier="mci_protocol",
        correct_hospital="regional",
        correct_staging="on_scene_hold",
    ),
    EmergencyCall(
        call_id="C004",
        incident_type="fire",
        description="Warehouse fire spreading to adjacent buildings. Gas lines threatened. Full evacuation ordered.",
        location="Industrial District, Warehouses 12-16",
        caller_info="Fire alarm system + security guard",
        severity="critical",
        correct_dispatch={"ambulance": 2, "police": 2, "fire": 5},
        correct_priority=4,
        needs_backup=True,
        coordination_tier="mutual_aid",
        correct_hospital="regional",
        correct_staging="stage_nearby",
    ),
]

TASK_CALL_MAP = {
    "easy": EASY_CALLS,
    "medium": MEDIUM_CALLS,
    "hard": HARD_CALLS,
    "crisis": CRISIS_CALLS,
}

# ---------------------------------------------------------------------------
# Hospital Configurations by Task Difficulty
# ---------------------------------------------------------------------------

HOSPITAL_CONFIGS = {
    "easy": {
        "hospital_a": {
            "name": "City General Hospital",
            "icu_beds": 8,
            "general_beds": 25,
            "trauma_capable": True,
            "distance_minutes": 5,
        },
        "hospital_b": {
            "name": "Riverside Medical Center",
            "icu_beds": 6,
            "general_beds": 20,
            "trauma_capable": False,
            "distance_minutes": 12,
        },
        "district_reserve_units": 6,
        "replenishment_rate": 2,
    },
    "medium": {
        "hospital_a": {
            "name": "City General Hospital",
            "icu_beds": 2,
            "general_beds": 8,
            "trauma_capable": True,
            "distance_minutes": 5,
        },
        "hospital_b": {
            "name": "Riverside Medical Center",
            "icu_beds": 5,
            "general_beds": 18,
            "trauma_capable": True,
            "distance_minutes": 15,
        },
        "district_reserve_units": 3,
        "replenishment_rate": 3,
    },
    "hard": {
        "hospital_a": {
            "name": "City General Hospital",
            "icu_beds": 1,
            "general_beds": 3,
            "trauma_capable": True,
            "distance_minutes": 5,
        },
        "hospital_b": {
            "name": "Riverside Medical Center",
            "icu_beds": 0,
            "general_beds": 6,
            "trauma_capable": False,
            "distance_minutes": 20,
        },
        "district_reserve_units": 2,
        "replenishment_rate": 4,
    },
    "crisis": {
        "hospital_a": {
            "name": "City General Hospital",
            "icu_beds": 0,
            "general_beds": 1,
            "trauma_capable": True,
            "distance_minutes": 5,
        },
        "hospital_b": {
            "name": "Riverside Medical Center",
            "icu_beds": 0,
            "general_beds": 0,
            "trauma_capable": True,
            "distance_minutes": 25,
        },
        "district_reserve_units": 0,
        "replenishment_rate": 5,
    },
}


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(action: DispatchGridAction, call: EmergencyCall) -> tuple[float, str]:
    """
    Compute reward for a dispatch action against the correct answer.

    Reward breakdown (max 0.90 per call):
      Per unit type (ambulance / police / fire) × 3:
        +0.10 correct type dispatched with sufficient units
        +0.05 correct type but too few units
         0.00 unit not needed and not sent (correct omission)
        -0.10 wrong type sent when not needed
        -0.10 needed type not sent at all

      Priority level: max 0.20
        +0.20 exact match
        +0.10 off by 1
        -0.10 off by 2+

      Coordination decision: max 0.15
        +0.15 correct coordination level
        +0.10 partial credit (mutual_aid when mci needed)
        -0.05 to -0.15 for wrong decisions

      Hospital routing: max 0.15
        +0.15 correct hospital choice
        +0.08 for "auto" fallback
        -0.05 to -0.10 for wrong choice

      Staging appropriateness: max 0.10
        +0.10 correct staging decision
        +0.05 partial credit
        -0.05 for wrong staging

    Max possible reward per call: 0.30 + 0.20 + 0.15 + 0.15 + 0.10 = 0.90
    """
    reward = 0.0
    feedback_parts = []

    correct = call.correct_dispatch  # {"ambulance": N, "police": N, "fire": N}
    sent = {
        "ambulance": action.ambulance_units,
        "police": action.police_units,
        "fire": action.fire_units,
    }

    # --- Per unit type scoring (×3 = max 0.30) ---
    for unit in ("ambulance", "police", "fire"):
        needed = correct[unit]
        dispatched = sent[unit]

        if needed == 0 and dispatched == 0:
            pass  # Correct omission
        elif needed == 0 and dispatched > 0:
            reward -= 0.10
            feedback_parts.append(f"❌ {unit}: not needed, but {dispatched} sent (-0.10)")
        elif needed > 0 and dispatched == 0:
            reward -= 0.10
            feedback_parts.append(f"❌ {unit}: {needed} needed, none sent (-0.10)")
        elif dispatched == needed:
            reward += 0.10
            feedback_parts.append(f"✅ {unit}: {dispatched} sent (min {needed}) (+0.10)")
        elif dispatched > needed:
            reward += 0.075
            feedback_parts.append(f"⚠️ {unit}: {dispatched} sent but {needed} needed (+0.075)")
        else:
            reward += 0.05
            feedback_parts.append(f"⚠️ {unit}: {dispatched} sent but {needed} needed (+0.05)")

    # --- Priority scoring (max 0.20) ---
    diff = abs(action.priority_level - call.correct_priority)
    if diff == 0:
        reward += 0.20
        feedback_parts.append(f"✅ priority {action.priority_level} correct (+0.20)")
    elif diff == 1:
        reward += 0.10
        feedback_parts.append(f"⚠️ priority {action.priority_level} (expected {call.correct_priority}) (+0.10)")
    else:
        reward -= 0.10
        feedback_parts.append(f"❌ priority {action.priority_level} (expected {call.correct_priority}) (-0.10)")

    # --- Coordination scoring (max 0.15) ---
    correct_coord = getattr(call, "coordination_tier", "none")
    chosen_coord = action.coordination_level

    if chosen_coord == correct_coord:
        reward += 0.15
        feedback_parts.append(f"✅ coordination '{chosen_coord}' correct (+0.15)")
    elif correct_coord == "mci_protocol" and chosen_coord == "mutual_aid":
        reward += 0.10  # Partial credit - at least escalated
        feedback_parts.append(f"⚠️ coordination 'mutual_aid' but 'mci_protocol' needed (+0.10)")
    elif correct_coord == "mutual_aid" and chosen_coord == "none":
        reward -= 0.08
        feedback_parts.append(f"❌ coordination 'none' but 'mutual_aid' needed (-0.08)")
    elif correct_coord == "mci_protocol" and chosen_coord == "none":
        reward -= 0.15
        feedback_parts.append(f"❌ coordination 'none' but 'mci_protocol' needed (-0.15)")
    elif chosen_coord == "mci_protocol" and correct_coord == "none":
        reward -= 0.10  # Over-escalation wastes reserves
        feedback_parts.append(f"❌ coordination 'mci_protocol' when not needed (-0.10)")
    else:
        reward -= 0.05
        feedback_parts.append(f"⚠️ coordination '{chosen_coord}' incorrect (-0.05)")

    # --- Hospital routing scoring (max 0.15) ---
    correct_hospital = getattr(call, "correct_hospital", "nearest")
    chosen_hospital = action.hospital_choice

    if chosen_hospital == correct_hospital:
        reward += 0.15
        feedback_parts.append(f"✅ hospital '{chosen_hospital}' correct (+0.15)")
    elif chosen_hospital == "auto":
        reward += 0.08  # Safe fallback, never optimal
        feedback_parts.append(f"⚠️ hospital 'auto' fallback (+0.08)")
    elif correct_hospital == "regional" and chosen_hospital == "nearest":
        reward -= 0.10  # Sent critical patient to wrong hospital
        feedback_parts.append(f"❌ hospital 'nearest' but 'regional' needed (-0.10)")
    elif correct_hospital == "nearest" and chosen_hospital == "regional":
        reward -= 0.05  # Unnecessary delay
        feedback_parts.append(f"⚠️ hospital 'regional' but 'nearest' correct (-0.05)")
    else:
        reward -= 0.05
        feedback_parts.append(f"⚠️ hospital '{chosen_hospital}' incorrect (-0.05)")

    # --- Staging scoring (max 0.10) ---
    correct_staging = getattr(call, "correct_staging", "dispatch")
    chosen_staging = action.ambulance_staging

    if chosen_staging == correct_staging:
        reward += 0.10
        feedback_parts.append(f"✅ staging '{chosen_staging}' correct (+0.10)")
    elif correct_staging == "stage_nearby" and chosen_staging == "dispatch":
        reward -= 0.05  # Sent too early but not catastrophic
        feedback_parts.append(f"⚠️ staging 'dispatch' but 'stage_nearby' optimal (-0.05)")
    elif correct_staging == "on_scene_hold" and chosen_staging == "dispatch":
        reward -= 0.05
        feedback_parts.append(f"⚠️ staging 'dispatch' but 'on_scene_hold' optimal (-0.05)")
    elif chosen_staging != correct_staging:
        reward -= 0.05
        feedback_parts.append(f"⚠️ staging '{chosen_staging}' suboptimal (-0.05)")

    reward = round(max(-1.0, min(1.0, reward)), 4)
    return reward, " | ".join(feedback_parts)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DispatchGridEnvironment(Environment):
    """
    Emergency Response Dispatch System.

    The agent dispatches ambulance / police / fire units per call.
    Resources are deducted after each dispatch.
    Episode ends when all calls in the queue are handled.

    Features:
    - Hospital bed tracking with capacity-aware routing
    - 3-tier coordination system (none/mutual_aid/mci_protocol)
    - Resource replenishment over time
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "easy"):
        if task not in TASK_CALL_MAP:
            raise ValueError(f"Unknown task '{task}'. Choose: easy, medium, hard, crisis")
        self._task = task
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._calls: list = []
        self._call_index: int = 0
        self._cumulative_score: float = 0.0
        self._total_response_time: float = 0.0

        # Base resources
        self._ambulances = 5
        self._police = 8
        self._fire = 4

        # Hospital state (per task difficulty)
        hospital_config = HOSPITAL_CONFIGS.get(task, HOSPITAL_CONFIGS["easy"])
        self._hospital_a = hospital_config["hospital_a"].copy()
        self._hospital_b = hospital_config["hospital_b"].copy()
        self._district_reserve_units = hospital_config["district_reserve_units"]
        self._replenishment_rate = hospital_config["replenishment_rate"]

        # Resource replenishment tracking
        self._ambulances_returning_in = 0
        self._police_returning_in = 0
        self._fire_returning_in = 0
        self._mci_protocol_active = False

    def reset(self) -> DispatchGridObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cumulative_score = 0.0
        self._total_response_time = 0.0
        self._ambulances = 5
        self._police = 8
        self._fire = 4

        # Reset hospital state
        hospital_config = HOSPITAL_CONFIGS.get(self._task, HOSPITAL_CONFIGS["easy"])
        self._hospital_a = hospital_config["hospital_a"].copy()
        self._hospital_b = hospital_config["hospital_b"].copy()
        self._district_reserve_units = hospital_config["district_reserve_units"]
        self._replenishment_rate = hospital_config["replenishment_rate"]

        # Reset replenishment tracking
        self._ambulances_returning_in = 0
        self._police_returning_in = 0
        self._fire_returning_in = 0
        self._mci_protocol_active = False
        self._units_pending_return = []

        pool = TASK_CALL_MAP[self._task].copy()
        random.shuffle(pool)
        self._calls = pool[:4]
        self._call_index = 0

        return self._make_observation(0.0, "Episode started. First emergency call incoming.")

    def step(self, action: DispatchGridAction) -> DispatchGridObservation:  # type: ignore[override]
        self._state.step_count += 1

        current_call = self._calls[self._call_index]
        reward, feedback = compute_reward(action, current_call)
        self._cumulative_score += reward

        # Process coordination level
        if action.coordination_level == "mci_protocol":
            self._mci_protocol_active = True
            if self._district_reserve_units > 0:
                self._district_reserve_units -= 1
        elif action.coordination_level == "mutual_aid":
            if self._district_reserve_units > 0:
                self._district_reserve_units -= 1

        # Simulate response time
        base_time = {4: 4.0, 3: 7.0, 2: 12.0, 1: 20.0}
        total_units = action.ambulance_units + action.police_units + action.fire_units
        response_time = max(2.0, base_time.get(action.priority_level, 10.0) - (total_units - 1) * 0.5)
        self._total_response_time += response_time

        # Deduct resources accurately per unit type
        self._ambulances = max(0, self._ambulances - action.ambulance_units)
        self._police = max(0, self._police - action.police_units)
        self._fire = max(0, self._fire - action.fire_units)

        # Track units for replenishment
        if action.ambulance_units > 0 or action.police_units > 0 or action.fire_units > 0:
            self._units_pending_return.append({
                "ambulances": action.ambulance_units,
                "police": action.police_units,
                "fire": action.fire_units,
                "return_in": self._replenishment_rate,
            })

        # Process replenishment (decrement counters and return units)
        for pending in self._units_pending_return:
            pending["return_in"] -= 1
        returned = [p for p in self._units_pending_return if p["return_in"] <= 0]
        for r in returned:
            self._ambulances += r["ambulances"]
            self._police += r["police"]
            self._fire += r["fire"]
        self._units_pending_return = [p for p in self._units_pending_return if p["return_in"] > 0]

        # Update returning_in counters
        if self._units_pending_return:
            self._ambulances_returning_in = min((p["return_in"] for p in self._units_pending_return if p["ambulances"] > 0), default=0)
            self._police_returning_in = min((p["return_in"] for p in self._units_pending_return if p["police"] > 0), default=0)
            self._fire_returning_in = min((p["return_in"] for p in self._units_pending_return if p["fire"] > 0), default=0)
        else:
            self._ambulances_returning_in = 0
            self._police_returning_in = 0
            self._fire_returning_in = 0

        self._call_index += 1
        done = self._call_index >= len(self._calls)

        if done:
            reward += 0.20
            self._cumulative_score += 0.20
            feedback += f" | Episode complete! Total score: {self._cumulative_score:.3f}"

        return self._make_observation(reward, feedback, done=done)

    def _make_observation(self, last_reward: float, last_feedback: str, done: bool = False) -> DispatchGridObservation:
        calls_handled = self._call_index
        total_calls = len(self._calls)
        calls_remaining = total_calls - calls_handled
        avg_rt = self._total_response_time / calls_handled if calls_handled > 0 else 0.0

        # Determine nearest and recommended hospital
        nearest = "A" if self._hospital_a["distance_minutes"] <= self._hospital_b["distance_minutes"] else "B"
        # Recommend regional (trauma-capable) if ICU needed or trauma, else nearest
        recommended = "A" if (self._hospital_a["trauma_capable"] and self._hospital_a["icu_beds"] > 0) else "B"
        if not recommended and self._hospital_b["trauma_capable"] and self._hospital_b["icu_beds"] > 0:
            recommended = "B"
        if recommended == nearest:
            pass  # Keep recommended
        elif self._hospital_a["trauma_capable"] and self._hospital_a["icu_beds"] > 0:
            recommended = "A"
        elif self._hospital_b["trauma_capable"] and self._hospital_b["icu_beds"] > 0:
            recommended = "B"
        else:
            recommended = nearest

        if done or calls_remaining == 0:
            return DispatchGridObservation(
                call_id="DONE",
                incident_type="",
                call_description="All calls handled. Episode complete.",
                location="", caller_info="", severity="",
                calls_handled=calls_handled, total_calls=total_calls, calls_remaining=0,
                cumulative_score=round(self._cumulative_score, 4),
                last_action_reward=round(last_reward, 4),
                last_action_feedback=last_feedback,
                available_ambulances=self._ambulances,
                available_police=self._police,
                available_fire=self._fire,
                avg_response_time_minutes=round(avg_rt, 2),
                # Hospital tracking
                hospital_a_name=self._hospital_a["name"],
                hospital_a_icu_beds=self._hospital_a["icu_beds"],
                hospital_a_general_beds=self._hospital_a["general_beds"],
                hospital_a_trauma_capable=self._hospital_a["trauma_capable"],
                hospital_a_distance_minutes=self._hospital_a["distance_minutes"],
                hospital_b_name=self._hospital_b["name"],
                hospital_b_icu_beds=self._hospital_b["icu_beds"],
                hospital_b_general_beds=self._hospital_b["general_beds"],
                hospital_b_trauma_capable=self._hospital_b["trauma_capable"],
                hospital_b_distance_minutes=self._hospital_b["distance_minutes"],
                nearest_hospital=nearest,
                recommended_hospital=recommended,
                # Coordination state
                district_reserve_units=self._district_reserve_units,
                mci_protocol_active=self._mci_protocol_active,
                coordination_cost_remaining=self._district_reserve_units,
                # Resource replenishment
                ambulances_returning_in=self._ambulances_returning_in,
                police_returning_in=self._police_returning_in,
                fire_returning_in=self._fire_returning_in,
                done=True, reward=round(last_reward, 4),
            )

        current_call = self._calls[self._call_index]
        return DispatchGridObservation(
            call_id=current_call.call_id,
            incident_type=current_call.incident_type,
            call_description=current_call.description,
            location=current_call.location,
            caller_info=current_call.caller_info,
            severity=current_call.severity,
            calls_handled=calls_handled, total_calls=total_calls, calls_remaining=calls_remaining,
            cumulative_score=round(self._cumulative_score, 4),
            last_action_reward=round(last_reward, 4),
            last_action_feedback=last_feedback,
            available_ambulances=self._ambulances,
            available_police=self._police,
            available_fire=self._fire,
            avg_response_time_minutes=round(avg_rt, 2),
            # Hospital tracking
            hospital_a_name=self._hospital_a["name"],
            hospital_a_icu_beds=self._hospital_a["icu_beds"],
            hospital_a_general_beds=self._hospital_a["general_beds"],
            hospital_a_trauma_capable=self._hospital_a["trauma_capable"],
            hospital_a_distance_minutes=self._hospital_a["distance_minutes"],
            hospital_b_name=self._hospital_b["name"],
            hospital_b_icu_beds=self._hospital_b["icu_beds"],
            hospital_b_general_beds=self._hospital_b["general_beds"],
            hospital_b_trauma_capable=self._hospital_b["trauma_capable"],
            hospital_b_distance_minutes=self._hospital_b["distance_minutes"],
            nearest_hospital=nearest,
            recommended_hospital=recommended,
            # Coordination state
            district_reserve_units=self._district_reserve_units,
            mci_protocol_active=self._mci_protocol_active,
            coordination_cost_remaining=self._district_reserve_units,
            # Resource replenishment
            ambulances_returning_in=self._ambulances_returning_in,
            police_returning_in=self._police_returning_in,
            fire_returning_in=self._fire_returning_in,
            done=False, reward=round(last_reward, 4),
        )

    @property
    def state(self) -> State:
        return self._state