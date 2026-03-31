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

TASK_CALL_MAP = {
    "easy": EASY_CALLS,
    "medium": MEDIUM_CALLS,
    "hard": HARD_CALLS,
}

# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(action: DispatchGridAction, call: EmergencyCall) -> tuple[float, str]:
    """
    Compute reward for a dispatch action against the correct answer.

    Reward breakdown:
      Per unit type (ambulance / police / fire):
        +0.10 correct type dispatched with sufficient units
        +0.05 correct type but too few units
         0.00 unit not needed and not sent (correct omission)
        -0.10 wrong type sent when not needed
        -0.10 needed type not sent at all

      Priority:
        +0.20 exact match
        +0.10 off by 1
        -0.10 off by 2+

      Backup:
        +0.10 correct decision
        -0.05 wrong decision

    Max possible reward per call: 0.10*3 + 0.20 + 0.10 = 0.60
    """
    reward = 0.0
    feedback_parts = []

    correct = call.correct_dispatch  # {"ambulance": N, "police": N, "fire": N}
    sent = {
        "ambulance": action.ambulance_units,
        "police": action.police_units,
        "fire": action.fire_units,
    }

    # --- Per unit type scoring ---
    for unit in ("ambulance", "police", "fire"):
        needed = correct[unit]
        dispatched = sent[unit]

        if needed == 0 and dispatched == 0:
            # Correct omission — no reward, no penalty, just silent pass
            pass
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

    # --- Priority scoring ---
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

    # --- Backup scoring ---
    if action.backup_requested == call.needs_backup:
        reward += 0.10
        status = "needed" if call.needs_backup else "not needed"
        feedback_parts.append(f"✅ backup {status} (+0.10)")
    else:
        reward -= 0.05
        if call.needs_backup:
            feedback_parts.append("❌ backup needed but not requested (-0.05)")
        else:
            feedback_parts.append("⚠️ backup not needed but requested (-0.05)")

    reward = round(max(-1.0, min(1.0, reward)), 4)
    return reward, " | ".join(feedback_parts)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DispatchGridEnvironment(Environment):
    """
    Emergency Response Dispatch System.

    The agent dispatches ambulance / police / fire units per call.
    Resources are deducted accurately after each dispatch.
    Episode ends when all calls in the queue are handled.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "easy"):
        if task not in TASK_CALL_MAP:
            raise ValueError(f"Unknown task '{task}'. Choose: easy, medium, hard")
        self._task = task
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._calls: list = []
        self._call_index: int = 0
        self._cumulative_score: float = 0.0
        self._total_response_time: float = 0.0
        self._ambulances = 5
        self._police = 8
        self._fire = 4

    def reset(self) -> DispatchGridObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cumulative_score = 0.0
        self._total_response_time = 0.0
        self._ambulances = 5
        self._police = 8
        self._fire = 4

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

        # Simulate response time
        base_time = {4: 4.0, 3: 7.0, 2: 12.0, 1: 20.0}
        total_units = action.ambulance_units + action.police_units + action.fire_units
        response_time = max(2.0, base_time.get(action.priority_level, 10.0) - (total_units - 1) * 0.5)
        self._total_response_time += response_time

        # Deduct resources accurately per unit type
        self._ambulances = max(0, self._ambulances - action.ambulance_units)
        self._police = max(0, self._police - action.police_units)
        self._fire = max(0, self._fire - action.fire_units)

        self._call_index += 1
        done = self._call_index >= len(self._calls)

        if done:
            reward += 0.20
            self._cumulative_score += 0.20
            feedback += f" | 🏁 Episode complete! Total score: {self._cumulative_score:.3f}"

        return self._make_observation(reward, feedback, done=done)

    def _make_observation(self, last_reward: float, last_feedback: str, done: bool = False) -> DispatchGridObservation:
        calls_handled = self._call_index
        total_calls = len(self._calls)
        calls_remaining = total_calls - calls_handled
        avg_rt = self._total_response_time / calls_handled if calls_handled > 0 else 0.0

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
            done=False, reward=round(last_reward, 4),
        )

    @property
    def state(self) -> State:
        return self._state