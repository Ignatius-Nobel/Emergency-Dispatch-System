# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Emergency Response Dispatch System.
"""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator


class DispatchGridAction(Action):
    """
    Action taken by the dispatch agent.

    Specify per-unit-type counts directly. Set to 0 to skip that type.
    At least one unit type must be >= 1.

    Valid combinations:
      ambulance only          → ambulance_units>=1, police_units=0, fire_units=0
      police only             → police_units>=1, others=0
      fire only               → fire_units>=1, others=0
      ambulance + police      → both>=1, fire_units=0
      ambulance + fire        → both>=1, police_units=0
      police + fire           → both>=1, ambulance_units=0
      ambulance + police + fire → all>=1
    """

    ambulance_units: int = Field(
        default=0, ge=0,
        description="Number of ambulance units (0 = not dispatched); validated in env (0-10).",
    )
    police_units: int = Field(
        default=0, ge=0,
        description="Number of police units (0 = not dispatched); validated in env (0-10).",
    )
    fire_units: int = Field(
        default=0, ge=0,
        description="Number of fire units (0 = not dispatched); validated in env (0-10).",
    )
    priority_level: int = Field(
        ...,
        description="Priority level: 1=low, 2=medium, 3=high, 4=critical; validated in env (1-4).",
    )
    backup_requested: bool = Field(
        default=False,
        description="Whether to request backup from neighboring districts (legacy field)",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional dispatcher notes",
    )
    # Enhanced action fields (FEATURE_UPGRADE.md)
    hospital_choice: str = Field(
        default="auto",
        description="Hospital selection: 'nearest' | 'regional' | 'auto'",
    )
    coordination_level: str = Field(
        default="none",
        description="Coordination tier: 'none' | 'mutual_aid' | 'mci_protocol'",
    )
    ambulance_staging: str = Field(
        default="dispatch",
        description="Ambulance staging: 'dispatch' | 'stage_nearby' | 'on_scene_hold'",
    )

    @model_validator(mode="after")
    def at_least_one_unit(self):
        if self.ambulance_units == 0 and self.police_units == 0 and self.fire_units == 0:
            raise ValueError("At least one unit type must be >= 1")
        return self

    @property
    def dispatch_summary(self) -> str:
        """Human-readable summary of what was dispatched."""
        parts = []
        if self.ambulance_units > 0:
            parts.append(f"{self.ambulance_units} ambulance")
        if self.police_units > 0:
            parts.append(f"{self.police_units} police")
        if self.fire_units > 0:
            parts.append(f"{self.fire_units} fire")
        return " + ".join(parts)


class EmergencyCall(object):
    """Represents an incoming emergency call (used internally)."""

    def __init__(
        self,
        call_id: str,
        incident_type: str,
        description: str,
        location: str,
        caller_info: str,
        severity: str,
        correct_dispatch: dict,   # e.g. {"ambulance": 1, "police": 2, "fire": 0}
        correct_priority: int,
        needs_backup: bool,
        coordination_tier: str = "none",  # "none" | "mutual_aid" | "mci_protocol"
        correct_hospital: str = "nearest",  # "nearest" | "regional"
        correct_staging: str = "dispatch",  # "dispatch" | "stage_nearby" | "on_scene_hold"
    ):
        self.call_id = call_id
        self.incident_type = incident_type
        self.description = description
        self.location = location
        self.caller_info = caller_info
        self.severity = severity
        self.correct_dispatch = correct_dispatch  # dict with ambulance/police/fire min units
        self.correct_priority = correct_priority
        self.needs_backup = needs_backup
        self.coordination_tier = coordination_tier
        self.correct_hospital = correct_hospital
        self.correct_staging = correct_staging


class DispatchGridObservation(Observation):
    """Observation returned to the agent after each step."""

    # Current emergency call info
    call_id: str = Field(default="", description="Unique ID of the current emergency call")
    incident_type: str = Field(default="", description="Type of incident")
    call_description: str = Field(default="", description="Full description of the emergency call")
    location: str = Field(default="", description="Location of the emergency")
    caller_info: str = Field(default="", description="Information about the caller")
    severity: str = Field(default="", description="Observed severity: minor/moderate/severe/critical")

    # Episode progress
    calls_handled: int = Field(default=0, description="Number of calls handled so far")
    total_calls: int = Field(default=0, description="Total calls in this episode")
    calls_remaining: int = Field(default=0, description="Calls remaining in episode")

    # Performance tracking
    cumulative_score: float = Field(default=0.0, description="Cumulative score so far")
    last_action_reward: float = Field(default=0.0, description="Reward from last action")
    last_action_feedback: str = Field(default="", description="Feedback on last action")

    # Resources (updated after each dispatch)
    available_ambulances: int = Field(default=5, description="Available ambulance units")
    available_police: int = Field(default=8, description="Available police units")
    available_fire: int = Field(default=4, description="Available fire units")
    avg_response_time_minutes: float = Field(default=0.0, description="Average response time so far")

    # Hospital tracking (FEATURE_UPGRADE.md)
    hospital_a_name: str = Field(default="", description="Hospital A name")
    hospital_a_icu_beds: int = Field(default=0, description="Hospital A ICU beds available")
    hospital_a_general_beds: int = Field(default=0, description="Hospital A general beds available")
    hospital_a_trauma_capable: bool = Field(default=False, description="Hospital A trauma capability")
    hospital_a_distance_minutes: int = Field(default=0, description="Hospital A distance in minutes")

    hospital_b_name: str = Field(default="", description="Hospital B name")
    hospital_b_icu_beds: int = Field(default=0, description="Hospital B ICU beds available")
    hospital_b_general_beds: int = Field(default=0, description="Hospital B general beds available")
    hospital_b_trauma_capable: bool = Field(default=False, description="Hospital B trauma capability")
    hospital_b_distance_minutes: int = Field(default=0, description="Hospital B distance in minutes")

    nearest_hospital: str = Field(default="", description="Nearest hospital: 'A' or 'B'")
    recommended_hospital: str = Field(default="", description="System recommended hospital")

    # Coordination state (FEATURE_UPGRADE.md)
    district_reserve_units: int = Field(default=6, description="Shared mutual-aid pool units")
    mci_protocol_active: bool = Field(default=False, description="MCI protocol active flag")
    coordination_cost_remaining: int = Field(default=0, description="Remaining coordination requests viable")

    # Resource replenishment (FEATURE_UPGRADE.md)
    ambulances_returning_in: int = Field(default=0, description="Calls until ambulances return")
    police_returning_in: int = Field(default=0, description="Calls until police return")
    fire_returning_in: int = Field(default=0, description="Calls until fire units return")