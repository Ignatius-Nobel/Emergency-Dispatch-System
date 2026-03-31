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
        default=0, ge=0, le=5,
        description="Number of ambulance units (0 = not dispatched)",
    )
    police_units: int = Field(
        default=0, ge=0, le=5,
        description="Number of police units (0 = not dispatched)",
    )
    fire_units: int = Field(
        default=0, ge=0, le=5,
        description="Number of fire units (0 = not dispatched)",
    )
    priority_level: int = Field(
        ..., ge=1, le=4,
        description="Priority level: 1=low, 2=medium, 3=high, 4=critical",
    )
    backup_requested: bool = Field(
        default=False,
        description="Whether to request backup from neighboring districts",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional dispatcher notes",
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