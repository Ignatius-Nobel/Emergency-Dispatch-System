"""Grader for Task 3: Multi-Hazard Cascading Emergencies (hard).

Scoring
-------
Uses the environment's compute_reward function and normalizes to 0.0-1.0 range.

1.0 — Optimal dispatch (all components correct)
0.5 — Partial credit (some components correct)
0.0 — Poor dispatch (most components wrong)

The grader evaluates:
- Unit type accuracy (ambulance/police/fire)
- Priority level match
- Coordination decision
- Hospital routing
- Staging appropriateness
"""

from __future__ import annotations
from typing import Any

try:
    from models import DispatchGridAction, EmergencyCall
    from server.dispatch_grid_environment import compute_reward, HARD_CALLS
except ImportError:
    try:
        from dispatch_grid.models import DispatchGridAction, EmergencyCall
        from dispatch_grid.server.dispatch_grid_environment import compute_reward, HARD_CALLS
    except ImportError:
        from ..models import DispatchGridAction, EmergencyCall
        from .dispatch_grid_environment import compute_reward, HARD_CALLS


class HardGrader:
    """Grade Task 3: Multi-Hazard Cascading Emergencies."""

    def __init__(self) -> None:
        self.last_breakdown: dict[str, Any] = {}
        self._calls_by_id = {c.call_id: c for c in HARD_CALLS}

    def grade(self, action: DispatchGridAction, ground_truth: dict[str, Any]) -> float:
        """Grade an action against ground truth.

        Args:
            action: The agent's dispatch action
            ground_truth: Dict with 'call_id' key identifying the call

        Returns:
            Normalized score in range 0.0-1.0
        """
        call_id = ground_truth.get("call_id", "")
        call = self._calls_by_id.get(call_id)

        if call is None:
            self.last_breakdown = {
                "error": f"Call ID '{call_id}' not found",
                "valid_ids": list(self._calls_by_id.keys()),
            }
            return 0.0

        raw_reward, feedback = compute_reward(action, call)

        # Normalize: raw_reward ranges from approx -0.45 to 0.60 per call
        # Normalize to 0.0-1.0
        max_r, min_r = 0.60, -0.45
        normalized = (raw_reward - min_r) / (max_r - min_r)
        normalized = max(0.0, min(1.0, normalized))

        self.last_breakdown = {
            "call_id": call_id,
            "incident_type": call.incident_type,
            "raw_reward": round(raw_reward, 4),
            "normalized_score": round(normalized, 4),
            "feedback": feedback,
            "ground_truth": {
                "correct_dispatch": call.correct_dispatch,
                "correct_priority": call.correct_priority,
                "needs_backup": call.needs_backup,
            },
            "agent_action": {
                "ambulance_units": action.ambulance_units,
                "police_units": action.police_units,
                "fire_units": action.fire_units,
                "priority_level": action.priority_level,
                "backup_requested": action.backup_requested,
                "hospital_choice": action.hospital_choice,
                "coordination_level": action.coordination_level,
                "ambulance_staging": action.ambulance_staging,
            },
        }

        return round(normalized, 4)
