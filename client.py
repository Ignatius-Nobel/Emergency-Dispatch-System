# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Emergency Response Dispatch System — Environment Client."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
try:
    from .models import DispatchGridAction, DispatchGridObservation
except ImportError:
    from models import DispatchGridAction, DispatchGridObservation


class DispatchGridEnv(EnvClient[DispatchGridAction, DispatchGridObservation, State]):
    """
    Client for the Emergency Response Dispatch System.

    Example:
        >>> with DispatchGridEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     obs = result.observation
        ...     print(obs.call_description)
        ...
        ...     action = DispatchGridAction(
        ...         ambulance_units=1,
        ...         police_units=0,
        ...         fire_units=0,
        ...         priority_level=3,
        ...         backup_requested=False,
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.last_action_feedback)
    """

    def _step_payload(self, action: DispatchGridAction) -> Dict:
        return {
            "ambulance_units": action.ambulance_units,
            "police_units": action.police_units,
            "fire_units": action.fire_units,
            "priority_level": action.priority_level,
            "backup_requested": action.backup_requested,
            "notes": action.notes,
            "hospital_choice": action.hospital_choice,
            "coordination_level": action.coordination_level,
            "ambulance_staging": action.ambulance_staging,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DispatchGridObservation]:
        obs_data = payload.get("observation", {})
        observation = DispatchGridObservation(
            call_id=obs_data.get("call_id", ""),
            incident_type=obs_data.get("incident_type", ""),
            call_description=obs_data.get("call_description", ""),
            location=obs_data.get("location", ""),
            caller_info=obs_data.get("caller_info", ""),
            severity=obs_data.get("severity", ""),
            calls_handled=obs_data.get("calls_handled", 0),
            total_calls=obs_data.get("total_calls", 0),
            calls_remaining=obs_data.get("calls_remaining", 0),
            cumulative_score=obs_data.get("cumulative_score", 0.0),
            last_action_reward=obs_data.get("last_action_reward", 0.0),
            last_action_feedback=obs_data.get("last_action_feedback", ""),
            available_ambulances=obs_data.get("available_ambulances", 5),
            available_police=obs_data.get("available_police", 8),
            available_fire=obs_data.get("available_fire", 4),
            avg_response_time_minutes=obs_data.get("avg_response_time_minutes", 0.0),
            # Hospital tracking
            hospital_a_name=obs_data.get("hospital_a_name", ""),
            hospital_a_icu_beds=obs_data.get("hospital_a_icu_beds", 0),
            hospital_a_general_beds=obs_data.get("hospital_a_general_beds", 0),
            hospital_a_trauma_capable=obs_data.get("hospital_a_trauma_capable", False),
            hospital_a_distance_minutes=obs_data.get("hospital_a_distance_minutes", 0),
            hospital_b_name=obs_data.get("hospital_b_name", ""),
            hospital_b_icu_beds=obs_data.get("hospital_b_icu_beds", 0),
            hospital_b_general_beds=obs_data.get("hospital_b_general_beds", 0),
            hospital_b_trauma_capable=obs_data.get("hospital_b_trauma_capable", False),
            hospital_b_distance_minutes=obs_data.get("hospital_b_distance_minutes", 0),
            nearest_hospital=obs_data.get("nearest_hospital", ""),
            recommended_hospital=obs_data.get("recommended_hospital", ""),
            # Coordination state
            district_reserve_units=obs_data.get("district_reserve_units", 6),
            mci_protocol_active=obs_data.get("mci_protocol_active", False),
            coordination_cost_remaining=obs_data.get("coordination_cost_remaining", 0),
            # Resource replenishment
            ambulances_returning_in=obs_data.get("ambulances_returning_in", 0),
            police_returning_in=obs_data.get("police_returning_in", 0),
            fire_returning_in=obs_data.get("fire_returning_in", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )