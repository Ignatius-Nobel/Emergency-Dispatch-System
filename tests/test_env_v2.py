"""MVE lean dynamics: traffic, ETA, beds, diversion, outcome reward (v2)."""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "server")

from dispatch_grid_environment import (  # noqa: E402
    DispatchGridEnvironment,
    HOSPITAL_CONFIGS,
    compute_reward,
)
from models import DispatchGridAction, EmergencyCall  # noqa: E402


def _synthetic_critical_medical_zone_c() -> EmergencyCall:
    return EmergencyCall(
        call_id="SYN_ICU",
        incident_type="medical",
        description="Critical trauma patient",
        location="Zone C",
        caller_info="EMS",
        severity="critical",
        correct_dispatch={"ambulance": 1, "police": 0, "fire": 0},
        correct_priority=4,
        needs_backup=False,
        coordination_tier="none",
        correct_hospital="nearest",
        correct_staging="dispatch",
        zone="C",
        patients=1,
    )


def _optimal_rubric_action(call: EmergencyCall) -> DispatchGridAction:
    d = call.correct_dispatch
    return DispatchGridAction(
        ambulance_units=int(d["ambulance"]),
        police_units=int(d["police"]),
        fire_units=int(d["fire"]),
        priority_level=int(call.correct_priority),
        coordination_level=call.coordination_tier,
        hospital_choice=call.correct_hospital,
        ambulance_staging=call.correct_staging,
    )


def test_rubric_reward_unchanged_vs_compute_reward():
    env = DispatchGridEnvironment(task="easy")
    env.reset(seed=123, reward_mode="rubric")
    call = env._calls[0]
    action = _optimal_rubric_action(call)
    r_expected, _ = compute_reward(action, call)
    obs = env.step(action)
    assert abs(float(obs.last_action_reward) - r_expected) < 1e-4


def test_same_seed_same_first_step_eta_and_traffic():
    env1 = DispatchGridEnvironment(task="hard")
    env1.reset(seed=42, reward_mode="outcome")
    c1 = env1._calls[0]
    a1 = _optimal_rubric_action(c1)
    o1 = env1.step(a1)

    env2 = DispatchGridEnvironment(task="hard")
    env2.reset(seed=42, reward_mode="outcome")
    c2 = env2._calls[0]
    a2 = _optimal_rubric_action(c2)
    o2 = env2.step(a2)

    assert c1.call_id == c2.call_id
    assert o1.last_eta_minutes == o2.last_eta_minutes
    assert o1.traffic_regime == o2.traffic_regime


def test_auto_avoids_diversion_when_nearest_hospital_has_no_icu():
    env = DispatchGridEnvironment(task="hard")
    env.reset(seed=0, reward_mode="outcome")
    env._calls = [_synthetic_critical_medical_zone_c()]
    env._call_index = 0
    env._hospital_a = HOSPITAL_CONFIGS["hard"]["hospital_a"].copy()
    env._hospital_b = HOSPITAL_CONFIGS["hard"]["hospital_b"].copy()
    env._hospital_a["icu_beds"] = 2
    env._hospital_b["icu_beds"] = 0

    obs0 = env._make_observation(0.0, "Episode started.")
    act_auto = DispatchGridAction(
        ambulance_units=1,
        police_units=0,
        fire_units=0,
        priority_level=4,
        hospital_choice="auto",
        coordination_level="none",
        ambulance_staging="dispatch",
    )
    oa = env.step(act_auto)
    assert oa.last_diverted is False
    assert env._hospital_a["icu_beds"] == 1

    env2 = DispatchGridEnvironment(task="hard")
    env2.reset(seed=0, reward_mode="outcome")
    env2._calls = [_synthetic_critical_medical_zone_c()]
    env2._call_index = 0
    env2._hospital_a = HOSPITAL_CONFIGS["hard"]["hospital_a"].copy()
    env2._hospital_b = HOSPITAL_CONFIGS["hard"]["hospital_b"].copy()
    env2._hospital_a["icu_beds"] = 2
    env2._hospital_b["icu_beds"] = 0
    env2._make_observation(0.0, "Episode started.")
    act_near = DispatchGridAction(
        ambulance_units=1,
        police_units=0,
        fire_units=0,
        priority_level=4,
        hospital_choice="nearest",
        coordination_level="none",
        ambulance_staging="dispatch",
    )
    on = env2.step(act_near)
    assert on.last_diverted is True


def test_outcome_auto_higher_reward_than_nearest_on_icu_mismatch():
    def run_once(hospital_choice: str) -> float:
        env = DispatchGridEnvironment(task="hard")
        env.reset(seed=0, reward_mode="outcome")
        env._calls = [_synthetic_critical_medical_zone_c()]
        env._call_index = 0
        env._hospital_a = HOSPITAL_CONFIGS["hard"]["hospital_a"].copy()
        env._hospital_b = HOSPITAL_CONFIGS["hard"]["hospital_b"].copy()
        env._hospital_a["icu_beds"] = 2
        env._hospital_b["icu_beds"] = 0
        env._make_observation(0.0, "Episode started.")
        act = DispatchGridAction(
            ambulance_units=1,
            police_units=0,
            fire_units=0,
            priority_level=4,
            hospital_choice=hospital_choice,
            coordination_level="none",
            ambulance_staging="dispatch",
        )
        out = env.step(act)
        return float(out.last_action_reward)

    r_auto = run_once("auto")
    r_near = run_once("nearest")
    assert r_auto > r_near


def test_medical_moderate_consumes_general_not_icu():
    env = DispatchGridEnvironment(task="easy")
    env.reset(seed=0, reward_mode="outcome")
    mod = EmergencyCall(
        call_id="SYN_GEN",
        incident_type="medical",
        description="Minor injury",
        location="Zone A",
        caller_info="x",
        severity="moderate",
        correct_dispatch={"ambulance": 1, "police": 0, "fire": 0},
        correct_priority=2,
        needs_backup=False,
        zone="A",
        patients=1,
    )
    env._calls = [mod]
    env._call_index = 0
    ia0 = env._hospital_a["icu_beds"]
    ga0 = env._hospital_a["general_beds"]
    act = DispatchGridAction(
        ambulance_units=1,
        police_units=0,
        fire_units=0,
        priority_level=2,
        hospital_choice="nearest",
        coordination_level="none",
        ambulance_staging="dispatch",
    )
    env.step(act)
    assert env._hospital_a["icu_beds"] == ia0
    assert env._hospital_a["general_beds"] == ga0 - 1


def test_fire_call_does_not_consume_hospital_beds():
    env = DispatchGridEnvironment(task="easy")
    env.reset(seed=0, reward_mode="outcome")
    fire_call = next(c for c in env._calls if c.incident_type == "fire")
    env._calls = [fire_call]
    env._call_index = 0
    ga0 = env._hospital_a["general_beds"]
    ia0 = env._hospital_a["icu_beds"]
    act = DispatchGridAction(
        ambulance_units=0,
        police_units=0,
        fire_units=max(1, fire_call.correct_dispatch.get("fire", 1)),
        priority_level=3,
        hospital_choice="nearest",
        coordination_level="none",
        ambulance_staging="dispatch",
    )
    env.step(act)
    assert env._hospital_a["general_beds"] == ga0
    assert env._hospital_a["icu_beds"] == ia0


@pytest.mark.parametrize("task", ("hard", "crisis"))
def test_hard_or_crisis_episode_has_five_calls_with_surge(task: str):
    env = DispatchGridEnvironment(task=task)
    env.reset(seed=0)
    assert len(env._calls) == 5
    assert any(c.call_id == "SURGE001" for c in env._calls)
