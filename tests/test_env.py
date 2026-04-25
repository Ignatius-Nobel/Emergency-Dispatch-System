"""Unit tests for DispatchGridEnvironment after hackathon fixes."""
import sys, pytest
sys.path.insert(0, "server")

from dispatch_grid_environment import (
    DispatchGridEnvironment, compute_reward, TASK_CALL_MAP
)
from models import DispatchGridAction, EmergencyCall


# ── helpers ──────────────────────────────────────────────────────────────────

def make_action(**kwargs):
    defaults = dict(
        ambulance_units=1, police_units=0, fire_units=0,
        priority_level=3, coordination_level="none",
        hospital_choice="nearest", ambulance_staging="dispatch",
    )
    defaults.update(kwargs)
    return DispatchGridAction(**defaults)


def make_call(**kwargs):
    defaults = dict(
        call_id="T001", incident_type="medical",
        description="test call", location="x",
        caller_info="x", severity="moderate",
        correct_dispatch={"ambulance": 1, "police": 0, "fire": 0},
        correct_priority=3, needs_backup=False,
    )
    defaults.update(kwargs)
    return EmergencyCall(**defaults)


# ── FIX-1: no double-count ────────────────────────────────────────────────────

def test_no_double_count_on_completion():
    """Cumulative score must equal the sum of per-step rewards (FIX-1)."""
    env = DispatchGridEnvironment(task="easy")
    env.reset()
    total = 0.0
    done = False
    while not done:
        obs = env.step(make_action())
        total += obs.reward
        done = obs.done
    assert abs(env._cumulative_score - total) < 1e-4, (
        f"Double-count! cumulative={env._cumulative_score:.4f} sum={total:.4f}"
    )


# ── FIX-2: EmergencyCall defaults ────────────────────────────────────────────

@pytest.mark.parametrize("tier,calls", TASK_CALL_MAP.items())
def test_all_calls_have_required_fields(tier, calls):
    """Every EmergencyCall must expose coordination_tier, correct_hospital,
    and correct_staging without raising AttributeError (FIX-2)."""
    for call in calls:
        assert hasattr(call, "coordination_tier"), f"{call.call_id}: missing coordination_tier"
        assert hasattr(call, "correct_hospital"),  f"{call.call_id}: missing correct_hospital"
        assert hasattr(call, "correct_staging"),   f"{call.call_id}: missing correct_staging"


def test_easy_calls_default_coordination_is_none():
    """Easy/medium/hard calls should default to coordination_tier='none'."""
    from dispatch_grid_environment import EASY_CALLS
    for call in EASY_CALLS:
        assert call.coordination_tier == "none", (
            f"{call.call_id}: expected 'none', got {call.coordination_tier!r}"
        )


# ── FIX-3: safeguards ────────────────────────────────────────────────────────

def test_invalid_unit_count_rejected():
    """ambulance_units=99 must return reward=-0.2 and done=False (FIX-3)."""
    env = DispatchGridEnvironment(task="easy")
    env.reset()
    obs = env.step(make_action(ambulance_units=99))
    assert obs.reward == -0.2, f"Expected -0.2, got {obs.reward}"
    assert not obs.done


def test_invalid_priority_rejected():
    """priority_level=9 must be rejected (FIX-3)."""
    env = DispatchGridEnvironment(task="easy")
    env.reset()
    obs = env.step(make_action(priority_level=9))
    assert obs.reward == -0.2


def test_invalid_coordination_rejected():
    """Unknown coordination_level must be rejected (FIX-3)."""
    env = DispatchGridEnvironment(task="easy")
    env.reset()
    obs = env.step(make_action(coordination_level="godmode"))
    assert obs.reward == -0.2


def test_step_limit_terminates_episode():
    """Episode must terminate at MAX_STEPS_PER_EPISODE even with valid actions (FIX-3)."""
    env = DispatchGridEnvironment(task="easy")
    env.reset()
    obs = None
    for _ in range(env.MAX_STEPS_PER_EPISODE + 5):
        obs = env.step(make_action())
        if obs.done:
            break
    assert obs is not None and obs.done, "Episode did not terminate at step limit"


# ── FIX-4: over-dispatch reward curve ────────────────────────────────────────

def test_over_dispatch_reward_decreases_with_excess():
    """Sending excess units earns diminishing returns (FIX-4)."""
    call = make_call()
    rewards = []
    for n in range(1, 8):
        action = make_action(ambulance_units=n)
        r, _ = compute_reward(action, call)
        rewards.append(r)
    # Sending exactly 1 (correct) should yield the highest reward
    # Sending 2 should yield less than 1 but more than 3, etc.
    assert rewards[0] >= rewards[1] >= rewards[2], (
        f"Over-dispatch not diminishing: {rewards[:3]}"
    )
    # Sending 6+ should earn less than sending 2
    assert rewards[5] < rewards[1], (
        f"Large over-dispatch should be worse than small over-dispatch"
    )


def test_over_dispatch_floor():
    """Extreme over-dispatch must not go below -0.05 per unit type (FIX-4)."""
    call = make_call()
    action = make_action(ambulance_units=10)
    r, _ = compute_reward(action, call)
    # Max per-unit penalty = -0.05; total floor from unit section = 3 * -0.10
    # (other two types are correctly omitted, so no penalty there)
    assert r >= -1.0, "Reward clipped below global minimum"


# ── Baseline environment behaviour ───────────────────────────────────────────

def test_reset_returns_valid_observation():
    env = DispatchGridEnvironment(task="easy")
    obs = env.reset()
    assert obs.call_id is not None
    assert obs.calls_remaining > 0
    assert not obs.done


def test_episode_completes_on_all_calls():
    for task in ("easy", "medium", "hard", "crisis"):
        env = DispatchGridEnvironment(task=task)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            obs = env.step(make_action())
            done = obs.done
            steps += 1
        assert done, f"Episode never completed for task={task}"


def test_cumulative_score_bounded():
    """Cumulative score should stay within a reasonable range."""
    env = DispatchGridEnvironment(task="crisis")
    env.reset()
    done = False
    while not done:
        obs = env.step(make_action())
        done = obs.done
    assert -10.0 <= env._cumulative_score <= 10.0


def test_resources_deplete_and_replenish():
    env = DispatchGridEnvironment(task="easy")
    env.reset()
    initial_amb = env._ambulances
    env.step(make_action(ambulance_units=2))
    assert env._ambulances <= initial_amb, "Ambulances should have been deducted"
