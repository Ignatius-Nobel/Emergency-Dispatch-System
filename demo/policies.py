"""Reference policies for MVE demo (baseline vs capacity-aware)."""
from __future__ import annotations

from typing import Any, Dict


def nearest_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Always route to nearest hospital; modest unit counts from keywords."""
    desc = (
        (obs.get("call_description") or "") + " " + (obs.get("incident_type") or "")
    ).lower()
    severity = (obs.get("severity") or "moderate").lower()
    priority = {"critical": 4, "severe": 3, "moderate": 2, "minor": 1}.get(severity, 2)

    medical_kw = (
        "medical", "ambulance", "collapsed", "chest", "unconscious", "breathing",
        "choking", "injured", "casualty", "cardiac", "patient", "trauma",
    )
    fire_kw = ("fire", "smoke", "burning", "explosion", "chemical", "hazmat", "gas", "flame")
    police_kw = (
        "crime", "robbery", "burglary", "armed", "weapon", "assault", "domestic",
        "shooter", "hostage", "theft", "fight", "stabbing",
    )

    needs_amb = any(k in desc for k in medical_kw)
    needs_fire = any(k in desc for k in fire_kw)
    needs_pol = any(k in desc for k in police_kw)
    if not needs_amb and not needs_fire and not needs_pol:
        needs_amb = True

    patients = max(1, int(obs.get("patients", 1)))
    amb = min(5, max(1, patients)) if needs_amb else 0
    pol = min(5, 2 if priority >= 3 else 1) if needs_pol else 0
    fire = min(5, 2 if priority >= 3 else 1) if needs_fire else 0
    if amb == 0 and pol == 0 and fire == 0:
        amb = 1

    return {
        "ambulance_units": amb,
        "police_units": pol,
        "fire_units": fire,
        "priority_level": priority,
        "backup_requested": False,
        "hospital_choice": "nearest",
        "coordination_level": "none",
        "ambulance_staging": "dispatch",
    }


def capacity_aware_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    For severe/critical medical, use bed-aware routing (auto); otherwise nearest.
    Contrasts with nearest_policy, which always uses zone-shortest hospital.
    """
    out = nearest_policy(obs)
    it = (obs.get("incident_type") or "").lower()
    sev = (obs.get("severity") or "").lower()
    if it == "medical" and sev in ("severe", "critical"):
        out["hospital_choice"] = "auto"
    else:
        out["hospital_choice"] = "nearest"
    return out
