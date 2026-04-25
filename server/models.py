# Copyright (c) Meta Platforms, Inc. and affiliates.
# Load canonical models from repository root (when only server/ is on sys.path).
import importlib.util
import os

_root_models = os.path.join(os.path.dirname(__file__), "..", "models.py")
_spec = importlib.util.spec_from_file_location("emergency_dispatch_models", _root_models)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DispatchGridAction = _mod.DispatchGridAction
DispatchGridObservation = _mod.DispatchGridObservation
EmergencyCall = _mod.EmergencyCall

__all__ = ("DispatchGridAction", "DispatchGridObservation", "EmergencyCall")
