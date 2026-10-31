"""Joint friction and rotor inertia of a real Go2 per joint type: Fs [Nm], mu [-], Fd [Nm s/rad], armature [kg m^2].

Friction torque tau_f = (Fs + mu |tau|) tanh(qd / 0.01) + Fd qd (gearbox friction growing with the transmitted torque).
The values measured on our robot come from a calibration that is not public. They are read from
real_joints_local.json at the repository root (not in git); without that file the illustrative values below are used,
so results with --real_joints / the Switch tasks differ from the report.
"""
import json
import os

EXAMPLE = {"hip": (0.3, 0.1, 0.2, 0.01), "thigh": (0.3, 0.1, 0.2, 0.01), "calf": (0.3, 0.1, 0.2, 0.04)}
LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "real_joints_local.json")


def load() -> dict:
    if os.path.exists(LOCAL):
        with open(LOCAL) as f:
            return {k: tuple(v) for k, v in json.load(f).items()}
    print(f"[real_joints] {LOCAL} not found: illustrative values", flush=True)
    return dict(EXAMPLE)


REAL_JOINTS = load()
