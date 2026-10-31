"""Robot-side policy: raw sensor readings in, joint position targets out (no Isaac / MuJoCo imports).

Reads the deploy.json + policy.onnx that scripts/export_onnx.py writes, and builds the observation exactly as the
training env did (go2_carry/refwalk_env_cfg.py, tripod_env_cfg.py):
  per term: base_ang_vel x0.25, projected_gravity, velocity_commands x(2, 2, 0.25), joint_pos_rel, joint_vel_rel x0.05,
  last_action, [gait_mode one-hot], gait clock (sin, cos of t / period); each clipped to +-100
  history: `history` frames per term, oldest -> newest, filled with the first frame at reset (Isaac Lab)
  clock: t counts from the reset (the step the policy starts), as episode_length_buf * step_dt
  commands: each component below `cmd_deadzone` is 0 (refwalk_mdp.DeadzoneVelocityCommand)
All joint-indexed arrays are in the policy's joint order (deploy.json joint_names, Isaac Lab order).
deploy/sim2sim_mujoco.py builds the same observation inline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def projected_gravity(qwxyz: np.ndarray) -> np.ndarray:
    """World -z in the body frame for an orientation (w, x, y, z) (same formula as sim2sim_mujoco.proj_gravity)."""
    w, x, y, z = qwxyz
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])


class Policy:
    def __init__(self, export_dir: str | Path, gait: str | None = None, pitch_trim_deg: float = 0.0):
        self.pitch_trim_deg = pitch_trim_deg  # observed gravity tilted about body y (sim2sim_mujoco --imu_bias_deg)
        export_dir = Path(export_dir)
        self.cfg = json.loads((export_dir / "deploy.json").read_text())
        c = self.cfg
        self.joint_names: list[str] = c["joint_names"]
        self.default = np.asarray(c["default_angles"], dtype=np.float64)
        self.kp, self.kd, self.action_scale = float(c["kp"]), float(c["kd"]), float(c["action_scale"])
        self.dt = float(c["policy_dt"])
        self.history = int(c["history"])
        self.terms = c["obs_terms"]
        self.deadzone = float(c.get("cmd_deadzone", 0.0))
        # gait modes: {name: (one-hot index, period)}; exports without the table carry one fixed mode
        if "gait_modes" in c:
            self.modes = {k: (int(v[0]), float(v[1])) for k, v in c["gait_modes"].items()}
        else:
            self.modes = {"default": (int(c.get("gait_mode", 0)), float(c["gait_period"]))}
        self.set_gait(gait or next(n for n, (i, _) in self.modes.items() if i == c.get("gait_mode", i)))
        self.session = ort.InferenceSession(str(export_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        n_in = sum(t[1] for t in self.terms) * self.history
        onnx_in = self.session.get_inputs()[0].shape[-1]
        if onnx_in != n_in:
            raise ValueError(f"deploy.json describes {n_in} inputs, policy.onnx takes {onnx_in}")
        self.num_joints = len(self.joint_names)
        self.reset()

    def set_gait(self, name: str) -> None:
        """Choose the gait mode. Tripod: training fixed it for a whole episode, so only before reset(). Switch policies
        (modes quad / biped): any time, as in training; the clock keeps running (same period in both modes)."""
        if name not in self.modes:
            raise ValueError(f"gait '{name}' not in {list(self.modes)}")
        self.gait = name
        self.mode_index, self.period = self.modes[name]

    def reset(self) -> None:
        self.step_count = 0
        self.hist = None
        self.last_action = np.zeros(self.num_joints)

    def clean_command(self, cmd) -> np.ndarray:
        cmd = np.asarray(cmd, dtype=np.float64)
        return cmd * (np.abs(cmd) > self.deadzone)

    def observe(self, quat_wxyz, gyro, cmd, q, dq) -> np.ndarray:
        """One observation (with history) from IMU (w, x, y, z quaternion, body rates), command (vx, vy, wz) and
        joint angles / velocities in policy order. Advances the gait clock and the history."""
        t = self.step_count * self.dt
        ph = (t % self.period) / self.period
        g = projected_gravity(np.asarray(quat_wxyz, dtype=np.float64))
        if self.pitch_trim_deg:
            c, sn = np.cos(np.radians(self.pitch_trim_deg)), np.sin(np.radians(self.pitch_trim_deg))
            g = np.array([c * g[0] + sn * g[2], g[1], -sn * g[0] + c * g[2]])
        vals = {"base_ang_vel": gyro, "projected_gravity": g,
                "velocity_commands": self.clean_command(cmd), "joint_pos_rel": np.asarray(q) - self.default,
                "joint_vel_rel": dq, "last_action": self.last_action,
                "gait_phase": [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)],
                "gait_mode": np.eye(2)[self.mode_index]}
        frame = [np.clip(np.asarray(vals[n], dtype=np.float64) * np.asarray(s), -100, 100) for n, _d, s in self.terms]
        if self.hist is None:
            self.hist = [[f.copy() for _ in range(self.history)] for f in frame]
        else:
            for i, f in enumerate(frame):
                self.hist[i] = self.hist[i][1:] + [f]
        self.step_count += 1
        return np.concatenate([np.concatenate(h) for h in self.hist]).astype(np.float32)[None]

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Joint position targets [rad], policy order. The raw action is kept for the next observation."""
        self.last_action = self.session.run(None, {self.input_name: obs})[0][0].astype(np.float64)
        return self.default + self.action_scale * self.last_action
