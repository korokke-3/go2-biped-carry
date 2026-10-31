"""Sim2sim: run an exported RefWalk policy (scripts/export_onnx.py) in MuJoCo (go2_quad2hand's Go2 model).

  ~/go2_sim2real/.venv_mujoco/bin/python deploy/sim2sim_mujoco.py <run>/exported [--video out.mp4]

Starts in the quadruped stance on the floor (as the real robot would) and runs a fixed command schedule:
stand up, forward, turn left, backward, turn right, stop. Prints per-segment tracking, falls and knee contacts.
Observation layout / scales / joint order come from deploy.json (Isaac Lab order, mapped to MuJoCo by joint name).
PD at 500 Hz (MuJoCo dt 0.002, like go2_quad2hand's deploy_mujoco), policy at 50 Hz.
"""
import argparse
import json
import os

import mujoco
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.expanduser("~/go2_sim2real/refs/go2_quad2hand/go2_deploy/deploy_mujoco/robots/go2_mujoco/scene.xml")

p = argparse.ArgumentParser()
p.add_argument("exported")
p.add_argument("--video", default=None)
p.add_argument("--schedule", default="0:4:0,0,0;4:8:0.3,0,0;8:12:0,0,0.5;12:16:-0.3,0,0;16:20:0,0,-0.5;20:24:0,0,0",
               help="t0:t1:vx,vy,wz;...")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--frictionloss", type=float, default=None, help="override joint frictionloss [Nm] (model: 0.2)")
p.add_argument("--damping", type=float, default=None, help="override joint damping [Nm s/rad] (model: 0.1)")
p.add_argument("--armature", type=float, default=None, help="override joint armature (model: 0.01)")
p.add_argument("--foot_condim", type=int, default=None, help="override foot contact dims (model: 6 = torsional + rolling)")
p.add_argument("--soft_floor", type=float, default=None, help="floor contact time constant [s] (MuJoCo solref, default 0.02): larger = softer (carpet)")
p.add_argument("--foot_friction", type=float, default=None, help="override foot sliding friction (model: 0.6)")
p.add_argument("--no_arm_contact", action="store_true", help="disable contacts between the left and right front legs")
p.add_argument("--unitree_curve", action="store_true",
               help="Isaac's training torque-speed curve (unitree_rl_lab Go2HV, knee x1.5) instead of MuJoCo's flat limits")
p.add_argument("--bag", type=float, default=0.0, help="(Tripod) static load [kg] on the lifted calf from 2 s (no swing)")
p.add_argument("--free_leg", default="FR", help="(Tripod) lifted leg: FR (Tripod / High / Cool) or RR (TripodHind)")
p.add_argument("--real_joints", action="store_true",
               help="joint friction / rotor inertia of a real Go2 (deploy/real_joints.py): tau_f = (Fs + mu |tau|) "
                    "tanh(qd / 0.01) + Fd qd per joint type, armature per joint type, base +1.0 kg; replaces the "
                    "model's frictionloss 0.2 / damping 0.1")
p.add_argument("--real_parts", default="friction,armature,payload",
               help="with --real_joints: which measured parts to apply (ablation)")
p.add_argument("--real_fric_scale", default="1,1,1",
               help="with --real_joints: multipliers on Fs, mu, Fd (ablation)")
p.add_argument("--real_fric_joints", default="hip,thigh,calf", help="with --real_joints: joint types given friction")
p.add_argument("--imu_bias_deg", type=float, default=0.0, help="tilt the observed gravity about the body y axis [deg] (IMU mounting / calibration error)")
p.add_argument("--torque_limit", type=float, default=None, help="clip all motor torques to +-this [Nm]")
p.add_argument("--down_at", type=float, default=None,
               help="(switch policies) set the gait mode to quadruped (one-hot index 0) from this time [s]")
p.add_argument("--scripted_down_at", type=float, default=None,
               help="(fallback test) from this time ignore the policy and move the joints linearly to the default "
                    "quadruped pose over --scripted_down_s with --scripted_kp / --scripted_kd (deploy STAND gains)")
p.add_argument("--scripted_down_s", type=float, default=1.0)
p.add_argument("--scripted_kp", type=float, default=60.0)
p.add_argument("--scripted_kd", type=float, default=5.0)
p.add_argument("--up_at", type=float, default=None, help="(switch policies, with --down_at) back to biped at this time")
a = p.parse_args()

cfg = json.load(open(os.path.join(a.exported, "deploy.json")))
policy = ort.InferenceSession(os.path.join(a.exported, "policy.onnx"), providers=["CPUExecutionProvider"])
model = mujoco.MjModel.from_xml_path(XML)
model.opt.timestep = 0.002
decim = int(round(cfg["policy_dt"] / model.opt.timestep))
if a.frictionloss is not None:
    model.dof_frictionloss[6:] = a.frictionloss
if a.damping is not None:
    model.dof_damping[6:] = a.damping
if a.armature is not None:
    model.dof_armature[6:] = a.armature
for fg in ["FL", "FR", "RL", "RR"]:
    if a.foot_condim is not None:
        model.geom_condim[model.geom(fg).id] = a.foot_condim
    if a.soft_floor is not None:
        model.geom_solref[:, 0] = a.soft_floor
    if a.foot_friction is not None:
        model.geom_friction[model.geom(fg).id, 0] = a.foot_friction
if a.no_arm_contact:  # FL geoms contype 2, FR geoms contype 4, both still collide with conaffinity-1 geoms (floor, body)
    for b in range(model.nbody):
        nm = model.body(b).name
        if nm.startswith(("FL_", "FR_")):
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == b:
                    model.geom_contype[g] = 2 if nm.startswith("FL_") else 4
                    model.geom_conaffinity[g] = 1
# hip / thigh / calf: Fs [Nm], mu [-], Fd [Nm s/rad], armature [kg m^2]
from real_joints import REAL_JOINTS  # noqa: E402  (measured values: real_joints_local.json, not in git)
RP = set(a.real_parts.split(",")) if a.real_joints else set()
if "friction" in RP:
    model.dof_frictionloss[6:] = 0.0
    model.dof_damping[6:] = 0.0
if "payload" in RP:
    model.body_mass[model.body("base_link").id] += 1.0
data = mujoco.MjData(model)

names = cfg["joint_names"]
qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
act_of = np.array([model.actuator(n.replace("_joint", "")).id for n in names])
default = np.array(cfg["default_angles"], dtype=np.float64)
kp, kd, ascale = cfg["kp"], cfg["kd"], cfg["action_scale"]
H = cfg["history"]
if a.real_joints:
    jf = np.array([REAL_JOINTS[n.split("_")[1]] for n in names])
    fs = np.array([float(x) for x in a.real_fric_scale.split(",")])
    on = np.array([n.split("_")[1] in a.real_fric_joints.split(",") for n in names], dtype=np.float64)
    jf[:, :3] *= fs[None, :] * on[:, None]
    if "armature" in RP:
        model.dof_armature[vadr] = jf[:, 3]
terms = cfg["obs_terms"]
TRIPOD = "gait_mode" in cfg and "biped" not in cfg.get("gait_modes", {})  # (switch policies carry a one-hot too)  # Tripod task: quadruped posture, FR lifted; "upright" = body z up, heading = body x
lo, hi = model.actuator_ctrlrange[act_of, 0], model.actuator_ctrlrange[act_of, 1]
# unitree_rl_lab Go2HV: Y1 20.2, Y2 23.4, X1 13.5, X2 30 (hip, thigh); knee: torques x1.5, speeds / 1.5 (RefWalk-V2+)
CURVE = np.array([[30.3, 35.1, 9.0, 20.0] if "calf" in n else [20.2, 23.4, 13.5, 30.0] for n in names])
if a.torque_limit is not None:
    lo, hi = np.maximum(lo, -a.torque_limit), np.minimum(hi, a.torque_limit)

# quadruped stance on the floor
data.qpos[0:3] = [0.0, 0.0, 0.33]
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qpos[qadr] = default
mujoco.mj_forward(model, data)

floor = model.geom("floor").id if "floor" in [model.geom(i).name for i in range(model.ngeom)] else 0
knee_bodies = {model.body(n).id for n in ["RL_thigh", "RR_thigh", "RL_calf", "RR_calf"]}
foot_geoms = {model.geom(n).id for n in ["FL", "FR", "RL", "RR"]}
fall_bodies = {model.body(n).id for n in ["base_link"] + [f"{l}_hip" for l in ["FL", "FR", "RL", "RR"]]}


def proj_gravity(qwxyz):
    w, x, y, z = qwxyz
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])


def quat_rot(qwxyz, v):
    w, x, y, z = qwxyz
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ v


def gait_mode_at(t):
    if a.down_at is not None and t >= a.down_at and (a.up_at is None or t < a.up_at):
        return 0
    return cfg.get("gait_mode", 0)


sched = []
for seg in a.schedule.split(";"):
    t0, t1, c = seg.split(":")
    sched.append((float(t0), float(t1), np.array([float(v) for v in c.split(",")])))
T = sched[-1][1]

hist = None
action = np.zeros(12)
target = default.copy()
renderer, frames = None, []
if a.video:
    os.environ.setdefault("MUJOCO_GL", "egl")
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.0, -15.0, 135.0

log = []
fell_t = None
knee_steps = 0
up_steps = 0
n_pol = int(round(T / cfg["policy_dt"]))
for k in range(n_pol):
    t = k * cfg["policy_dt"]
    cmd = next((c for t0, t1, c in sched if t0 <= t < t1), np.zeros(3))
    q, dq = data.qpos[qadr], data.qvel[vadr]
    quat = data.qpos[3:7].copy()
    omega_b = data.qvel[3:6].copy()  # free joint angular velocity: body frame
    g = proj_gravity(quat)
    if a.imu_bias_deg:
        c, sn = np.cos(np.radians(a.imu_bias_deg)), np.sin(np.radians(a.imu_bias_deg))
        g = np.array([c * g[0] + sn * g[2], g[1], -sn * g[0] + c * g[2]])
    phase = (t % cfg["gait_period"]) / cfg["gait_period"]
    vals = {"base_ang_vel": omega_b, "projected_gravity": g, "velocity_commands": cmd,
            "joint_pos_rel": q - default, "joint_vel_rel": dq, "last_action": action,
            "gait_phase": np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]),
            "gait_mode": np.eye(2)[gait_mode_at(t)]}
    frame = [np.clip(np.asarray(vals[n], dtype=np.float64) * np.asarray(s), -100, 100) for n, d, s in terms]
    if hist is None:
        hist = [[f.copy() for _ in range(H)] for f in frame]  # Isaac Lab fills the history with the first frame
    else:
        for i, f in enumerate(frame):
            hist[i] = hist[i][1:] + [f]
    obs = np.concatenate([np.concatenate(h) for h in hist]).astype(np.float32)[None]
    action = policy.run(None, {"obs": obs})[0][0].astype(np.float64)
    target = default + ascale * action
    kp_, kd_ = kp, kd
    if a.scripted_down_at is not None and t >= a.scripted_down_at:
        if t - a.scripted_down_at < 1e-9 + cfg["policy_dt"] / 2:
            q_from = data.qpos[qadr].copy()
        f = min((t - a.scripted_down_at) / a.scripted_down_s, 1.0)
        target = q_from + f * (default - q_from)
        kp_, kd_ = a.scripted_kp, a.scripted_kd
        vz_min = min(globals().get("vz_min", 0.0), data.qvel[2])
        head_min = min(globals().get("head_min", 9.0), data.xpos[model.body("FL_hip").id][2], data.xpos[model.body("FR_hip").id][2])
    for _ in range(decim):
        tau = kp_ * (target - data.qpos[qadr]) - kd_ * data.qvel[vadr]
        if a.unitree_curve:  # Y1 same direction / Y2 opposite, flat up to X1, linear to 0 at X2 (UnitreeActuator)
            qd = data.qvel[vadr]
            same = qd * tau > 0
            ymax = np.where(same, CURVE[:, 0], CURVE[:, 1])
            lim = np.where(np.abs(qd) < CURVE[:, 2], ymax,
                           np.clip(ymax - ymax / (CURVE[:, 3] - CURVE[:, 2]) * (np.abs(qd) - CURVE[:, 2]), 0.0, None))
            tau = np.clip(tau, -lim, lim)
        data.ctrl[act_of] = np.clip(tau, lo, hi)
        if "friction" in RP:  # gearbox friction against the joint motion, growing with the transmitted torque
            qd = data.qvel[vadr]
            tm = np.abs(data.ctrl[act_of])
            data.qfrc_applied[vadr] = -((jf[:, 0] + jf[:, 1] * tm) * np.tanh(qd / 0.01) + jf[:, 2] * qd)
        if a.bag > 0 and t >= 2.0:
            data.xfrc_applied[model.body(f"{a.free_leg}_calf").id, 2] = -a.bag * 9.81
        mujoco.mj_step(model, data)
    # metrics
    up = -proj_gravity(data.qpos[3:7])[2 if TRIPOD else 0]  # cos(body z / body x, world up)
    belly = quat_rot(data.qpos[3:7], np.array([1.0, 0.0, 0.0] if TRIPOD else [0.0, 0.0, -1.0]))[:2]
    h = belly / max(np.linalg.norm(belly), 1e-6)
    v = data.qvel[0:2]
    vx, vy = v @ h, v @ np.array([-h[1], h[0]])
    wz = data.qvel[3:6] @ quat_rot(data.qpos[3:7], np.eye(3))[2]  # world z component of the body-frame rate
    touch_knee = touch_fall = touch_fr = False
    for c in data.contact[:data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if c.geom1 == floor or c.geom2 == floor:
            other = b2 if c.geom1 == floor else b1
            g_other = c.geom2 if c.geom1 == floor else c.geom1
            touch_knee |= (other in knee_bodies) and g_other not in foot_geoms  # the foot sphere sits on the calf body
            touch_fall |= other in fall_bodies
            touch_fr |= g_other == model.geom(a.free_leg).id
    fr_steps = (fr_steps if k else 0) + touch_fr
    if up > 0.8:
        up_steps += 1
        knee_steps += touch_knee
    # in the quadruped mode of a switch policy the body lies level: a fall = base / hips on the floor or tipped > 60 deg
    # from level, from 3 s after the mode change (the transition passes through every pitch)
    quad_now = a.down_at is not None and gait_mode_at(t) == 0
    tipped = (-proj_gravity(data.qpos[3:7])[2] < 0.5) if quad_now else (up < 0.5)
    settling = a.down_at is not None and (a.down_at <= t < a.down_at + 3.0 or
                                          (a.up_at is not None and a.up_at <= t < a.up_at + 3.0))
    if fell_t is None and t > 3.0 and (touch_fall or (tipped and not settling)):
        fell_t = t
    log.append((t, *cmd, vx, vy, wz, up, data.qpos[2]))
    if renderer is not None and k % 2 == 0:
        cam.lookat[:] = data.qpos[0:3]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

L = np.array(log)
if a.down_at is not None:
    g_end = proj_gravity(data.qpos[3:7])
    print(f"end: body tilt {np.degrees(np.arccos(np.clip(-g_end[2], -1, 1))):.0f} deg, base height {data.qpos[2]:.2f} m, "
          f"joint dev from default max {np.abs(data.qpos[qadr] - default).max():.2f} rad")
if a.scripted_down_at is not None:
    print(f"scripted down: min base vz {globals().get('vz_min', 0):+.2f} m/s, min front hip height {globals().get('head_min', 9):.2f} m")
print(f"fell at {fell_t:.2f} s" if fell_t is not None else "no fall")
print(f"upright from {L[np.argmax(L[:, 7] > 0.8), 0]:.2f} s" if (L[:, 7] > 0.8).any() else "never upright")
print(f"knee/shin on the floor while upright: {knee_steps / max(up_steps, 1) * 100:.1f} % of the time")
if TRIPOD:
    print(f"lifted {a.free_leg} paw on the floor: {fr_steps / n_pol * 100:.1f} % of the time")
for t0, t1, c in sched:
    m = (L[:, 0] >= t0 + 1.0) & (L[:, 0] < t1)  # skip the first second of each segment
    if m.any():
        print(f"  {t0:4.0f}-{t1:4.0f} s cmd {c}: vx {L[m, 4].mean():+.2f} vy {L[m, 5].mean():+.2f} "
              f"wz {L[m, 6].mean():+.2f}  upright {L[m, 7].mean():.2f}  height {L[m, 8].mean():.2f}")
if renderer is not None:
    import imageio
    imageio.mimsave(a.video, frames, fps=25)
    print("video", a.video)
