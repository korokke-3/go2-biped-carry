"""MuJoCo Go2 that talks unitree_sdk2 DDS like the robot (rt/lowstate out, rt/lowcmd in), in real time, so that
deploy/go2_real.py runs unchanged against it. A remote-control script presses the buttons and moves the sticks
in rt/lowstate.wireless_remote, the way the real remote does.

  ~/go2_sim2real/.venv/bin/python deploy/sim_dds.py [--scenario tripod] [--unitree_curve] [--real_joints] [--video x.mp4]
  # in another shell: deploy/go2_real.py --mode sim ...   (deploy/run_sim_test.sh starts both)

DDS domain 1 on loopback (the robot uses domain 0), so nothing here can reach a robot.
Model, motor options and metrics as deploy/sim2sim_mujoco.py (go2_quad2hand's Go2 scene): torque
tau_ff + kp (q* - q) + kd (dq* - dq) per motor from the latest LowCmd at 500 Hz, clipped to the model's ctrlrange,
optionally to Unitree's torque-speed curve (knee x1.5, as trained) and with the measured gearbox friction / rotor
inertia / +1 kg base (--real_joints). The robot starts lying on the floor with the motors limp.
Prints per-segment velocity tracking (body frame), falls and lifted-paw contacts, and writes them to --out (JSON).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import remote  # noqa: E402
from go2_real import fallen  # noqa: E402
from policy import projected_gravity  # noqa: E402

XML = os.path.expanduser("~/go2_sim2real/refs/go2_quad2hand/go2_deploy/deploy_mujoco/robots/go2_mujoco/scene.xml")
SDK_JOINTS = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")]
from real_joints import REAL_JOINTS  # noqa: E402  hip / thigh / calf: Fs, mu, Fd, armature (deploy/real_joints.py)
# Unitree torque-speed curve (sim2sim_mujoco.CURVE): Y1 same dir, Y2 opposite, X1 flat to, X2 zero at [Nm, rad/s]
CURVE = {"calf": (30.3, 35.1, 9.0, 20.0), "other": (20.2, 23.4, 13.5, 30.0)}
LYING = {"hip": 0.0, "thigh": 1.2, "calf": -2.7}  # folded legs, belly on the floor (Unitree's lie-down pose, approx.)

# remote scripts: ("press", t, button) taps a button for 0.1 s, ("stick", t0, t1, {axis: value}) holds the sticks
# Tripod: hop then walk, every direction at full stick (go2_real defaults: fwd 0.3, back 0.2, side 0.15, turn 0.5)
SCENARIOS = {
    "tripod": [
        ("press", 0.3, "select"), ("press", 0.8, "start"), ("press", 4.5, "X"), ("press", 5.0, "A"),
        ("stick", 7, 11, {"ly": 1.0}), ("stick", 11, 15, {"rx": -1.0}), ("stick", 15, 19, {"ly": -1.0}),
        ("stick", 19, 23, {"rx": 1.0}), ("stick", 23, 27, {"lx": -1.0}),
        ("press", 29.0, "B"), ("press", 38.0, "Y"), ("press", 38.5, "A"),
        ("stick", 40.5, 44.5, {"ly": 1.0}), ("stick", 44.5, 48.5, {"rx": -1.0}), ("stick", 48.5, 52.5, {"ly": -1.0}),
        ("stick", 52.5, 56.5, {"lx": -1.0}),
        ("press", 58.5, "B"), ("press", 67.5, "select"), ("end", 69.5),
    ],
    # the walk gait only (Y before the first A), same sticks as the tripod scenario's walk part
    "walk": [
        ("press", 0.3, "select"), ("press", 0.8, "start"), ("press", 4.5, "Y"), ("press", 5.0, "A"),
        ("stick", 7, 11, {"ly": 1.0}), ("stick", 11, 15, {"rx": -1.0}), ("stick", 15, 19, {"ly": -1.0}),
        ("stick", 19, 23, {"lx": -1.0}), ("press", 25.0, "B"), ("press", 27.5, "select"), ("end", 29.5),
    ],
    # A, B, A twice in a row (the 2026-10-30 robot run fell on the second A), then select, start again, A, B
    "restand": [
        ("press", 0.3, "select"), ("press", 0.8, "start"), ("press", 4.5, "Y"), ("press", 5.0, "A"),
        ("press", 10.0, "B"), ("press", 19.0, "A"), ("press", 24.0, "B"), ("press", 33.0, "X"), ("press", 33.5, "A"),
        ("press", 38.5, "B"), ("press", 47.5, "select"), ("press", 50.5, "start"), ("press", 54.5, "A"),
        ("press", 59.5, "B"), ("press", 68.5, "select"), ("end", 70.5),
    ],
    # switch policies: stand up on the hind legs, walk every direction, down to four legs, up again, B (down, stop,
    # lie down, stand up), A again, select
    "biped": [
        ("press", 0.3, "select"), ("press", 0.8, "start"), ("press", 4.5, "A"),
        ("stick", 8, 12, {"ly": 1.0}), ("stick", 12, 16, {"rx": -1.0}), ("stick", 16, 20, {"lx": -1.0}),
        ("stick", 20, 24, {"ly": -1.0}), ("press", 26.0, "down"), ("press", 33.0, "up"), ("press", 40.0, "B"),
        ("press", 53.0, "A"), ("press", 60.0, "down"), ("press", 66.0, "select"), ("end", 68.0),
    ],
    # switch policies: stand up, the Y demo for filming (deploy/go2_real.py DEMO, 34 s), down, select
    "demo": [("press", 0.3, "select"), ("press", 0.8, "start"), ("press", 4.5, "A"), ("press", 10.0, "Y"),
             ("press", 46.0, "down"), ("press", 50.0, "select"), ("end", 52.0)],
    # stand, run the policy with no command, B, select (the first check on the robot, on the floor)
    "idle": [("press", 0.3, "select"), ("press", 0.8, "start"), ("press", 4.5, "A"), ("press", 14.5, "B"), ("press", 17.0, "select"), ("end", 19.0)],
}


def quat_rot(q, v):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="tripod", choices=list(SCENARIOS))
    p.add_argument("--unitree_curve", action="store_true")
    p.add_argument("--real_joints", action="store_true")
    p.add_argument("--free_leg", default="FR", help="lifted leg, for the paw-contact metric")
    p.add_argument("--bag", type=float, default=0.0, help="static load [kg] on the free calf (as sim2sim --bag)")
    p.add_argument("--bag_window", default="40,58", help="t0,t1 [s] of the load (tripod scenario: the walk gait)")
    p.add_argument("--temp_ramp", default=None,
                   help="'joint,t0,t1,T0,T1': one motor's reported temperature rises T0 -> T1 over t0..t1 (tests the stop)")
    p.add_argument("--video", default=None)
    p.add_argument("--out", default=None, help="metrics JSON")
    a = p.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

    model = mujoco.MjModel.from_xml_path(XML)
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in SDK_JOINTS])
    vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in SDK_JOINTS])
    act = np.array([model.actuator(n.replace("_joint", "")).id for n in SDK_JOINTS])
    lo, hi = model.actuator_ctrlrange[act, 0], model.actuator_ctrlrange[act, 1]
    kind = [n.split("_")[1] for n in SDK_JOINTS]
    curve = np.array([CURVE["calf" if k == "calf" else "other"] for k in kind])
    if a.real_joints:
        jf = np.array([REAL_JOINTS[k] for k in kind])
        model.dof_frictionloss[vadr] = 0.0
        model.dof_damping[vadr] = 0.0
        model.dof_armature[vadr] = jf[:, 3]
        model.body_mass[model.body("base_link").id] += 1.0

    data.qpos[0:3] = [0.0, 0.0, 0.12]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr] = [LYING[k] for k in kind]
    mujoco.mj_forward(model, data)
    for _ in range(500):  # settle limp on the floor
        mujoco.mj_step(model, data)

    ChannelFactoryInitialize(1, "lo")
    cmd_lock = threading.Lock()
    cmd = {"q": np.zeros(12), "dq": np.zeros(12), "kp": np.zeros(12), "kd": np.zeros(12), "tau": np.zeros(12), "n": 0}

    def on_cmd(msg):
        m = msg.motor_cmd
        with cmd_lock:
            for k in ("q", "dq", "kp", "kd", "tau"):
                cmd[k] = np.array([getattr(m[i], k) for i in range(12)])
            cmd["n"] += 1

    sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub.Init(on_cmd, 10)
    pub = ChannelPublisher("rt/lowstate", LowState_)
    pub.Init()
    state = unitree_go_msg_dds__LowState_()

    script = SCENARIOS[a.scenario]
    bag_t0, bag_t1 = (float(x) for x in a.bag_window.split(",")) if a.bag > 0 else (1e9, 1e9)
    t_end = next(e[1] for e in script if e[0] == "end")
    ramp = None
    if a.temp_ramp:
        j, t0, t1, T0, T1 = a.temp_ramp.split(",")
        ramp = (SDK_JOINTS.index(j), float(t0), float(t1), float(T0), float(T1))

    def remote_at(t):
        buttons, sticks = [], {}
        for e in script:
            if e[0] == "press" and e[1] <= t < e[1] + 0.1:
                buttons.append(e[2])
            elif e[0] == "stick" and e[1] <= t < e[2]:
                sticks.update(e[3])
        return remote.encode(buttons, **sticks)

    floor = model.geom("floor").id
    free_geom = model.geom(a.free_leg).id
    fall_bodies = {model.body(n).id for n in ["base_link"] + [f"{l}_hip" for l in ["FL", "FR", "RL", "RR"]]}
    bag_body = model.body(f"{a.free_leg}_calf").id
    renderer, frames = None, []
    if a.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        renderer = mujoco.Renderer(model, height=360, width=480)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 1.8, -15.0, 135.0

    print(f"[SIM] DDS domain 1 on lo, scenario '{a.scenario}' ({t_end:.0f} s), waiting for the controller ...", flush=True)
    t_wait = time.monotonic() + 30.0
    while cmd["n"] == 0 and time.monotonic() < t_wait:  # publish while waiting: the controller needs a lowstate first
        _publish(state, data, qadr, vadr, np.zeros(12), 40, remote.encode(), pub)
        time.sleep(0.002)
    if cmd["n"] == 0:
        sys.exit("[SIM] no rt/lowcmd within 30 s")

    log = []
    n = int(round(t_end / model.opt.timestep))
    t_start = time.perf_counter()
    lag_max = 0.0
    for k in range(n):
        t = k * model.opt.timestep
        with cmd_lock:
            c = {kk: v.copy() for kk, v in cmd.items() if kk != "n"}
        q, dq = data.qpos[qadr], data.qvel[vadr]
        tau = c["tau"] + c["kp"] * (c["q"] - q) + c["kd"] * (c["dq"] - dq)
        if a.unitree_curve:
            same = dq * tau > 0
            ymax = np.where(same, curve[:, 0], curve[:, 1])
            lim = np.where(np.abs(dq) < curve[:, 2], ymax,
                           np.clip(ymax - ymax / (curve[:, 3] - curve[:, 2]) * (np.abs(dq) - curve[:, 2]), 0.0, None))
            tau = np.clip(tau, -lim, lim)
        tau = np.clip(tau, lo, hi)
        data.ctrl[act] = tau
        if a.real_joints:
            data.qfrc_applied[vadr] = -((jf[:, 0] + jf[:, 1] * np.abs(tau)) * np.tanh(dq / 0.01) + jf[:, 2] * dq)
        data.xfrc_applied[bag_body, 2] = -a.bag * 9.81 if bag_t0 <= t < bag_t1 else 0.0
        mujoco.mj_step(model, data)
        temp = np.full(12, 40.0)
        if ramp:
            i, t0, t1, T0, T1 = ramp
            temp[i] = T0 + (T1 - T0) * np.clip((t - t0) / (t1 - t0), 0, 1)
        _publish(state, data, qadr, vadr, tau, temp, remote_at(t), pub)

        if k % 10 == 0:  # metrics at 50 Hz
            quat = data.qpos[3:7]
            up = quat_rot(quat, np.array([0.0, 0.0, 1.0]))[2]
            # heading: the nose on four / three legs, the belly when upright on the hind legs (biped tracking frame)
            upright = quat_rot(quat, np.array([1.0, 0.0, 0.0]))[2] > 0.7
            hx = quat_rot(quat, np.array([0.0, 0.0, -1.0] if upright else [1.0, 0.0, 0.0]))[:2]
            h = hx / max(np.linalg.norm(hx), 1e-6)
            v = data.qvel[0:2]
            wz = data.qvel[3:6] @ quat_rot(quat, np.eye(3))[2]
            fall = paw = False
            for con in data.contact[:data.ncon]:
                if floor in (con.geom1, con.geom2):
                    g = con.geom2 if con.geom1 == floor else con.geom1
                    fall |= model.geom_bodyid[g] in fall_bodies
                    paw |= g == free_geom
            fall |= fallen(projected_gravity(quat), np.sin(np.radians(50.0))) is not None  # go2_real's test
            log.append((t, v @ h, v @ np.array([-h[1], h[0]]), wz, up, data.qpos[2], fall, paw, float(upright)))
            if renderer is not None and k % 20 == 0:
                cam.lookat[:] = data.qpos[0:3]
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
        # real time
        lag = time.perf_counter() - t_start - (t + model.opt.timestep)
        lag_max = max(lag_max, lag)
        if lag < 0:
            time.sleep(-lag)

    L = np.array(log, dtype=np.float64)
    sel_t = next((e[1] for e in script if e[0] == "press" and e[2] == "select"), t_end)
    res = {"scenario": a.scenario, "unitree_curve": a.unitree_curve, "real_joints": a.real_joints, "bag": a.bag,
           "max_lag_s": lag_max, "lowcmd_msgs": cmd["n"], "segments": []}
    # a fall: base / hip on the floor or tipped over, while standing (the robot starts and ends lying on its belly)
    up_once = np.flatnonzero(L[:, 5] > 0.25)
    fell = L[(L[:, 0] > (L[up_once[0], 0] if len(up_once) else 1e9)) & (L[:, 0] < sel_t) & (L[:, 6] > 0), 0]
    res["fell_at"] = float(fell[0]) if len(fell) else None
    # lifted paw on the floor while the policy runs: from 2 s after each A to the next B
    presses = sorted((e[1], e[2]) for e in script if e[0] == "press")
    win = np.zeros(len(L), dtype=bool)
    for i, (tA, b) in enumerate(presses):
        if b == "A":
            tB = next((t for t, bb in presses[i + 1:] if bb in ("B", "select")), t_end)
            win |= (L[:, 0] >= tA + 2.0) & (L[:, 0] < tB)
    res["paw_on_floor_frac_policy"] = float(L[win, 7].mean()) if win.any() else None
    print(f"[SIM] done: real-time lag max {lag_max * 1e3:.1f} ms, {cmd['n']} lowcmd msgs")
    print(f"[SIM] {'fell at %.2f s' % res['fell_at'] if res['fell_at'] else 'no fall'}; lifted paw on the floor "
          f"{(res['paw_on_floor_frac_policy'] or 0) * 100:.1f} % of the policy time")
    for e in script:
        if e[0] != "stick":
            continue
        m = (L[:, 0] >= e[1] + 1.0) & (L[:, 0] < e[2])  # skip the first second of each segment
        seg = {"t": [e[1], e[2]], "sticks": e[3], "vx": L[m, 1].mean(), "vy": L[m, 2].mean(), "wz": L[m, 3].mean(),
               "up": L[m, 4].mean(), "height": L[m, 5].mean(), "upright": L[m, 8].mean()}
        res["segments"].append(seg)
        print(f"  {e[1]:5.1f}-{e[2]:5.1f} s {e[3]}: vx {seg['vx']:+.2f} vy {seg['vy']:+.2f} wz {seg['wz']:+.2f}  "
              f"up {seg['up']:.2f}  height {seg['height']:.2f}  upright {seg['upright']:.2f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    if renderer is not None:
        import imageio
        imageio.mimsave(a.video, frames, fps=25)
        print(f"[SIM] video {a.video}")
    sys.stdout.flush()
    os._exit(0)  # DDS threads would keep the process alive


def _publish(state, data, qadr, vadr, tau, temp, rc, pub):
    for i in range(12):
        m = state.motor_state[i]
        m.q, m.dq, m.tau_est = float(data.qpos[qadr[i]]), float(data.qvel[vadr[i]]), float(tau[i])
        m.temperature = int(np.broadcast_to(temp, (12,))[i])
    q = data.qpos[3:7]
    state.imu_state.quaternion = [float(x) for x in q]
    state.imu_state.gyroscope = [float(x) for x in data.qvel[3:6]]  # free-joint angular velocity: body frame
    acc_world = data.qacc[0:3] + np.array([0.0, 0.0, 9.81])
    state.imu_state.accelerometer = [float(x) for x in quat_rot(q, np.eye(3)).T @ acc_world]
    state.wireless_remote = rc
    state.tick += 1
    pub.Write(state)


if __name__ == "__main__":
    main()
