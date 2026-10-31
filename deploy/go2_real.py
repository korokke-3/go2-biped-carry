"""Run an exported policy (scripts/export_onnx.py) on the Go2 over unitree_sdk2 low-level control, or against the DDS
simulator deploy/sim_dds.py with the very same code and messages.

  # simulator (DDS domain 1 on loopback; start deploy/sim_dds.py first)
  ~/go2_sim2real/.venv/bin/python deploy/go2_real.py --mode sim --export_dir <run>/exported_real
  # robot (DDS domain 0 on the interface wired to the Go2, 192.168.123.0/24)
  ~/go2_sim2real/.venv/bin/python deploy/go2_real.py --mode real --iface <if> --export_dir <run>/exported_real

Policies: Tripod (quadruped start, FR leg lifted by the policy, back to four legs on B) and the biped <-> quadruped
switch policies (switch_env_cfg.py: quadruped start, the policy stands up on its hind legs and goes down again).
Design follows the usual Unitree low-level controller pattern:

Remote
  select  (first, while limp) proves the emergency key reads before anything moves: start is refused until then
  start   stand up: limp -> the policy's default (quadruped) pose over --stand_time s, Kp/Kd --stand_kp/--stand_kd
  X / Y   (Tripod, standing) choose the gait mode: X = hop, Y = walk. Training fixed the mode for a whole episode, so
          it can only be changed here, not while the policy runs
  down/up (switch policies) as the ref's robot code (FSMState_Walk UP / DOWN). down in the biped mode: with --down
          scripted (default) a straight move to the default quadruped pose in --down_time s, then HOLD on four legs
          (the policies did not learn to go down; MuJoCo with the measured joints: no fall in 36 / 36); with --down
          policy the policy's quadruped mode. up on four legs (HOLD) = A: the policy stands up on its hind legs
  A       run the policy (gait clock and history start at 0, as an episode reset; Kp/Kd from deploy.json)
  B       stop the policy (switch policies in the biped mode: first down as above - with --down policy: once the
          body is level for 0.5 s, damping if not level within 8 s): joints move back to the default pose over
          --return_time s (a lifted leg folds, swings
          back and puts the paw down under the shoulder), then the robot lies down and stands up again so that the
          feet are where the first stand put them. A again only when the pose is within --start_tol
  select  damping (Kp 0, Kd --damping_kd): the robot sinks down. start stands it up again once it lies still
  left stick up/down = vx, left/right = vy; right stick left/right = wz (full stick = --max_fwd / --max_back /
          --max_vy / --max_wz; each component below the training deadzone, 0.1, is 0)

Safety
  damping when rt/lowstate is older than --stale_timeout, and while the policy runs when the body has fallen (rolled
  past --max_tilt sideways, on its back, or nose more than 30 deg down: fallen() - valid on four, three and two legs)
  or a joint is --joint_margin outside its range; on Ctrl-C / SIGTERM / SIGHUP; on any exception
  targets are clamped to --target_margin rad outside the joint limits and to --max_step rad per policy step; raw
  actions to +-(--action_clip). Not to the limits themselves: training (Isaac Lab JointPositionAction) and the ref's
  robot code (go2_quad2hand FSMState_Walk) send unclipped targets, and the Tripod policies set the stance knee's
  target past its straight limit for extension torque (TripodCool c2 FL calf -0.48 vs limit -0.84: clamping halved it)
  motor temperature: a warning at --temp_warn; at --temp_stop the policy stops as on B (a switch policy goes down to
  four legs first), and A / up are refused until every motor is below --temp_stop
Log: every policy step to an .npz (deploy/real_logs/ by default): state, command, obs, action, targets, q, dq,
tau_est, IMU, motor temperatures, foot force, buttons.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import remote  # noqa: E402
from policy import Policy, projected_gravity  # noqa: E402

ZERO_TORQUE, STAND, HOLD, POLICY, DAMPING = range(5)
NAMES = ("ZERO_TORQUE", "STAND", "HOLD", "POLICY", "DAMPING")
DOMAIN = {"real": 0, "sim": 1}  # the simulator never shares a DDS domain with the robot
WRITE_PERIOD = 0.002  # rt/lowcmd at 500 Hz (unitree_sdk2 Go2 low-level examples)

# unitree_sdk2 Go2 motor order (LowCmd.motor_cmd[0:12] / LowState.motor_state[0:12])
SDK_JOINTS = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")]
# joint limits [rad], Unitree go2.urdf
LIMITS = {"hip": (-1.0472, 1.0472), "F_thigh": (-1.5708, 3.4907), "R_thigh": (-0.5236, 4.5379), "calf": (-2.7227, -0.83776)}


def _limit(name):
    part = name.split("_")[1]
    return LIMITS[(name[0] + "_thigh") if part == "thigh" else part]


# lying down on B before standing up again: thighs as on the real robot lying after the release (1.13-1.43), knees
# folded within the limit; the limp second that follows lets the body settle onto the belly (real calves -2.8)
LIE_THIGH = 1.25
CALF_FOLD = -2.6  # [rad] knee folded (limit -2.72) while a lifted leg swings back on B: the paw stays ~0.2 m up
LOWER = np.array([_limit(n)[0] for n in SDK_JOINTS])
LIE = np.array([{"hip": 0.0, "thigh": LIE_THIGH, "calf": CALF_FOLD}[n.split("_")[1]] for n in SDK_JOINTS])
UPPER = np.array([_limit(n)[1] for n in SDK_JOINTS])


def check_iface(iface):
    """CycloneDDS needs the interface up with an IPv4 address on the robot's network (192.168.123.0/24)."""
    import subprocess
    out = subprocess.run(["ip", "-4", "-br", "addr", "show", iface], capture_output=True, text=True).stdout.split()
    if not out:
        sys.exit(f"[ERROR] インターフェース {iface} がありません(`ip -br link` で名前を確認してください)。")
    addrs = [x for x in out[2:] if "/" in x]
    if out[1] == "DOWN" or not any(x.startswith("192.168.123.") for x in addrs):
        sys.exit(f"[ERROR] {iface} が Go2 のネットワークにつながっていません(状態 {out[1]}、アドレス {addrs or 'なし'})。\n"
                 f"        ケーブルを確かめて `nmcli con up go2` を実行し、`ip -br addr show {iface}` に "
                 f"192.168.123.99 が出てから、もう一度実行してください。")


def fallen(g: np.ndarray, sin_roll: float) -> str | None:
    """Fall test from gravity in the body frame, for any of the trained postures: on four / three legs g ~ (0, 0, -1),
    upright on the hind legs g ~ (-1, 0, 0), and every pitch in between while standing up or going down. Fallen =
    rolled sideways (|g_y| > sin_roll), on the back (g_z > 0.5: the backwards fall from upright) or nose down by more
    than 30 deg (g_x > 0.5)."""
    if abs(g[1]) > sin_roll:
        return f"rolled {np.degrees(np.arcsin(min(1.0, abs(g[1])))):.0f} deg"
    if g[2] > 0.5:
        return "on its back"
    if g[0] > 0.5:
        return "nose down"
    return None


# Demo for filming (Y while upright, switch policies): (duration s, (vx, vy, wz)), each move ramped in and out over
# DEMO_RAMP s, pauses between; opposite moves of equal length so the robot ends near where it started. Speeds: above
# the 0.1 training deadzone; MuJoCo with the measured joints gives about half of them (logs/summary_switch.txt).
DEMO = [(2.0, (0.0, 0.0, 0.0)), (4.0, (0.0, 0.0, 0.6)), (2.0, (0.0, 0.0, 0.0)), (4.0, (0.0, 0.0, -0.6)),
        (2.0, (0.0, 0.0, 0.0)), (3.0, (0.3, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0)), (3.0, (-0.3, 0.0, 0.0)),
        (2.0, (0.0, 0.0, 0.0)), (3.0, (0.0, 0.25, 0.0)), (2.0, (0.0, 0.0, 0.0)), (3.0, (0.0, -0.25, 0.0)),
        (2.0, (0.0, 0.0, 0.0))]
DEMO_RAMP = 0.5
DEMO_S = sum(d for d, _ in DEMO)


def demo_command(t: float, scale: float) -> np.ndarray:
    for dur, c in DEMO:
        if t < dur:
            k = min(1.0, t / DEMO_RAMP, (dur - t) / DEMO_RAMP)
            v = np.array(c) * scale * k
            return v * (np.abs(v) >= 0.1)  # ramps start at the deadzone edge, not below it (it would read as stop)
        t -= dur
    return np.zeros(3)


class Controller:
    def __init__(self, a):
        self.a = a
        self.policy = Policy(a.export_dir, a.gait)
        # pitch trim, reached --trim_ramp s after standing up (the robot leant 3-8 deg back after standing up with the
        # full trim from the start, 2026-10-31); 0 on four legs
        self.trim = a.pitch_trim if a.pitch_trim is not None else (
            -5.0 if a.mode == "real" and "biped" in self.policy.modes else 0.0)
        self.biped_since = None
        self.demo_t0 = None  # demo sequence start (Y while upright, --demo)
        p = self.policy
        self.sdk_to_pol = np.array([SDK_JOINTS.index(n) for n in p.joint_names])  # sdk_array[sdk_to_pol] -> policy
        self.pol_to_sdk = np.array([p.joint_names.index(n) for n in SDK_JOINTS])  # policy_array[pol_to_sdk] -> sdk
        self.default = p.default[self.pol_to_sdk]
        self.kp, self.kd = np.full(12, p.kp), np.full(12, p.kd)
        self.stand_kp, self.stand_kd = np.full(12, a.stand_kp), np.full(12, a.stand_kd)
        self.zeros = np.zeros(12)
        self.sin_roll = float(np.sin(np.radians(a.max_tilt)))
        self.switchable = "biped" in self.policy.modes  # biped <-> quadruped switch policy
        if self.switchable:
            self.policy.set_gait("biped")
        self.stop_pending = None  # time B (or heat) asked a switch policy to go down before stopping
        self.level_since = None
        self.state = ZERO_TORQUE
        self.edges = remote.Edges()
        self.q_des = None
        self.move = None
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._msg, self._msg_t, self._msg_n = None, 0.0, 0
        self.warned_hot = False
        self.select_ok = False
        self.stats = {"overruns": 0, "step_clamps": 0, "limit_clamps": 0, "action_clips": 0, "infer_ms_max": 0.0,
                      "max_abs_action": 0.0, "max_temp": 0}
        self.log = {k: [] for k in ("t", "state", "gait", "age", "cmd", "obs", "action", "q_des", "q", "dq", "tau",
                                    "quat", "gyro", "acc", "temp", "foot_force", "keys", "sticks")}

    # ------------------------------------------------------------------ DDS
    def connect(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.utils.thread import RecurrentThread

        iface = self.a.iface if self.a.mode == "real" else "lo"
        if self.a.mode == "real":
            check_iface(iface)
        ChannelFactoryInitialize(DOMAIN[self.a.mode], iface)
        print(f"[INFO] DDS domain {DOMAIN[self.a.mode]} on {iface}")
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        t_end = time.monotonic() + 5.0
        while self._msg is None:
            if time.monotonic() > t_end:
                raise RuntimeError("no rt/lowstate within 5 s: cable / interface / simulator not running?")
            time.sleep(0.05)
        if self.a.mode == "real":
            self._release_sport_mode()
        self._crc = CRC()
        self._cmd = unitree_go_msg_dds__LowCmd_()
        self._cmd.head[0], self._cmd.head[1] = 0xFE, 0xEF
        self._cmd.level_flag = 0xFF  # low-level
        self._cmd.gpio = 0
        for i in range(20):
            m = self._cmd.motor_cmd[i]
            # zero torque; q / dq 0 rather than the SDK's PosStopF / VelStopF sentinels, which a simulator would use
            # literally in kp * (q - q_meas) + kd * (dq - dq_meas)
            m.mode, m.q, m.dq, m.kp, m.kd, m.tau = 0x01, 0.0, 0.0, 0.0, 0.0, 0.0
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        self._writer = RecurrentThread(interval=WRITE_PERIOD, target=self._write, name="lowcmd")
        self._writer.Start()

    def _on_state(self, msg):
        with self._lock:
            self._msg, self._msg_t = msg, time.monotonic()
            self._msg_n += 1

    def _write(self):
        with self._cmd_lock:
            self._cmd.crc = self._crc.Crc(self._cmd)
            self._pub.Write(self._cmd)

    def _set_targets(self, q, kp, kd):
        with self._cmd_lock:
            for i in range(12):
                m = self._cmd.motor_cmd[i]
                m.q, m.dq, m.kp, m.kd, m.tau = float(q[i]), 0.0, float(kp[i]), float(kd[i]), 0.0

    def _release_sport_mode(self):
        """Hand the motors over from Unitree's built-in controller (it lies down first, then lets go)."""
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        sport, sw = SportClient(), MotionSwitcherClient()
        for c in (sport, sw):
            c.SetTimeout(5.0)
            c.Init()

        def active():
            code, res = sw.CheckMode()
            if code != 0 or res is None:  # a failed query is not "released": both would drive the motors
                raise RuntimeError(f"MotionSwitcher CheckMode failed (code {code})")
            return res.get("name", "")

        for attempt in range(1, 6):
            name = active()
            if not name:
                print("[INFO] built-in controller released; the motors are ours")
                return
            print(f"[INFO] built-in controller '{name}' active: lying down and releasing ({attempt}/5)")
            sport.StandDown()
            time.sleep(3.0)
            sw.ReleaseMode()
            time.sleep(1.0)
        raise RuntimeError(f"built-in controller '{active()}' still active")

    # -------------------------------------------------------------- control
    def enter(self, s, why=""):
        if s != self.state:
            print(f"[STATE] {NAMES[self.state]} -> {NAMES[s]}" + (f"  ({why})" if why else ""), flush=True)
            self.state = s

    def start_move(self, q_from, duration=None, waypoints=None):
        """Joint-space move through waypoints [(seconds, q), ...] (default: straight to the default pose in
        `duration` s), HOLD at the end. q None = limp for that long (Kp 0, Kd --damping_kd): the body settles and the
        next segment starts from wherever the joints are then."""
        wps = waypoints or [(duration, self.default)]
        self.move = (time.monotonic(), [(max(d, 1e-3), None if w is None else np.asarray(w, dtype=np.float64).copy())
                                        for d, w in wps], np.asarray(q_from, dtype=np.float64).copy())
        self.limp_q = self.move[2]
        self.enter(STAND)

    def return_path(self, q_from):
        """B: back to the default pose. A lifted leg (thigh > 0.5 rad from default) folds its knee, swings the thigh
        back, then puts the paw straight down under the shoulder. Moved straight, the Tripod's lifted paw touched the
        floor ahead of the default foothold and stayed there (real run 2026-10-30: FR thigh 0.50 vs 0.80, the other
        legs up to 0.25 rad off), and the policy started from that pose tipped over within 0.4 s (MuJoCo, from that
        pose: 4 / 4 falls; from the default: 0 / 4)."""
        q_from = np.asarray(q_from, dtype=np.float64)
        lifted = [k for k in range(4) if abs(q_from[3 * k + 1] - self.default[3 * k + 1]) > 0.5]
        if not lifted:
            return [(self.a.return_time, self.default)]
        w1, w2 = self.default.copy(), self.default.copy()
        for k in lifted:
            w1[3 * k + 1], w1[3 * k + 2] = q_from[3 * k + 1], CALF_FOLD
            w2[3 * k + 2] = CALF_FOLD
        t = self.a.return_time
        path = [(0.3 * t, w1), (0.4 * t, w2), (0.3 * t, self.default)]
        if self.a.reseat:
            # Even then the stance feet stay where the tripod gait left them (MuJoCo after this path: FR thigh -0.11,
            # FL +0.10, RR +0.11 rad off, and the next A tipped over although 0.14 rad of symmetric knee sag on the
            # first A did not). Lying down unloads the feet; standing up again sets them as on the first stand.
            path += [(2.0, LIE), (1.0, None), (self.a.stand_time, self.default)]
        return path

    def step(self):
        with self._lock:
            msg, msg_t = self._msg, self._msg_t
        now = time.monotonic()
        age = now - msg_t
        q = np.array([msg.motor_state[i].q for i in range(12)])
        dq = np.array([msg.motor_state[i].dq for i in range(12)])
        temp = np.array([msg.motor_state[i].temperature for i in range(12)])
        quat = np.array(msg.imu_state.quaternion, dtype=np.float64)
        gyro = np.array(msg.imu_state.gyroscope, dtype=np.float64)
        buttons, sticks = remote.decode(msg.wireless_remote)
        pressed = self.edges.update(buttons)
        a = self.a
        hot = int(temp.max()) >= a.temp_stop
        self.stats["max_temp"] = max(self.stats["max_temp"], int(temp.max()))
        if temp.max() >= a.temp_warn and not self.warned_hot:
            print(f"[WARN] motor {SDK_JOINTS[int(temp.argmax())]} at {int(temp.max())} C", flush=True)
            self.warned_hot = True

        if "select" in pressed and self.state == ZERO_TORQUE and not self.select_ok:
            self.select_ok = True  # the robot is limp already; this press only proves the key works
            pressed = pressed - {"select"}
            print("[OK] select(非常停止)が読めました。start で立ち上がります。", flush=True)
        if "select" in pressed:
            self.enter(DAMPING, "select")
        if self.state in (STAND, HOLD, POLICY) and age > a.stale_timeout:
            self.enter(DAMPING, f"rt/lowstate {age * 1e3:.0f} ms old")

        cmd = np.zeros(3)
        obs = np.zeros((1, 1), dtype=np.float32)
        kp, kd = self.kp, self.kd
        if self.state == ZERO_TORQUE:
            q_des, kp, kd = q, self.zeros, self.zeros
            if "start" in pressed and not self.select_ok:
                print("[WARN] 先に select を押してください(非常停止のボタンが効くかの確認)。", flush=True)
            elif "start" in pressed:
                self.start_move(q, a.stand_time)
        if self.state == STAND:
            t0, wps, q_prev = self.move
            el, q_des = now - t0, None
            kp, kd = self.stand_kp, self.stand_kd
            for dur, w in wps:
                if el < dur:
                    if w is None:
                        q_des, kp, kd = q, self.zeros, np.full(12, a.damping_kd)
                        self.limp_q = q.copy()
                    else:
                        q_des = q_prev + (el / dur) * (w - q_prev)
                    break
                el, q_prev = el - dur, (self.limp_q if w is None else w)
            if q_des is None:
                q_des = wps[-1][1]
                keys = "down / up while running" if self.switchable else "X hop / Y walk"
                self.enter(HOLD, f"default pose; gait '{self.policy.gait}' ({keys}), A to run")
        if self.state == HOLD:
            q_des, kp, kd = self.default, self.stand_kp, self.stand_kd
            for key, g in ((("X", "hop"), ("Y", "walk")) if not self.switchable else ()):
                if key in pressed and g in self.policy.modes and g != self.policy.gait:
                    self.policy.set_gait(g)
                    print(f"[INFO] gait -> {g}", flush=True)
            if "A" in pressed or (self.switchable and "up" in pressed):  # up: stand up on the hind legs (biped)
                off = np.abs(q - self.default)
                if hot:
                    print(f"[WARN] A refused: motor {SDK_JOINTS[int(temp.argmax())]} at {int(temp.max())} C "
                          f"(>= --temp_stop {a.temp_stop})", flush=True)
                elif off.max() > a.start_tol:
                    print(f"[WARN] A を受け付けません: {SDK_JOINTS[int(off.argmax())]} が既定の姿勢から "
                          f"{off.max():.2f} rad ずれています(上限 {a.start_tol})。select で伏せてから start で"
                          f"立ち直してください。", flush=True)
                else:
                    if self.switchable:
                        self.policy.set_gait("biped")
                    self.policy.reset()
                    self.stop_pending = None
                    self.enter(POLICY, f"A, gait '{self.policy.gait}'")
        if (self.state == POLICY and self.switchable and a.down == "scripted" and self.policy.gait == "biped"
                and ({"B", "down"} & pressed or hot) and fallen(projected_gravity(quat), self.sin_roll) is None):
            # scripted descent (2026-10-31): straight to the default quadruped pose in --down_time s from the
            # measured joints, stand gains. MuJoCo with the measured joints: 36 / 36 without a fall (0.3-0.7 s, 4 gait
            # phases, standing / walking / turning), level on four legs, base at -0.5 m/s at 0.5 s. The policies did
            # not learn to go down (logs/summary_switch.txt). B / heat then lie down and stand up as for Tripod.
            why = "B" if "B" in pressed else ("down" if "down" in pressed else
                                              f"motor {SDK_JOINTS[int(temp.argmax())]} at {int(temp.max())} C")
            path = [(a.down_time, self.default)] + (self.return_path(self.default)[1:] if why != "down" else [])
            self.start_move(q, waypoints=path)
            print(f"[INFO] {why}: scripted descent to four legs ({a.down_time} s)", flush=True)
            q_des, kp, kd = q, self.stand_kp, self.stand_kd
        if {"L1", "R1"} & pressed:  # pitch trim (--pitch_trim): L1 back, R1 forward
            self.trim += -0.5 if "L1" in pressed else 0.5
            print(f"[INFO] pitch trim {self.trim:+.1f} deg", flush=True)
        upright_mode = self.state == POLICY and self.policy.gait == "biped"
        if upright_mode and self.biped_since is None:
            self.biped_since = now
        elif not upright_mode:
            self.biped_since, self.demo_t0 = None, None
        ramp = min(1.0, (now - self.biped_since) / a.trim_ramp) if self.biped_since is not None and a.trim_ramp > 0 else 1.0
        self.policy.pitch_trim_deg = self.trim * ramp if upright_mode else 0.0
        if self.state == POLICY:
            g = projected_gravity(quat)
            out = (q < LOWER - a.joint_margin) | (q > UPPER + a.joint_margin)
            fall = fallen(g, self.sin_roll)
            level = g[2] < -0.95  # body within 18 deg of level: on four legs
            self.level_since = (self.level_since or now) if level else None
            stop_now = False
            if self.switchable and fall is None and not out.any():
                if ("B" in pressed or hot) and self.stop_pending is None:
                    why = "B" if "B" in pressed else f"motor {SDK_JOINTS[int(temp.argmax())]} at {int(temp.max())} C"
                    self.stop_pending = now
                    self.policy.set_gait("quad")
                    print(f"[INFO] {why}: going down to four legs before stopping", flush=True)
                elif "down" in pressed and self.policy.gait != "quad":
                    self.policy.set_gait("quad")
                    print("[INFO] mode -> quad (down)", flush=True)
                elif "up" in pressed and self.policy.gait != "biped" and self.stop_pending is None:
                    if hot:
                        print(f"[WARN] up refused: motor {SDK_JOINTS[int(temp.argmax())]} at {int(temp.max())} C", flush=True)
                    else:
                        self.policy.set_gait("biped")
                        print("[INFO] mode -> biped (up)", flush=True)
                if self.stop_pending is not None:
                    if self.level_since is not None and now - self.level_since >= 0.5:
                        stop_now = True
                    elif now - self.stop_pending > 8.0:
                        fall = "not level 8 s after the stop"
            elif not self.switchable:
                stop_now = "B" in pressed or hot
            if fall is not None:
                self.enter(DAMPING, f"fallen: {fall}")
            elif out.any():
                self.enter(DAMPING, f"joint out of range: {[SDK_JOINTS[i] for i in np.flatnonzero(out)]}")
            elif stop_now:
                self.stop_pending = None
                self.start_move(self.q_des, waypoints=self.return_path(self.q_des))
                print("[INFO] policy stopped: back to the default pose", flush=True)
                q_des, kp, kd = self.q_des, self.stand_kp, self.stand_kd
            else:
                cmd = np.array(remote.sticks_to_command(sticks, a.max_fwd, a.max_back, a.max_vy, a.max_wz))
                if self.policy.gait == "biped" and "Y" in pressed and self.demo_t0 is None:
                    self.demo_t0 = now
                    print(f"[INFO] demo: {DEMO_S:.0f} s (any stick or Y stops it)", flush=True)
                elif self.demo_t0 is not None and (np.any(cmd != 0.0) or "Y" in pressed or now - self.demo_t0 > DEMO_S):
                    self.demo_t0 = None
                    print("[INFO] demo end", flush=True)
                if self.demo_t0 is not None:
                    cmd = demo_command(now - self.demo_t0, a.demo_scale)
                if self.policy.gait == "quad":
                    cmd[:] = 0.0  # trained: the quadruped mode stands still
                t_inf = time.perf_counter()
                obs = self.policy.observe(quat, gyro, cmd, q[self.sdk_to_pol], dq[self.sdk_to_pol])
                target = self.policy.act(obs)
                raw = self.policy.last_action
                self.stats["max_abs_action"] = max(self.stats["max_abs_action"], float(np.abs(raw).max()))
                if np.abs(raw).max() > a.action_clip:
                    self.stats["action_clips"] += 1
                    self.policy.last_action = np.clip(raw, -a.action_clip, a.action_clip)
                    target = self.policy.default + self.policy.action_scale * self.policy.last_action
                self.stats["infer_ms_max"] = max(self.stats["infer_ms_max"], (time.perf_counter() - t_inf) * 1e3)
                q_des = target[self.pol_to_sdk]
                d = np.clip(q_des - self.q_des, -a.max_step, a.max_step)
                self.stats["step_clamps"] += int(np.any(d != q_des - self.q_des))
                q_des = self.q_des + d
        if self.state == DAMPING:
            q_des, kp, kd = q, self.zeros, np.full(12, a.damping_kd)
            if "start" in pressed:  # stand up again from lying, as after the release
                up = -projected_gravity(quat)[2]
                if up > np.cos(np.radians(20.0)) and np.abs(dq).max() < 0.5:
                    self.start_move(q, a.stand_time)
                else:
                    print("[WARN] 伏せて止まってから start を押してください(胴体が傾いているか、まだ動いています)。", flush=True)

        clamped = np.clip(q_des, LOWER - a.target_margin, UPPER + a.target_margin)
        if self.state not in (ZERO_TORQUE, DAMPING):
            self.stats["limit_clamps"] += int(np.any(clamped != q_des))
        self.q_des = clamped
        self._set_targets(self.q_des, kp, kd)

        L = self.log
        L["t"].append(now)
        L["state"].append(self.state)
        L["gait"].append(self.policy.mode_index)  # Tripod: 0 hop / 1 walk; switch: 0 quad / 1 biped
        L["age"].append(age)
        L["cmd"].append(cmd)
        L["obs"].append(obs.reshape(-1) if self.state == POLICY else np.zeros(0))
        L["action"].append(self.policy.last_action.copy())
        L["q_des"].append(self.q_des.copy())
        L["q"].append(q)
        L["dq"].append(dq)
        L["tau"].append([msg.motor_state[i].tau_est for i in range(12)])
        L["quat"].append(quat)
        L["gyro"].append(gyro)
        L["acc"].append(list(msg.imu_state.accelerometer))
        L["temp"].append(temp)
        L["foot_force"].append(list(msg.foot_force))  # FR FL RR RL
        L["keys"].append(sum(1 << remote.BUTTONS.index(b) for b in buttons))
        L["sticks"].append([sticks["lx"], sticks["ly"], sticks["rx"], sticks["ry"]])

    def status(self):
        with self._lock:
            n = self._msg_n
        c = self.log["cmd"][-1] if self.log["cmd"] else np.zeros(3)
        sk = self.log["sticks"][-1] if self.log["sticks"] else [0.0] * 4
        T = self.log["temp"][-1] if self.log["temp"] else np.zeros(12)
        return (f"[{NAMES[self.state]:11s}] gait {self.policy.gait} trim {self.trim:+.1f} ({self.policy.pitch_trim_deg:+.1f}){' DEMO' if self.demo_t0 is not None else ''} sticks L {sk[0]:+.1f} {sk[1]:+.1f} R {sk[2]:+.1f} "
                f"cmd {c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f} | "
                f"lowstate {n} | infer {self.stats['infer_ms_max']:.1f} ms | overruns {self.stats['overruns']} | "
                f"clamps step {self.stats['step_clamps']} limit {self.stats['limit_clamps']} | "
                f"max temp {int(np.max(T))} C ({SDK_JOINTS[int(np.argmax(T))][:-6]})")

    def run(self):
        period = self.policy.dt
        nxt, nxt_status = time.perf_counter(), time.monotonic()
        while not self.stop.is_set():
            self.step()
            if time.monotonic() >= nxt_status:
                print(self.status(), flush=True)
                nxt_status += 1.0
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                self.stats["overruns"] += 1
                nxt = time.perf_counter()

    def shutdown(self):
        """Leave the motors in damping, not in the last command."""
        if getattr(self, "_writer", None) is None:
            return
        self.enter(DAMPING, "shutting down")
        with self._lock:
            msg = self._msg
        q = np.array([msg.motor_state[i].q for i in range(12)])
        self._set_targets(q, self.zeros, np.full(12, self.a.damping_kd))
        time.sleep(0.5)
        self._writer.Wait(1.0)

    def save(self, path: Path):
        if not self.log["t"]:
            print("[INFO] no control step ran: no log")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: (np.asarray(v) if k != "obs" else np.array(v, dtype=object)) for k, v in self.log.items()}
        data["t"] = data["t"] - data["t"][0]
        meta = {"mode": self.a.mode, "export_dir": str(self.a.export_dir), "args": vars(self.a), "sdk_joints": SDK_JOINTS,
                "policy_joints": self.policy.joint_names, "states": NAMES, "buttons": remote.BUTTONS,
                "gaits": self.policy.modes, "stats": self.stats}
        data["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **data)
        print(f"[INFO] saved {len(self.log['t'])} steps to {path}")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=("sim", "real"), required=True)
    p.add_argument("--iface", default=None, help="(real) network interface wired to the Go2")
    p.add_argument("--export_dir", required=True, help="directory with policy.onnx and deploy.json")
    p.add_argument("--down", choices=("scripted", "policy"), default="scripted",
                   help="switch policies: down / B / heat in the biped mode = scripted move to four legs (default) or "
                        "the policy's quadruped mode")
    p.add_argument("--down_time", type=float, default=0.5, help="duration of the scripted descent [s]")
    p.add_argument("--gait", default=None, help="initial gait mode (Tripod: hop / walk)")
    # Tripod eval commands (scripts/eval_tripod.py): hop fwd 0.3 / back 0.2 / turn 0.5, bag walk 0.2 / 0.15 / 0.4;
    # training range vx -0.3..0.5, vy +-0.2, wz +-0.6. Start low on the robot.
    p.add_argument("--pitch_trim", type=float, default=None,
                   help="[deg] tilt the observed gravity about the body y axis; L1 / R1 change it by 0.5 while running. "
                        "Upright on the hind legs, the policy walks forward when it is too high (MuJoCo: +3 -> +0.31 m/s, "
                        "-3 -> -0.24 m/s with no command); the first biped run on a carpet walked forward. Default: the value set with "
                        "L1 / R1 on the second run, -5 (2026-10-31, carpet, HoldBall-Switch c1), for biped policies on the robot; "
                        "0 otherwise (simulator, Tripod)")
    p.add_argument("--trim_ramp", type=float, default=3.0, help="[s] the pitch trim grows from 0 over this after standing up")
    p.add_argument("--demo_scale", type=float, default=1.0, help="speed scale of the Y demo (turn 0.6, fwd / back 0.3, side 0.25)")
    p.add_argument("--max_fwd", type=float, default=None)
    p.add_argument("--max_back", type=float, default=None)
    p.add_argument("--max_vy", type=float, default=None)
    p.add_argument("--max_wz", type=float, default=None)
    p.add_argument("--stand_time", type=float, default=3.0)
    p.add_argument("--return_time", type=float, default=2.0, help="[s] B: back to the default pose (3 phases)")
    p.add_argument("--no_reseat", dest="reseat", action="store_false",
                   help="B: do not lie down and stand up again after the paw is down (feet stay where they were)")
    p.add_argument("--start_tol", type=float, default=0.2,
                   help="[rad] A only when every joint is this close to the default pose (training starts within "
                        "0.05; real 2026-10-30: 0.14 (knee sag under load) ran, 0.30 tipped over)")
    p.add_argument("--stand_kp", type=float, default=60.0, help="unitree_sdk2 go2_stand_example")
    p.add_argument("--stand_kd", type=float, default=5.0, help="unitree_sdk2 go2_stand_example")
    p.add_argument("--damping_kd", type=float, default=2.0)
    p.add_argument("--max_tilt", type=float, default=50.0,
                   help="[deg] sideways roll that counts as a fall (the Tripod / biped bodies stay within ~10 deg)")
    p.add_argument("--max_step", type=float, default=1.0,
                   help="[rad] largest target change per policy step. Walking needs up to 0.70 (HoldBall knees, DDS run "
                        "2026-10-31; 0.5 clipped 9 %% of the steps), the biped stand-up up to 2 in its first steps")
    p.add_argument("--action_clip", type=float, default=10.0, help="raw action bound (target +-2.5 rad at scale 0.25)")
    p.add_argument("--joint_margin", type=float, default=0.1, help="[rad] measured joint this far out of range: damping")
    p.add_argument("--target_margin", type=float, default=0.5,
                   help="[rad] targets may lie this far outside the joint limits (TripodCool c2: 0.36 seen)")
    p.add_argument("--stale_timeout", type=float, default=0.1)
    p.add_argument("--temp_warn", type=int, default=60, help="[C] hind hips can reach about 60 C just standing on four legs")
    p.add_argument("--temp_stop", type=int, default=70, help="[C] Unitree's own protection limit is not known yet")
    p.add_argument("--log", default=None)
    p.add_argument("--yes", action="store_true", help="(real) skip the confirmation")
    a = p.parse_args()
    # full-stick speeds: biped (switch) policies the training range while holding an object (x -0.8..1.2, y +-0.5,
    # z +-1.5 times cmd_scale 0.8, runs/holdball_switch.status), forward as backward (0.64; the robot fell going from
    # 0.96 forward to -0.6 back within a second, 2026-10-31); others the Tripod values checked in the simulator
    biped = "biped" in json.loads((Path(a.export_dir) / "deploy.json").read_text()).get("gait_modes", {})
    for k, vb, vt in [("max_fwd", 0.64, 0.3), ("max_back", 0.64, 0.2), ("max_vy", 0.4, 0.15), ("max_wz", 1.2, 0.5)]:
        if getattr(a, k) is None:
            setattr(a, k, vb if biped else vt)
    if a.mode == "real":
        if not a.iface:
            p.error("--iface is required with --mode real")
        print("\n  *** REAL ROBOT ***\n  The built-in controller (if active) lies the Go2 down and goes limp.\n"
              "  Robot lying on the floor, 2 m clear around it, hands away from the legs.\n"
              "  start = stand, X/Y = hop/walk, A = run, B = stop, select = damping (emergency).\n"
              f"  Full stick: fwd {a.max_fwd} back {a.max_back} side {a.max_vy} m/s, turn {a.max_wz} rad/s\n")
        if not a.yes and input("  Type 'yes' to continue: ").strip() != "yes":
            sys.exit("aborted")
    log = Path(a.log) if a.log else Path(__file__).resolve().parent / "real_logs" / time.strftime(f"{a.mode}_%Y%m%d_%H%M%S.npz")
    log.parent.mkdir(parents=True, exist_ok=True)
    txt = open(log.with_suffix(".txt"), "w")  # the screen output (and any traceback) is kept next to the .npz
    sys.stdout, sys.stderr = Tee(sys.stdout, txt), Tee(sys.stderr, txt)

    def _sig(signum, _f):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGHUP, _sig)
    c = Controller(a)
    print(f"[INFO] policy {a.export_dir}: gaits {list(c.policy.modes)}, Kp {c.policy.kp} Kd {c.policy.kd}")
    try:
        c.connect()
        print("[INFO] ready: まず select を押して(非常停止の確認)、次に start で立ち上がります", flush=True)
        c.run()
    except KeyboardInterrupt as e:
        print(f"[INFO] interrupted ({e or 'Ctrl-C'})")
    finally:
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)  # a second Ctrl-C must not skip the damping and the log
        print("[INFO] stopping: damping, saving the log ...", flush=True)
        c.shutdown()
        c.save(log)


if __name__ == "__main__":
    main()
