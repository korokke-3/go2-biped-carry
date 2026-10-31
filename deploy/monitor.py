"""Read-only check of the robot's rt/lowstate before the first deploy/go2_real.py run. Sends nothing: the built-in
controller stays in charge. Guides the operator in Japanese, step by step, detects each step from the data itself
and records everything for later analysis:
  deploy/real_logs/monitor_<time>.npz   lowstate at 50 Hz (q, dq, tau_est, IMU, temperatures, foot force, remote)
  deploy/real_logs/monitor_<time>.json  the steps: detected or not, when, and the values that decided it

  ~/go2_sim2real/.venv/bin/python deploy/monitor.py --iface eno1     # robot (DDS domain 0)
  ~/go2_sim2real/.venv/bin/python deploy/monitor.py --sim            # a simulator on DDS domain 1 / loopback
Steps: 5 s still (message rate, posture, IMU, temperatures). The remote is checked later, inside go2_real.py once the
built-in controller is released: pressing buttons now would drive the built-in controller.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import remote  # noqa: E402
from policy import projected_gravity  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--iface", default=None)
p.add_argument("--sim", action="store_true")
p.add_argument("--timeout", type=float, default=30.0)
p.add_argument("--log", default=None, help="path prefix; default deploy/real_logs/monitor_<time>")
a = p.parse_args()
if not a.sim and not a.iface:
    p.error("--iface (robot) or --sim")
prefix = Path(a.log) if a.log else Path(__file__).resolve().parent / "real_logs" / time.strftime("monitor_%Y%m%d_%H%M%S")
prefix.parent.mkdir(parents=True, exist_ok=True)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_  # noqa: E402

KEYS = ("t", "q", "dq", "tau", "quat", "gyro", "acc", "temp", "foot_force", "keys", "sticks")
rec = {k: [] for k in KEYS}
lock = threading.Lock()
box = {"msg": None, "n": 0, "t_last": 0.0}
t0 = time.monotonic()


def on_state(msg):
    now = time.monotonic()
    with lock:
        box["msg"], box["n"] = msg, box["n"] + 1
        if now - box["t_last"] < 0.02:  # record at 50 Hz
            return
        box["t_last"] = now
        b, st = remote.decode(msg.wireless_remote)
        rec["t"].append(now - t0)
        rec["q"].append([msg.motor_state[i].q for i in range(12)])
        rec["dq"].append([msg.motor_state[i].dq for i in range(12)])
        rec["tau"].append([msg.motor_state[i].tau_est for i in range(12)])
        rec["quat"].append(list(msg.imu_state.quaternion))
        rec["gyro"].append(list(msg.imu_state.gyroscope))
        rec["acc"].append(list(msg.imu_state.accelerometer))
        rec["temp"].append([msg.motor_state[i].temperature for i in range(12)])
        rec["foot_force"].append(list(msg.foot_force))
        rec["keys"].append(sum(1 << remote.BUTTONS.index(x) for x in b))
        rec["sticks"].append([st["lx"], st["ly"], st["rx"], st["ry"]])


def latest():
    with lock:
        return box["msg"]


steps = []


def wait_for(name, guide, test, timeout=None):
    """Show the guide, poll the latest message until test(msg) returns a (passed, values) with passed True."""
    print(f"\n▶ {guide}", flush=True)
    t_start = time.monotonic()
    vals = None
    while time.monotonic() - t_start < (timeout or a.timeout):
        ok, vals = test(latest())
        if ok:
            print("   OK", flush=True)
            steps.append({"step": name, "detected": True, "t": time.monotonic() - t0, "values": vals})
            return True
        time.sleep(0.02)
    print("   確認できませんでした(記録して次へ進みます)", flush=True)
    steps.append({"step": name, "detected": False, "t": time.monotonic() - t0, "values": vals})
    return False


def button(b):
    return lambda m: (b in remote.decode(m.wireless_remote)[0], None)


def stick(axis, sign):
    def test(m):
        v = remote.decode(m.wireless_remote)[1][axis]
        return sign * v > 0.5, {axis: v}
    return test


def released(m):
    b, st = remote.decode(m.wireless_remote)
    return (not b and all(abs(v) < 0.1 for v in st.values())), None


def save():
    with lock:
        data = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    data["meta"] = np.array(json.dumps({"iface": a.iface, "sim": a.sim, "buttons": remote.BUTTONS,
                                        "sdk_joints": [f"{l}_{p}" for l in ("FR", "FL", "RR", "RL") for p in ("hip", "thigh", "calf")]}))
    np.savez_compressed(f"{prefix}.npz", **data)
    Path(f"{prefix}.json").write_text(json.dumps(steps, indent=1, ensure_ascii=False, default=float))
    print(f"\n記録しました: {prefix}.npz と .json")


if not a.sim:
    from go2_real import check_iface  # noqa: E402
    check_iface(a.iface)
ChannelFactoryInitialize(1 if a.sim else 0, "lo" if a.sim else a.iface)
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_state, 10)
try:
    while latest() is None:
        if time.monotonic() - t0 > 5.0:
            steps.append({"step": "receive", "detected": False})
            print("ロボットからのデータが届きません。ケーブル、`nmcli con up go2`、ファイアウォールを確認してください。")
            sys.exit(1)
        time.sleep(0.05)
    print("ロボットからのデータが届いています。案内に従って操作してください(Ctrl-C で中止、そこまでの記録は残ります)。")

    # 1. still: rate, posture, IMU at rest
    print("\n▶ ロボットに触らず、そのまま 5 秒待ってください。", flush=True)
    n0, ts = box["n"], time.monotonic()
    time.sleep(5.0)
    m = latest()
    g = projected_gravity(np.array(m.imu_state.quaternion, dtype=np.float64))
    still = {"rate_hz": (box["n"] - n0) / (time.monotonic() - ts), "q": [m.motor_state[i].q for i in range(12)],
             "gravity_body": g.tolist(), "temp": [m.motor_state[i].temperature for i in range(12)]}
    steps.append({"step": "still", "detected": True, "t": time.monotonic() - t0, "values": still})
    print(f"   受信 {still['rate_hz']:.0f} 回/秒、モーターの最高温度 {max(still['temp'])} °C")

    lying = still["q"][2] < -2.3 and still["q"][5] < -2.3  # front calves folded (Unitree's lie-down pose)
    level = still["gravity_body"][2] < -0.95
    print(f"   姿勢: {'伏せている' if lying else '伏せていない(立っているかも)'}、胴体 {'水平' if level else '傾いている'}")
    print("\n確認は以上です。")
except KeyboardInterrupt:
    print("\n中止しました。")
finally:
    save()
    sys.stdout.flush()
    import os
    os._exit(0)  # DDS threads would keep the process alive
