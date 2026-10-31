"""Choose the waiting ("ready") arm pose without a policy: the base is FIXED upright in the air (fix_base), the front
legs are PD-held in candidate poses, objects are handed over, then pushed with a fictitious inertial force.

Candidate pose = (front hip adduction, thigh, calf): top view "ha" (forearm tips closer than the elbows), side view
"V" (thigh + calf). Per pose x shape x test, one env (shape = env % 8, as in the Hold tasks):
  test "keep"   : no push; is the object still on the arms 1.5 s after the hand-over?
  test fwd/back/left/right : after 1.5 s, a force m * a(t) in that direction (heading frame), a ramps 0 -> 6 m/s^2
                  over 6 s; the acceleration at which the object leaves is recorded (= tolerated body acceleration)
  test sway_fwd / sway_lat : m * A(t) * sin(2 pi t / 0.4 s) (the gait period), A ramps 0 -> 6 m/s^2 over 6 s
"Leaves" = object more than 0.25 m from the forearm centre or below the forearm tops by 0.15 m.

  python scripts/diagnostics/diag_ready_pose.py [--drop 0.10] [--mass 2.0]
Writes logs/diag_ready_pose_<drop>.json and prints the table (best poses first).
"""
import argparse
import itertools
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--drop", type=float, default=0.0, help="hand-over drop height [m]")
p.add_argument("--mass", type=float, default=2.0)
p.add_argument("--hips", default="0,0.15,0.3,0.45", help="front hip adduction (inwards) [rad]")
p.add_argument("--thighs", default="1.0,1.3,1.6")
p.add_argument("--calves", default="-1.3,-1.6,-1.9,-2.2")
p.add_argument("--task", default="Unitree-Go2-Biped-HoldRef",
               help="HoldRef: the 8 old boxes (env %% 8); HoldV4: its fixed evaluation set (8 boxes + spheres 0.30/0.36/0.44)")
p.add_argument("--tests", default="keep,fwd,back,left,right,sway_fwd,sway_lat")
p.add_argument("--amax", type=float, default=6.0, help="push acceleration reached after the 6 s ramp [m/s^2]")
p.add_argument("--poses", default=None, help="explicit poses 'hip:thigh:calf;...' instead of the grid")
p.add_argument("--reps", type=int, default=1, help="repeat every pose x shape x test this many times (with --noise)")
p.add_argument("--noise", action="store_true", help="training placement noise (0-3 cm fwd, +-3 cm side, +-10 deg yaw)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import go2_carry  # noqa: E402,F401
from go2_carry import box_mdp as hmdp  # noqa: E402
from go2_carry import mdp as bmdp  # noqa: E402

TESTS = args.tests.split(",")
V4 = "HoldV4" in args.task
K = len(hmdp.V4_REF_BOXES) + len(hmdp.V4_REF_SPHERES) if V4 else len(hmdp.BOX_SHAPES)
# envs per (pose, test) block: HoldRef 8 (shape = env % 8); HoldV4 96 = lcm(4 x 8 boxes, 4 x 3 spheres) so that every
# block holds every reference object (asset = env % 4, box ref = (env // 4) % 8, sphere ref = (env // 4) % 3)
BLK = 96 if V4 else K
poses = list(itertools.product([float(v) for v in args.hips.split(",")], [float(v) for v in args.thighs.split(",")],
                               [float(v) for v in args.calves.split(",")]))
if args.poses:
    poses = [tuple(float(v) for v in q.split(":")) for q in args.poses.split(";")]
base_poses = poses
poses = poses * args.reps  # repeats are aggregated per pose at the end
P, T = len(poses), len(TESTS)
N = P * T * BLK  # env = (pose * T + test) * BLK + j
TASK = args.task
cfg = load_cfg_from_registry(TASK, "play_env_cfg_entry_point")
cfg.scene.num_envs = N
cfg.scene.env_spacing = 1.5
cfg.scene.robot.spawn.fix_base = True
cfg.events.reset_robot.params.update({"biped_prob": 1.0, "mid_prob": 0.0})
cfg.events.push_robot = None
cfg.events.object_mass.params["fixed_mass"] = args.mass
cfg.events.object_handover.params.update({"p_object": 1.0, "fixed_level": hmdp.DISABLED_LEVEL, "fixed_ready_s": 0.3,
                                           "fixed_drop": args.drop, "fixed_release_s": 0.0, "p_remove": 0.0,
                                           "place_noise": (0.03, 0.03, math.radians(10.0)) if args.noise else (0.0, 0.0, 0.0)})
cfg.terminations.object_dropped = None
cfg.terminations.body_contact = None
cfg.episode_length_s = 20.0
cfg.curriculum = None
if V4:
    cfg.events.object_sizes.params["fixed_refs"] = True
hmdp.Box.assist = 0.0
env = gym.make(TASK, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
obj = u.scene["object"]
dev = u.device
env.reset()

# joint targets per env: hind legs / everything else at the upright reset pose, front legs at the candidate pose
q_up = robot.data.joint_pos.clone()
names = robot.joint_names
ids = {n: names.index(n) for n in names}
pose_of = torch.arange(N, device=dev) // (T * BLK)
test_of = (torch.arange(N, device=dev) // BLK) % T
shape_of = hmdp.Box.ref_id.to(dev) if V4 else torch.arange(N, device=dev) % K
tgt = q_up.clone()
for i, (h, th, ca) in enumerate(poses):
    m = pose_of == i
    tgt[m, ids["FL_hip_joint"]] = -h  # FL: negative = inwards (box_mdp.HOLD_JOINT_POS uses -0.15 / +0.15)
    tgt[m, ids["FR_hip_joint"]] = h
    tgt[m, ids["FL_thigh_joint"]] = th
    tgt[m, ids["FR_thigh_joint"]] = th
    tgt[m, ids["FL_calf_joint"]] = ca
    tgt[m, ids["FR_calf_joint"]] = ca
scale = u.action_manager.get_term("JointPositionAction")._scale
act = (tgt - robot.data.default_joint_pos) / scale
mass = bmdp.object_mass(u)[:, 0]

left = torch.full((N,), float("nan"), device=dev)      # time [s] the object left the arms
# load share carried by the upper arms (thighs) vs forearms (calves) while held, 0.5-1.5 s after placement (keep test)
f_thigh = torch.zeros(N, device=dev); f_calf = torch.zeros(N, device=dev)
a_left = torch.full((N,), float("nan"), device=dev)    # push acceleration amplitude at that time
t_place = torch.full((N,), float("nan"), device=dev)
geo = {}
dt = u.step_dt
steps = int((0.3 + 1.5 + 6.5) / dt)
for k in range(steps):
    env.step(act)
    t = (k + 1) * dt
    placed = hmdp.Box.placed
    t_place = torch.where(torch.isnan(t_place) & placed, torch.full_like(t_place, t), t_place)
    if k == int(0.25 / dt):  # arm geometry before the hand-over (pose settled)
        b = robot.find_bodies(["FL_calf", "FL_foot", "FR_calf", "FR_foot", "FL_thigh", "FR_thigh"], preserve_order=True)[0]
        bp = robot.data.body_pos_w[:, b] - robot.data.root_pos_w[:, None]
        fwd, lft = bmdp._heading_frame(u)
        proj = lambda v: torch.stack([(v * fwd[:, None]).sum(-1), (v * lft[:, None]).sum(-1), v[..., 2]], -1)  # noqa: E731
        bh = proj(bp)  # (N, 6, 3): FL_calf (knee/elbow), FL_foot (tip), FR_calf, FR_foot, FL_thigh, FR_thigh
        for i in range(P):
            e = (pose_of == i).nonzero()[0, 0]
            tip_gap = float((bh[e, 1, 1] - bh[e, 3, 1]).abs())
            elbow_gap = float((bh[e, 0, 1] - bh[e, 2, 1]).abs())
            fore = bh[e, 1] - bh[e, 0]
            tilt = math.degrees(math.atan2(float(fore[2]), float(fore[0])))  # forearm tip up (+) / down (-)
            geo[i] = {"tip_gap_m": round(tip_gap, 3), "elbow_gap_m": round(elbow_gap, 3), "forearm_tilt_deg": round(tilt, 1),
                      "elbow_height_m": round(float(bh[e, 0, 2]), 3), "tip_fwd_m": round(float(bh[e, 1, 0]), 3)}
    # push: fictitious inertial force in the heading frame (body accelerating with +a pushes the object with -m a)
    since = t - (t_place + 1.5)
    ramp = (since / 6.0).clamp(0.0, 1.0) * args.amax
    ramp = torch.where(torch.isnan(ramp), torch.zeros_like(ramp), ramp)
    fwd, lft = bmdp._heading_frame(u)
    sway = torch.sin(torch.tensor(2 * math.pi * t / 0.4, device=dev))
    dir_w = torch.zeros(N, 3, device=dev)
    amp = torch.zeros(N, device=dev)
    for j, name in enumerate(TESTS):
        m = test_of == j
        if name == "keep":
            continue
        d = {"fwd": -fwd, "back": fwd, "left": -lft, "right": lft, "sway_fwd": fwd * sway, "sway_lat": lft * sway}[name]
        dir_w[m] = d[m]
        amp[m] = ramp[m]
    f = torch.zeros(N, 1, 3, device=dev)
    f[:, 0] = dir_w * (mass * amp)[:, None] * placed.float()[:, None]
    obj.set_external_force_and_torque(f, torch.zeros_like(f), is_global=True)
    if "object_contact" in u.scene.sensors:
        fm = u.scene.sensors["object_contact"].data.force_matrix_w.norm(dim=-1).reshape(N, -1)  # FL_calf FR_calf FL_thigh FR_thigh
        win = (t - t_place > 0.5) & (t - t_place < 1.5) & torch.isnan(left)
        win = torch.where(torch.isnan(t_place), torch.zeros_like(win), win)
        f_calf += win.float() * fm[:, :2].sum(dim=1)
        f_thigh += win.float() * fm[:, 2:].sum(dim=1)
    # left the arms?
    c, top = bmdp.forearm_center_w(u)
    off = (obj.data.root_pos_w - c)[:, :2].norm(dim=1) > 0.25
    low = obj.data.root_pos_w[:, 2] < top - 0.15
    gone = (off | low) & ~torch.isnan(t_place) & torch.isnan(left)
    left[gone] = t
    a_left[gone] = amp[gone]

res = []
nb = len(base_poses)
for i, (h, th, ca) in enumerate(base_poses):
    row = {"hip_in": h, "thigh": th, "calf": ca, "reps": args.reps, **geo.get(i, {})}
    same = (pose_of % nb) == i
    for j, name in enumerate(TESTS):
        m = same & (test_of == j)
        kept_at_push = torch.isnan(left[m]) | (left[m] > t_place[m] + 1.5)
        sh = shape_of[m]
        if name == "keep":
            row["keep_per_shape"] = [round(float(kept_at_push[sh == q].float().mean()), 3) for q in range(K)]
            share = f_thigh[m] / (f_thigh[m] + f_calf[m]).clamp(min=1e-6)
            row["thigh_share_per_shape"] = [round(float(share[sh == q].mean()), 2) for q in range(K)]
        else:
            a = torch.where(torch.isnan(a_left[m]), torch.full_like(a_left[m], args.amax), a_left[m])
            a = torch.where(kept_at_push, a, torch.zeros_like(a))  # lost before the push -> tolerates 0
            kept = kept_at_push
            row[f"a_{name}_per_shape"] = [round(float(a[(sh == q) & kept].mean()), 2) if ((sh == q) & kept).any() else 0.0
                                          for q in range(K)]  # mean tolerated acceleration among objects still held
    keep = torch.tensor(row["keep_per_shape"])
    row["keep_all"] = round(float(keep.mean()), 3)
    row["keep_wide(0-2)"] = round(float(keep[:3].mean()), 3)
    row["keep_narrow(3-7)"] = round(float(keep[3:8].mean()), 3)
    if V4:
        row["keep_spheres"] = [round(float(v), 3) for v in keep[8:].tolist()]
    for name in TESTS[1:]:
        row[f"a_{name}_mean"] = round(sum(row[f"a_{name}_per_shape"]) / K, 2)
    res.append(row)
res.sort(key=lambda r: (r["keep_all"], min(r[f"a_{n}_mean"] for n in TESTS[1:])), reverse=True)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", f"diag_ready_pose{'_v4' if V4 else ''}_drop{args.drop:.2f}{'_r%d' % args.reps if args.reps > 1 else ''}{'_a%g' % args.amax if args.amax != 6.0 else ''}.json")
with open(out, "w") as fh:
    json.dump(res, fh, indent=1)
print(f"{'hip':>5} {'thigh':>5} {'calf':>5} | tip/elbow gap  tilt  | keep all/wide/narrow | a fwd back left right swayF swayL")
for r in res:
    print(f"{r['hip_in']:5.2f} {r['thigh']:5.2f} {r['calf']:5.2f} | {r.get('tip_gap_m', -1):.2f}/{r.get('elbow_gap_m', -1):.2f} "
          f"{r.get('forearm_tilt_deg', 0):5.1f} | {r['keep_all']:.2f} {r['keep_wide(0-2)']:.2f} {r['keep_narrow(3-7)']:.2f} | "
          + " ".join(f"{r[f'a_{n}_mean']:4.1f}" for n in TESTS[1:]) + (f" | spheres {r['keep_spheres']}" if V4 else ""))
print("saved", out)
sys.stdout.flush()
env.close()
app.close()
