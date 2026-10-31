"""Thin wrapper: register go2_carry tasks, then run unitree_rl_lab RSL-RL play script.

  python scripts/play.py --headless --task Unitree-Go2-Biped-Stand --video --video_length 500 --enable_cameras
Logs go to ./logs/rsl_rl/<task>/<timestamp>.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_SCRIPTS = os.path.expanduser(os.environ.get("UNITREE_RL_LAB_SCRIPTS", "~/go2_sim2real/unitree_rl_lab/scripts/rsl_rl"))
sys.path.insert(0, ROOT)
sys.path.insert(0, URL_SCRIPTS)
sys.path.insert(0, os.path.dirname(URL_SCRIPTS))

import go2_carry  # noqa: E402,F401  (gym registration)

script = os.path.join(URL_SCRIPTS, os.path.basename(__file__))
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
