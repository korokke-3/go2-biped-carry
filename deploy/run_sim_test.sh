#!/bin/bash
# Run deploy/go2_real.py against the DDS simulator deploy/sim_dds.py with a scripted remote, end to end.
#   deploy/run_sim_test.sh <export_dir> <out_prefix> [sim_dds options...] [-- go2_real options...]
# -> <out_prefix>_sim.json (tracking per segment), <out_prefix>_ctrl.npz (controller log), <out_prefix>.log
set -u
EXP=$1; OUT=$2; shift 2
SIM_ARGS=(); CTRL_ARGS=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; CTRL_ARGS=("$@"); break; fi
  SIM_ARGS+=("$1"); shift
done
PY=~/go2_sim2real/.venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$(dirname "$OUT")"
"$PY" "$HERE/sim_dds.py" --out "${OUT}_sim.json" "${SIM_ARGS[@]}" > "${OUT}.log" 2>&1 &
SIM=$!
sleep 2
"$PY" "$HERE/go2_real.py" --mode sim --export_dir "$EXP" --log "${OUT}_ctrl.npz" "${CTRL_ARGS[@]}" > "${OUT}_ctrl.log" 2>&1 &
CTRL=$!
wait $SIM
kill -TERM $CTRL 2>/dev/null
wait $CTRL
cat "${OUT}.log"
grep -E "STATE|WARN|INFO\] (policy stopped|gait|saved)" "${OUT}_ctrl.log"
tail -1 "${OUT}_ctrl.log"
