#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/future-seed-posttrain

R10_PID_FILE="runs/run_hotpot_qafter_stabilized_len4096_round10_lstart10_s012.pid"
R11_SCRIPT="run_hotpot_qafter_stabilized_len4096_round11_grid_lstart10_12_alpha_m2_m3_s012.sh"
R11_LOG="runs/run_hotpot_qafter_stabilized_len4096_round11_grid_lstart10_12_alpha_m2_m3_s012.log"

if [[ -f "$R10_PID_FILE" ]]; then
  R10_PID="$(cat "$R10_PID_FILE" || true)"
  if [[ -n "${R10_PID}" ]]; then
    echo "Waiting for R10 pid=${R10_PID} ..."
    while kill -0 "${R10_PID}" 2>/dev/null; do
      sleep 20
    done
    echo "R10 finished; starting R11 grid."
  fi
fi

bash "${R11_SCRIPT}" > "${R11_LOG}" 2>&1

