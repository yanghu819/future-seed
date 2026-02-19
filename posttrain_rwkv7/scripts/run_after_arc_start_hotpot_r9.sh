#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/future-seed-posttrain

ARC_PID_FILE="runs/run_arc_optionsfirst_stabilized_round4_sched_linear.pid"
HOTPOT_SCRIPT="run_hotpot_qafter_stabilized_len4096_round9_sched_linear_s012.sh"
HOTPOT_LOG="runs/run_hotpot_qafter_stabilized_len4096_round9_sched_linear_s012.log"

if [[ -f "$ARC_PID_FILE" ]]; then
  ARC_PID="$(cat "$ARC_PID_FILE" || true)"
  if [[ -n "${ARC_PID}" ]]; then
    echo "Waiting for ARC scheduler run pid=${ARC_PID} ..."
    while kill -0 "${ARC_PID}" 2>/dev/null; do
      sleep 20
    done
    echo "ARC run finished; starting Hotpot R9."
  fi
fi

bash "${HOTPOT_SCRIPT}" > "${HOTPOT_LOG}" 2>&1

