#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/future-seed-posttrain

R11_WRAPPER_PID_FILE="runs/run_after_r10_start_r11.pid"

if [[ -f "$R11_WRAPPER_PID_FILE" ]]; then
  PID="$(cat "$R11_WRAPPER_PID_FILE" || true)"
  if [[ -n "${PID}" ]]; then
    echo "Waiting for R11 wrapper pid=${PID} ..."
    while kill -0 "${PID}" 2>/dev/null; do
      sleep 30
    done
    echo "R11 wrapper finished; starting MBPP + Sudoku suite."
  fi
fi

bash run_mbpp_qafter_stabilized_len4096_round1_s012.sh > runs/run_mbpp_qafter_stabilized_len4096_round1_s012.log 2>&1
bash run_mbpp_qfirst_stabilized_len4096_round1_s012.sh > runs/run_mbpp_qfirst_stabilized_len4096_round1_s012.log 2>&1
bash run_sudoku_suite_round1_s012.sh > runs/run_sudoku_suite_round1_s012.log 2>&1

