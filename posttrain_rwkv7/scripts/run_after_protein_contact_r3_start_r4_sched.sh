#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/future-seed-posttrain

echo "Waiting protein contact round3 to finish..."
while pgrep -f "run_protein_contact_pair_qafter_len2048_round3_balanced_s012.sh|train_protein_contact_pair_sft.py" >/dev/null 2>&1; do
  sleep 20
done

echo "Starting protein contact round4 schedule..."
bash run_protein_contact_pair_qafter_len2048_round4_sched_s012.sh \
  > runs/run_protein_contact_pair_qafter_len2048_round4_sched_s012.log 2>&1

