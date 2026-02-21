#!/usr/bin/env bash
set -euo pipefail

# Round21: targeted follow-up after round20 findings.
# - keep strict serial execution (single 4090)
# - fix dataset-build failures (mbpp/protein_contact)
# - search only promising FS variants

cd /root/autodl-tmp/future-seed-posttrain

export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions
export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_transformers
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.9

SEED=0
Q_BUDGET=90
Q_STEPS=140
M_BUDGET=260
M_STEPS=380
PRUNE_DACC=0.002

RUN_REC=runs/_round21_targeted_search_records.jsonl
SUMMARY=runs/_summary_round21_targeted_search_s0.txt
: > "$RUN_REC"

metric_best() {
  local run_dir="$1"
  ./.venv/bin/python - <<PY
import json
from pathlib import Path
p=Path("$run_dir")/"metrics.jsonl"
best_loss=None; best_acc=None
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    r=json.loads(line)
    if "val_loss" in r:
        v=float(r["val_loss"]); best_loss=v if best_loss is None or v<best_loss else best_loss
    if "val_tok_acc" in r:
        v=float(r["val_tok_acc"]); best_acc=v if best_acc is None or v>best_acc else best_acc
print(f"{best_loss}\t{best_acc}")
PY
}

record() {
  printf "%s\n" "$1" >> "$RUN_REC"
}

run_train() {
  local task="$1"; shift
  local stage="$1"; shift
  local cfg="$1"; shift
  local trainer="$1"; shift
  local mode="$1"; shift
  local bsz="$1"; shift
  local budget="$1"; shift
  local steps="$1"; shift
  local log="runs/_round21_${task}_${stage}_${cfg}.log"
  rm -f "$log"

  echo "RUN task=$task stage=$stage cfg=$cfg mode=$mode bsz=$bsz budget=$budget" >&2
  set +e
  ./.venv/bin/python "$trainer" "$@" --mode "$mode" --seed "$SEED" --bsz "$bsz" \
    --time_budget_sec "$budget" --max_steps "$steps" > "$log" 2>&1
  local code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    local err
    err=$(tail -n 1 "$log" | sed 's/"/\\"/g')
    record "{\"task\":\"$task\",\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"fail\",\"bsz\":$bsz,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\",\"error\":\"$err\"}"
    echo -e "nan\tnan\t"
    return
  fi

  local run_dir
  run_dir=$(tail -n 1 "$log")
  if [[ ! -f "$run_dir/metrics.jsonl" ]]; then
    record "{\"task\":\"$task\",\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"fail\",\"bsz\":$bsz,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"$run_dir\",\"error\":\"missing_metrics\"}"
    echo -e "nan\tnan\t$run_dir"
    return
  fi
  read -r best_loss best_acc < <(metric_best "$run_dir")
  record "{\"task\":\"$task\",\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"ok\",\"bsz\":$bsz,\"best_val_loss\":$best_loss,\"best_val_tok_acc\":$best_acc,\"run_dir\":\"$run_dir\"}"
  echo -e "${best_loss}\t${best_acc}\t${run_dir}"
}

probe_bsz() {
  local task="$1"; shift
  local trainer="$1"; shift
  local bsz_list="$1"; shift
  local chosen=1
  for b in $bsz_list; do
    local log="runs/_round21_probe_${task}_b${b}.log"
    rm -f "$log"
    echo "PROBE task=$task bsz=$b" >&2
    set +e
    ./.venv/bin/python "$trainer" "$@" --mode no_fs --seed "$SEED" --bsz "$b" \
      --time_budget_sec 35 --max_steps 25 --eval_every 10 --val_batches 2 > "$log" 2>&1
    local code=$?
    set -e
    if [[ $code -eq 0 ]]; then
      chosen="$b"
      break
    fi
  done
  echo "$chosen"
}

run_task_mbpp() {
  local task=mbpp_fix
  local trainer=train_mbpp_longctx_sft.py
  local base=(
    --train_data_seed 0 --val_data_seed 1234
    --ds mbpp --train_split train --val_split test
    --fill_notes_to_max --note_pool_size 1024
    --n_train 320 --n_val 80
    --max_prompt_tokens 3072 --min_prompt_tokens 768 --max_answer_tokens 160
    --eval_every 20 --val_batches 4
    --model_lr 3e-5 --seed_scale 1.0
  )
  local bsz
  bsz=$(probe_bsz "$task" "$trainer" "2 1" "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0)
  echo "SELECT task=$task bsz=$bsz" >&2
  record "{\"task\":\"$task\",\"stage\":\"probe\",\"config\":\"bsz\",\"status\":\"ok\",\"bsz\":$bsz,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\"}"

  read -r b_loss b_acc _ < <(
    run_train "$task" quick baseline "$trainer" no_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" \
      "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0
  )
  if [[ "$b_acc" == "nan" ]]; then
    echo "TASK $task baseline failed" >&2
    return
  fi

  for cfg in scalar_l8_norm_node scalar_l8_trainable scalar_l8_sched_cos head_l8; do
    local args=""
    case "$cfg" in
      scalar_l8_norm_node) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_clip 1.0";;
      scalar_l8_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l8_sched_cos) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0";;
      head_l8) args="--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0";;
    esac
    read -r q_loss q_acc _ < <(run_train "$task" quick "$cfg" "$trainer" prompt_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" "${base[@]}" $args)
    if [[ "$q_acc" == "nan" ]]; then
      echo "PRUNE task=$task cfg=$cfg reason=run_fail" >&2
      continue
    fi
    d_acc=$(python3 - <<PY
print(float("$q_acc")-float("$b_acc"))
PY
)
    keep=$(python3 - <<PY
print(1 if float("$d_acc") >= float("$PRUNE_DACC") else 0)
PY
)
    if [[ "$keep" != "1" ]]; then
      echo "PRUNE task=$task cfg=$cfg d_acc=$d_acc (<$PRUNE_DACC)" >&2
      continue
    fi
    echo "KEEP task=$task cfg=$cfg quick_d_acc=$d_acc -> med" >&2
    run_train "$task" med "$cfg" "$trainer" prompt_fs "$bsz" "$M_BUDGET" "$M_STEPS" "${base[@]}" $args >/dev/null
  done
}

run_task_protein_contact() {
  local task=protein_contact_fix
  local trainer=train_protein_contact_pair_sft.py
  local base=(
    --train_data_seed 0 --val_data_seed 1234
    --ds heya5/protein_contact_map --train_split train --val_split valid
    --n_train 1200 --n_val 140
    --max_seq_len 384 --min_seq_len 96 --num_pairs 32 --min_sep 8 --contact_cutoff 8.0
    --fill_notes_to_max --note_pool_size 1024 --max_note_seq_len 192
    --max_prompt_tokens 1536 --min_prompt_tokens 768 --max_answer_tokens 128
    --eval_every 20 --val_batches 4
    --model_lr 3e-5 --seed_scale 1.0
  )
  local bsz
  bsz=$(probe_bsz "$task" "$trainer" "4 3 2 1" "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0)
  echo "SELECT task=$task bsz=$bsz" >&2
  record "{\"task\":\"$task\",\"stage\":\"probe\",\"config\":\"bsz\",\"status\":\"ok\",\"bsz\":$bsz,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\"}"

  read -r b_loss b_acc _ < <(
    run_train "$task" quick baseline "$trainer" no_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" \
      "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0
  )
  if [[ "$b_acc" == "nan" ]]; then
    echo "TASK $task baseline failed" >&2
    return
  fi

  for cfg in scalar_l8_norm_node scalar_l8_trainable scalar_l8_sched_cos head_l8; do
    local args=""
    case "$cfg" in
      scalar_l8_norm_node) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_clip 1.0";;
      scalar_l8_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l8_sched_cos) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0";;
      head_l8) args="--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0";;
    esac
    read -r q_loss q_acc _ < <(run_train "$task" quick "$cfg" "$trainer" prompt_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" "${base[@]}" $args)
    if [[ "$q_acc" == "nan" ]]; then
      echo "PRUNE task=$task cfg=$cfg reason=run_fail" >&2
      continue
    fi
    d_acc=$(python3 - <<PY
print(float("$q_acc")-float("$b_acc"))
PY
)
    keep=$(python3 - <<PY
print(1 if float("$d_acc") >= float("$PRUNE_DACC") else 0)
PY
)
    if [[ "$keep" != "1" ]]; then
      echo "PRUNE task=$task cfg=$cfg d_acc=$d_acc (<$PRUNE_DACC)" >&2
      continue
    fi
    echo "KEEP task=$task cfg=$cfg quick_d_acc=$d_acc -> med" >&2
    run_train "$task" med "$cfg" "$trainer" prompt_fs "$bsz" "$M_BUDGET" "$M_STEPS" "${base[@]}" $args >/dev/null
  done
}

run_task_protein_ss() {
  local task=protein_ss_refine
  local trainer=train_protein_ss_spot_sft.py
  local base=(
    --train_data_seed 0 --val_data_seed 1234
    --ds lamm-mit/protein_secondary_structure_from_PDB --split train
    --n_train 1400 --n_val 280
    --max_seq_len 512 --min_seq_len 96 --num_queries 48 --query_region random
    --fill_notes_to_max --note_pool_size 2048 --max_note_seq_len 256
    --max_prompt_tokens 2048 --min_prompt_tokens 1024 --max_answer_tokens 128
    --eval_every 20 --val_batches 6
    --model_lr 3e-5 --seed_scale 1.0
  )
  local bsz=8
  record "{\"task\":\"$task\",\"stage\":\"probe\",\"config\":\"bsz\",\"status\":\"ok\",\"bsz\":$bsz,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\"}"
  read -r b_loss b_acc _ < <(
    run_train "$task" quick baseline "$trainer" no_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" \
      "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0
  )
  if [[ "$b_acc" == "nan" ]]; then
    echo "TASK $task baseline failed" >&2
    return
  fi

  for cfg in scalar_l10_norm_node scalar_l10_trainable scalar_l10_sched_cos head_l10; do
    local args=""
    case "$cfg" in
      scalar_l10_norm_node) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_clip 1.0";;
      scalar_l10_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l10_sched_cos) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0";;
      head_l10) args="--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0";;
    esac
    read -r q_loss q_acc _ < <(run_train "$task" quick "$cfg" "$trainer" prompt_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" "${base[@]}" $args)
    if [[ "$q_acc" == "nan" ]]; then
      echo "PRUNE task=$task cfg=$cfg reason=run_fail" >&2
      continue
    fi
    d_acc=$(python3 - <<PY
print(float("$q_acc")-float("$b_acc"))
PY
)
    keep=$(python3 - <<PY
print(1 if float("$d_acc") >= float("$PRUNE_DACC") else 0)
PY
)
    if [[ "$keep" != "1" ]]; then
      echo "PRUNE task=$task cfg=$cfg d_acc=$d_acc (<$PRUNE_DACC)" >&2
      continue
    fi
    echo "KEEP task=$task cfg=$cfg quick_d_acc=$d_acc -> med" >&2
    run_train "$task" med "$cfg" "$trainer" prompt_fs "$bsz" "$M_BUDGET" "$M_STEPS" "${base[@]}" $args >/dev/null
  done
}

run_task_mbpp
run_task_protein_contact
run_task_protein_ss

./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path

rows=[json.loads(x) for x in Path("runs/_round21_targeted_search_records.jsonl").read_text().splitlines() if x.strip()]
tasks=sorted({r["task"] for r in rows})
print("="*108)
print("Round21 targeted search summary (single seed)")
print("="*108)
for t in tasks:
    print(f"[{t}]")
    b=[r for r in rows if r["task"]==t and r["stage"]=="quick" and r["config"]=="baseline" and r["status"]=="ok"]
    if not b:
        print("  baseline failed or missing")
        print("-"*108)
        continue
    b=b[0]; bacc=float(b["best_val_tok_acc"]); bloss=float(b["best_val_loss"])
    print(f"  baseline quick: acc={bacc:.4f} loss={bloss:.4f} bsz={b['bsz']}")
    q=[r for r in rows if r["task"]==t and r["stage"]=="quick" and r["config"]!="baseline" and r["status"]=="ok"]
    q=sorted(q,key=lambda x:float(x["best_val_tok_acc"])-bacc, reverse=True)
    for r in q:
        da=float(r["best_val_tok_acc"])-bacc
        mark="KEEP" if da>=0.002 else "PRUNE"
        print(f"    {r['config']:24s} d_acc={da:+.4f} acc={float(r['best_val_tok_acc']):.4f} [{mark}]")
    m=[r for r in rows if r["task"]==t and r["stage"]=="med" and r["status"]=="ok"]
    if m:
        print("  med confirmed:")
        for r in sorted(m,key=lambda x:float(x["best_val_tok_acc"]), reverse=True):
            da=float(r["best_val_tok_acc"])-bacc
            print(f"    {r['config']:24s} d_acc={da:+.4f} acc={float(r['best_val_tok_acc']):.4f} loss={float(r['best_val_loss']):.4f}")
    else:
        print("  med confirmed: none")
    print("-"*108)
PY

