#!/usr/bin/env bash
set -euo pipefail

# Round22: serial adaptive search with early pruning.
# Goals:
# - keep one-GPU serial execution
# - prioritize real tasks (MBPP, protein SS), add broader Sudoku map
# - maximize utilization by probing largest stable batch size first
# - prune fast on weak quick deltas, spend budget only on promising configs

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
  cd "$SCRIPT_DIR/.."
else
  cd "$SCRIPT_DIR"
fi

if [[ -f scripts/train_mbpp_longctx_sft.py ]]; then
  PY_PREFIX="scripts/"
else
  PY_PREFIX=""
fi

export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions
export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_transformers
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.9

SEED=0
Q_BUDGET=120
Q_STEPS=180
M_BUDGET=420
M_STEPS=650
L_BUDGET=900
L_STEPS=1400

MBPP_PRUNE_DACC=0.003
PROTEIN_PRUNE_DACC=0.004
SUDOKU_PRUNE_DACC=0.002

RUN_REC=runs/_round22_adaptive_search_records.jsonl
SUMMARY=runs/_summary_round22_adaptive_search_s0.txt
LOG=runs/_log_round22_adaptive_search_s0.$(date +%Y%m%d_%H%M%S).log
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
  local log="runs/_round22_${task}_${stage}_${cfg}.log"
  rm -f "$log"

  echo "RUN task=$task stage=$stage cfg=$cfg mode=$mode bsz=$bsz budget=$budget" >&2
  set +e
  ./.venv/bin/python "${PY_PREFIX}${trainer}" "$@" --mode "$mode" --seed "$SEED" --bsz "$bsz" \
    --time_budget_sec "$budget" --max_steps "$steps" > "$log" 2>&1
  local code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    local err
    err=$(tail -n 2 "$log" | tr '\n' ' ' | sed 's/"/\\"/g')
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

probe_bsz_highest() {
  local task="$1"; shift
  local trainer="$1"; shift
  local bsz_list="$1"; shift
  local chosen=1
  for b in $bsz_list; do
    local log="runs/_round22_probe_${task}_b${b}.log"
    rm -f "$log"
    echo "PROBE task=$task bsz=$b" >&2
    set +e
    ./.venv/bin/python "${PY_PREFIX}${trainer}" "$@" --mode no_fs --seed "$SEED" --bsz "$b" \
      --time_budget_sec 40 --max_steps 30 --eval_every 12 --val_batches 2 > "$log" 2>&1
    local code=$?
    set -e
    if [[ $code -eq 0 ]]; then
      chosen="$b"
      break
    fi
  done
  echo "$chosen"
}

pick_best_med_cfg() {
  local task="$1"
  ./.venv/bin/python - <<PY
import json
from pathlib import Path
rows=[json.loads(x) for x in Path("$RUN_REC").read_text().splitlines() if x.strip()]
rows=[r for r in rows if r.get("task")=="$task" and r.get("stage")=="med" and r.get("status")=="ok"]
if not rows:
    print("")
else:
    rows=sorted(rows, key=lambda r: float(r["best_val_tok_acc"]), reverse=True)
    print(rows[0]["config"])
PY
}

run_task_mbpp() {
  local task=mbpp_focus
  local trainer=train_mbpp_longctx_sft.py
  local base=(
    --train_data_seed 0 --val_data_seed 1234
    --ds mbpp --train_split train --val_split test
    --fill_notes_to_max --note_pool_size 1024
    --n_train 320 --n_val 80
    --max_prompt_tokens 3072 --min_prompt_tokens 768 --max_answer_tokens 160
    --eval_every 24 --val_batches 4
    --model_lr 3e-5 --seed_scale 1.0
  )
  local bsz
  bsz=$(probe_bsz_highest "$task" "$trainer" "4 3 2 1" "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0)
  record "{\"task\":\"$task\",\"stage\":\"probe\",\"config\":\"bsz\",\"status\":\"ok\",\"bsz\":$bsz,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\"}"

  read -r _ b_acc _ < <(
    run_train "$task" quick baseline "$trainer" no_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" \
      "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0
  )
  if [[ "$b_acc" == "nan" ]]; then
    echo "TASK $task baseline failed" >&2
    return
  fi

  for cfg in scalar_l8_trainable scalar_l8_sched_cos scalar_l10_trainable head_l8 scalar_l8_nodetach; do
    local args=""
    case "$cfg" in
      scalar_l8_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l8_sched_cos) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0";;
      scalar_l10_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0";;
      head_l8) args="--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l8_nodetach) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_clip 1.0";;
    esac
    read -r _ q_acc _ < <(run_train "$task" quick "$cfg" "$trainer" prompt_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" "${base[@]}" $args)
    if [[ "$q_acc" == "nan" ]]; then
      continue
    fi
    local d_acc
    d_acc=$(python3 - <<PY
print(float("$q_acc")-float("$b_acc"))
PY
)
    local keep
    keep=$(python3 - <<PY
print(1 if float("$d_acc") >= float("$MBPP_PRUNE_DACC") else 0)
PY
)
    if [[ "$keep" != "1" ]]; then
      echo "PRUNE task=$task cfg=$cfg d_acc_pp=$(python3 - <<PY
print((float("$d_acc")*100.0))
PY
)%" >&2
      continue
    fi
    run_train "$task" med "$cfg" "$trainer" prompt_fs "$bsz" "$M_BUDGET" "$M_STEPS" "${base[@]}" $args >/dev/null
  done

  local best_cfg
  best_cfg=$(pick_best_med_cfg "$task")
  if [[ -n "$best_cfg" ]]; then
    local args=""
    case "$best_cfg" in
      scalar_l8_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l8_sched_cos) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0";;
      scalar_l10_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0";;
      head_l8) args="--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l8_nodetach) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_clip 1.0";;
    esac
    run_train "$task" long "$best_cfg" "$trainer" prompt_fs "$bsz" "$L_BUDGET" "$L_STEPS" "${base[@]}" $args >/dev/null
  fi
}

run_task_protein_ss() {
  local task=protein_ss_expand
  local trainer=train_protein_ss_spot_sft.py
  local base=(
    --train_data_seed 0 --val_data_seed 1234
    --ds lamm-mit/protein_secondary_structure_from_PDB --split train
    --n_train 1800 --n_val 320
    --max_seq_len 512 --min_seq_len 96 --num_queries 64 --query_region random
    --fill_notes_to_max --note_pool_size 2048 --max_note_seq_len 256
    --max_prompt_tokens 2048 --min_prompt_tokens 1024 --max_answer_tokens 128
    --eval_every 24 --val_batches 6
    --model_lr 3e-5 --seed_scale 1.0
  )
  local bsz
  bsz=$(probe_bsz_highest "$task" "$trainer" "16 12 10 8 6" "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0)
  record "{\"task\":\"$task\",\"stage\":\"probe\",\"config\":\"bsz\",\"status\":\"ok\",\"bsz\":$bsz,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\"}"

  read -r _ b_acc _ < <(
    run_train "$task" quick baseline "$trainer" no_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" \
      "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0
  )
  if [[ "$b_acc" == "nan" ]]; then
    echo "TASK $task baseline failed" >&2
    return
  fi

  for cfg in scalar_l10_sched_cos scalar_l10_trainable scalar_l12_sched_cos head_l10 scalar_l10_nodetach; do
    local args=""
    case "$cfg" in
      scalar_l10_sched_cos) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0";;
      scalar_l10_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l12_sched_cos) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 12 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0";;
      head_l10) args="--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l10_nodetach) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_clip 1.0";;
    esac
    read -r _ q_acc _ < <(run_train "$task" quick "$cfg" "$trainer" prompt_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" "${base[@]}" $args)
    if [[ "$q_acc" == "nan" ]]; then
      continue
    fi
    local d_acc
    d_acc=$(python3 - <<PY
print(float("$q_acc")-float("$b_acc"))
PY
)
    local keep
    keep=$(python3 - <<PY
print(1 if float("$d_acc") >= float("$PROTEIN_PRUNE_DACC") else 0)
PY
)
    if [[ "$keep" != "1" ]]; then
      continue
    fi
    run_train "$task" med "$cfg" "$trainer" prompt_fs "$bsz" "$M_BUDGET" "$M_STEPS" "${base[@]}" $args >/dev/null
  done

  local best_cfg
  best_cfg=$(pick_best_med_cfg "$task")
  if [[ -n "$best_cfg" ]]; then
    local args=""
    case "$best_cfg" in
      scalar_l10_sched_cos) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0";;
      scalar_l10_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l12_sched_cos) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 12 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0";;
      head_l10) args="--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l10_nodetach) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_clip 1.0";;
    esac
    run_train "$task" long "$best_cfg" "$trainer" prompt_fs "$bsz" "$L_BUDGET" "$L_STEPS" "${base[@]}" $args >/dev/null
  fi
}

run_task_sudoku() {
  local task="$1"; shift
  local baseN="$1"; shift
  local mask_count="$1"; shift
  local mask_region="$1"; shift
  local trainer=train_sudoku_sft.py
  local base=(--base "$baseN" --mask_count "$mask_count" --mask_region "$mask_region" --eval_every 60 --val_batches 16 --model_lr 3e-5 --seed_scale 1.0)
  local bsz_list="64 48 32 24 16 12 8"
  if [[ "$baseN" == "3" ]]; then
    bsz_list="16 12 10 8 6 4"
  fi
  local bsz
  bsz=$(probe_bsz_highest "$task" "$trainer" "$bsz_list" "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0)
  record "{\"task\":\"$task\",\"stage\":\"probe\",\"config\":\"bsz\",\"status\":\"ok\",\"bsz\":$bsz,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\"}"

  read -r _ b_acc _ < <(
    run_train "$task" quick baseline "$trainer" no_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" \
      "${base[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0
  )
  if [[ "$b_acc" == "nan" ]]; then
    return
  fi
  for cfg in scalar_l6_trainable scalar_l6_sched_cos scalar_l8_trainable head_l6; do
    local args=""
    case "$cfg" in
      scalar_l6_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0";;
      scalar_l6_sched_cos) args="--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0";;
      scalar_l8_trainable) args="--alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0";;
      head_l6) args="--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0";;
    esac
    read -r _ q_acc _ < <(run_train "$task" quick "$cfg" "$trainer" prompt_fs "$bsz" "$Q_BUDGET" "$Q_STEPS" "${base[@]}" $args)
    if [[ "$q_acc" == "nan" ]]; then
      continue
    fi
    local d_acc
    d_acc=$(python3 - <<PY
print(float("$q_acc")-float("$b_acc"))
PY
)
    local keep
    keep=$(python3 - <<PY
print(1 if float("$d_acc") >= float("$SUDOKU_PRUNE_DACC") else 0)
PY
)
    if [[ "$keep" != "1" ]]; then
      continue
    fi
    run_train "$task" med "$cfg" "$trainer" prompt_fs "$bsz" "$M_BUDGET" "$M_STEPS" "${base[@]}" $args >/dev/null
  done
}

{
  echo "Round22 started: $(date)"
  run_task_mbpp
  run_task_protein_ss
  run_task_sudoku sudoku4_refine 2 8 random
  run_task_sudoku sudoku9_probe 3 28 random
  ./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path

rows=[json.loads(x) for x in Path("runs/_round22_adaptive_search_records.jsonl").read_text().splitlines() if x.strip()]
tasks=sorted({r["task"] for r in rows})
print("="*118)
print("Round22 adaptive search summary (single seed, serial early-prune)")
print("="*118)
for t in tasks:
    print(f"[{t}]")
    b=[r for r in rows if r["task"]==t and r["stage"]=="quick" and r["config"]=="baseline" and r["status"]=="ok"]
    if not b:
        print("  baseline failed or missing")
        print("-"*118)
        continue
    b=b[0]
    bacc=float(b["best_val_tok_acc"])
    print(f"  baseline quick: acc={bacc*100:.2f}% loss={float(b['best_val_loss']):.4f} bsz={b['bsz']}")
    q=[r for r in rows if r["task"]==t and r["stage"]=="quick" and r["config"]!="baseline" and r["status"]=="ok"]
    q=sorted(q,key=lambda x:float(x["best_val_tok_acc"])-bacc, reverse=True)
    for r in q:
        da=float(r["best_val_tok_acc"])-bacc
        print(f"    {r['config']:24s} d_acc={da*100:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}%")
    m=[r for r in rows if r["task"]==t and r["stage"]=="med" and r["status"]=="ok"]
    if m:
        print("  med confirmed:")
        for r in sorted(m,key=lambda x:float(x["best_val_tok_acc"]), reverse=True):
            da=float(r["best_val_tok_acc"])-bacc
            print(f"    {r['config']:24s} d_acc={da*100:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}% loss={float(r['best_val_loss']):.4f}")
    l=[r for r in rows if r["task"]==t and r["stage"]=="long" and r["status"]=="ok"]
    if l:
        print("  long confirmed:")
        for r in sorted(l,key=lambda x:float(x["best_val_tok_acc"]), reverse=True):
            da=float(r["best_val_tok_acc"])-bacc
            print(f"    {r['config']:24s} d_acc={da*100:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}% loss={float(r['best_val_loss']):.4f}")
    print("-"*118)
PY
  echo "Round22 finished: $(date)"
} 2>&1 | tee "$LOG"
