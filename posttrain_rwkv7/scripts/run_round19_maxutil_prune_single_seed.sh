#!/usr/bin/env bash
set -euo pipefail

# Round19 goals:
# - maximize single-GPU utilization via per-task batch-size probing
# - prune aggressively with short-budget screening
# - search algorithm variants + task suitability (single seed, no multi-seed padding)

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
QUICK_BUDGET=90
QUICK_STEPS=140
MED_BUDGET=240
MED_STEPS=360
TOPK=2
KEEP_DACC=0.001

RUN_REC=runs/_round19_maxutil_prune_records.jsonl
SUMMARY=runs/_summary_round19_maxutil_prune_single_seed.txt
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
    if 'val_loss' in r:
        v=float(r['val_loss']); best_loss=v if best_loss is None or v<best_loss else best_loss
    if 'val_tok_acc' in r:
        v=float(r['val_tok_acc']); best_acc=v if best_acc is None or v>best_acc else best_acc
print(f"{best_loss}\t{best_acc}")
PY
}

record_json() {
  local payload="$1"
  ./.venv/bin/python - <<PY >> "$RUN_REC"
import json
print(json.dumps($payload))
PY
}

run_exp() {
  local task="$1"; shift
  local stage="$1"; shift
  local cfg="$1"; shift
  local mode="$1"; shift
  local trainer="$1"; shift
  local bsz="$1"; shift
  local budget="$1"; shift
  local steps="$1"; shift
  local log_file="runs/_round19_${task}_${stage}_${cfg}.log"
  rm -f "$log_file"

  echo "RUN task=$task stage=$stage cfg=$cfg mode=$mode bsz=$bsz budget=$budget"
  set +e
  ./.venv/bin/python "$trainer" "$@" --mode "$mode" --seed "$SEED" --bsz "$bsz" \
    --time_budget_sec "$budget" --max_steps "$steps" > "$log_file" 2>&1
  local code=$?
  set -e

  if [[ $code -ne 0 ]]; then
    local err
    err=$(tail -n 1 "$log_file" | sed 's/"/\\"/g')
    record_json "{\"task\":\"$task\",\"stage\":\"$stage\",\"config\":\"$cfg\",\"mode\":\"$mode\",\"status\":\"fail\",\"seed\":$SEED,\"bsz\":$bsz,\"budget\":$budget,\"steps\":$steps,\"run_dir\":\"\",\"best_val_loss\":null,\"best_val_tok_acc\":null,\"error\":\"$err\"}"
    echo "FAIL task=$task cfg=$cfg (see $log_file)"
    echo $'\t'
    return
  fi

  local run_dir
  run_dir=$(tail -n 1 "$log_file")
  if [[ ! -f "$run_dir/metrics.jsonl" ]]; then
    record_json "{\"task\":\"$task\",\"stage\":\"$stage\",\"config\":\"$cfg\",\"mode\":\"$mode\",\"status\":\"fail\",\"seed\":$SEED,\"bsz\":$bsz,\"budget\":$budget,\"steps\":$steps,\"run_dir\":\"$run_dir\",\"best_val_loss\":null,\"best_val_tok_acc\":null,\"error\":\"missing_metrics\"}"
    echo "FAIL task=$task cfg=$cfg (missing metrics)"
    echo $'\t'
    return
  fi

  read -r best_loss best_acc < <(metric_best "$run_dir")
  record_json "{\"task\":\"$task\",\"stage\":\"$stage\",\"config\":\"$cfg\",\"mode\":\"$mode\",\"status\":\"ok\",\"seed\":$SEED,\"bsz\":$bsz,\"budget\":$budget,\"steps\":$steps,\"run_dir\":\"$run_dir\",\"best_val_loss\":$best_loss,\"best_val_tok_acc\":$best_acc}"
  echo -e "${best_loss}\t${best_acc}\t${run_dir}"
}

probe_bsz() {
  local task="$1"; shift
  local trainer="$1"; shift
  local bsz_candidates="$1"; shift
  local probe_log_prefix="runs/_round19_probe_${task}"

  local chosen=1
  for b in $bsz_candidates; do
    local log_file="${probe_log_prefix}_b${b}.log"
    rm -f "$log_file"
    echo "PROBE task=$task bsz=$b"
    set +e
    ./.venv/bin/python "$trainer" "$@" --mode no_fs --seed "$SEED" --bsz "$b" \
      --time_budget_sec 35 --max_steps 25 --eval_every 10 --val_batches 2 > "$log_file" 2>&1
    local code=$?
    set -e
    if [[ $code -eq 0 ]]; then
      chosen="$b"
      break
    fi
    if ! grep -qi "out of memory" "$log_file"; then
      # non-OOM error: still try smaller bsz but keep log
      :
    fi
  done
  echo "$chosen"
}

promote_from_quick() {
  local task="$1"
  local topk="$2"
  local keep_dacc="$3"
  ./.venv/bin/python - <<PY
import json
from pathlib import Path

task="$task"
topk=int("$topk")
keep=float("$keep_dacc")
rows=[json.loads(x) for x in Path("$RUN_REC").read_text().splitlines() if x.strip()]
q=[r for r in rows if r["task"]==task and r["stage"]=="quick" and r["status"]=="ok"]
base=[r for r in q if r["config"]=="baseline"]
if not base:
    print("")
    raise SystemExit(0)
base=base[0]
ba=float(base["best_val_tok_acc"])
cands=[]
for r in q:
    if r["config"]=="baseline":
        continue
    da=float(r["best_val_tok_acc"])-ba
    cands.append((da,r["config"]))
cands.sort(reverse=True)
keep_cfg=[cfg for da,cfg in cands if da>=keep][:topk]
print(",".join(keep_cfg))
PY
}

run_task_hotpot() {
  local task=hotpot
  local trainer=train_hotpot_longctx_sft.py
  local base_args=(
    --train_data_seed 0 --val_data_seed 1234
    --ds hotpot_qa --ds_cfg distractor --train_split train --val_split validation
    --n_train 700 --n_val 140
    --max_prompt_tokens 4096 --min_prompt_tokens 1536 --max_answer_tokens 24
    --eval_every 20 --val_batches 4
    --model_lr 3e-5 --seed_scale 1.0
  )

  local bsz
  bsz=$(probe_bsz "$task" "$trainer" "8 6 4 3 2 1" "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0)
  echo "SELECT task=$task bsz=$bsz"
  record_json "{\"task\":\"$task\",\"stage\":\"probe\",\"config\":\"bsz\",\"mode\":\"na\",\"status\":\"ok\",\"seed\":$SEED,\"bsz\":$bsz,\"budget\":0,\"steps\":0,\"run_dir\":\"\",\"best_val_loss\":null,\"best_val_tok_acc\":null}"

  run_exp "$task" quick baseline no_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l10_norm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l12_norm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 12 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l10_norm_node prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l10_nonorm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l10_sched_cos prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0 >/dev/null
  run_exp "$task" quick head_l10 prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick inputgate_l10 prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 --fs_in_gate --fs_w_lr 5e-4 --fs_b_lr 5e-4 >/dev/null

  local promote
  promote=$(promote_from_quick "$task" "$TOPK" "$KEEP_DACC")
  if [[ -z "$promote" ]]; then
    echo "PRUNE task=$task (no quick winner >= $KEEP_DACC)"
    return
  fi
  echo "PROMOTE task=$task configs=$promote"

  run_exp "$task" med baseline no_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" \
    "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  IFS=',' read -r -a cfgs <<< "$promote"
  for cfg in "${cfgs[@]}"; do
    case "$cfg" in
      scalar_l10_norm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l12_norm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 12 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l10_norm_node)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_clip 1.0 >/dev/null;;
      scalar_l10_nonorm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l10_sched_cos)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0 >/dev/null;;
      head_l10)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
      inputgate_l10)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 --fs_in_gate --fs_w_lr 5e-4 --fs_b_lr 5e-4 >/dev/null;;
    esac
  done
}

run_task_mbpp() {
  local task=mbpp
  local trainer=train_mbpp_longctx_sft.py
  local base_args=(
    --train_data_seed 0 --val_data_seed 1234
    --ds mbpp --train_split train --val_split test
    --fill_notes_to_max --note_pool_size 2048
    --n_train 900 --n_val 180
    --max_prompt_tokens 4096 --min_prompt_tokens 1536 --max_answer_tokens 160
    --eval_every 20 --val_batches 4
    --model_lr 3e-5 --seed_scale 1.0
  )

  local bsz
  bsz=$(probe_bsz "$task" "$trainer" "3 2 1" "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0)
  echo "SELECT task=$task bsz=$bsz"
  record_json "{\"task\":\"$task\",\"stage\":\"probe\",\"config\":\"bsz\",\"mode\":\"na\",\"status\":\"ok\",\"seed\":$SEED,\"bsz\":$bsz,\"budget\":0,\"steps\":0,\"run_dir\":\"\",\"best_val_loss\":null,\"best_val_tok_acc\":null}"

  run_exp "$task" quick baseline no_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_norm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l8_norm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_norm_node prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_nonorm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_sched_cos prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0 >/dev/null
  run_exp "$task" quick head_l6 prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_trainable prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null

  local promote
  promote=$(promote_from_quick "$task" "$TOPK" "$KEEP_DACC")
  if [[ -z "$promote" ]]; then
    echo "PRUNE task=$task (no quick winner >= $KEEP_DACC)"
    return
  fi
  echo "PROMOTE task=$task configs=$promote"
  run_exp "$task" med baseline no_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" \
    "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null

  IFS=',' read -r -a cfgs <<< "$promote"
  for cfg in "${cfgs[@]}"; do
    case "$cfg" in
      scalar_l6_norm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l8_norm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l6_norm_node)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_clip 1.0 >/dev/null;;
      scalar_l6_nonorm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l6_sched_cos)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0 >/dev/null;;
      head_l6)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l6_trainable)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
    esac
  done
}

run_task_protein_ss() {
  local task=protein_ss
  local trainer=train_protein_ss_spot_sft.py
  local base_args=(
    --train_data_seed 0 --val_data_seed 1234
    --ds lamm-mit/protein_secondary_structure_from_PDB --split train
    --n_train 1200 --n_val 240
    --max_seq_len 512 --min_seq_len 96 --num_queries 48 --query_region random
    --fill_notes_to_max --note_pool_size 2048 --max_note_seq_len 256
    --max_prompt_tokens 2048 --min_prompt_tokens 1024 --max_answer_tokens 128
    --eval_every 20 --val_batches 4
    --model_lr 3e-5 --seed_scale 1.0
  )

  local bsz
  bsz=$(probe_bsz "$task" "$trainer" "8 6 4 3 2 1" "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0)
  echo "SELECT task=$task bsz=$bsz"
  record_json "{\"task\":\"$task\",\"stage\":\"probe\",\"config\":\"bsz\",\"mode\":\"na\",\"status\":\"ok\",\"seed\":$SEED,\"bsz\":$bsz,\"budget\":0,\"steps\":0,\"run_dir\":\"\",\"best_val_loss\":null,\"best_val_tok_acc\":null}"

  run_exp "$task" quick baseline no_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_norm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l8_norm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_norm_node prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_nonorm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_sched_cos prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0 >/dev/null
  run_exp "$task" quick head_l6 prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null

  local promote
  promote=$(promote_from_quick "$task" "$TOPK" "$KEEP_DACC")
  if [[ -z "$promote" ]]; then
    echo "PRUNE task=$task (no quick winner >= $KEEP_DACC)"
    return
  fi
  echo "PROMOTE task=$task configs=$promote"
  run_exp "$task" med baseline no_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" \
    "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null

  IFS=',' read -r -a cfgs <<< "$promote"
  for cfg in "${cfgs[@]}"; do
    case "$cfg" in
      scalar_l6_norm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l8_norm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l6_norm_node)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_clip 1.0 >/dev/null;;
      scalar_l6_nonorm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l6_sched_cos)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0 >/dev/null;;
      head_l6)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
    esac
  done
}

run_task_protein_contact() {
  local task=protein_contact
  local trainer=train_protein_contact_pair_sft.py
  local base_args=(
    --train_data_seed 0 --val_data_seed 1234
    --ds heya5/protein_contact_map --train_split train --val_split valid
    --n_train 1400 --n_val 180
    --max_seq_len 384 --min_seq_len 96 --num_pairs 32 --min_sep 8 --contact_cutoff 8.0
    --fill_notes_to_max --note_pool_size 2048 --max_note_seq_len 256
    --max_prompt_tokens 2048 --min_prompt_tokens 1024 --max_answer_tokens 128
    --eval_every 20 --val_batches 4
    --model_lr 3e-5 --seed_scale 1.0
  )

  local bsz
  bsz=$(probe_bsz "$task" "$trainer" "8 6 4 3 2 1" "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0)
  echo "SELECT task=$task bsz=$bsz"
  record_json "{\"task\":\"$task\",\"stage\":\"probe\",\"config\":\"bsz\",\"mode\":\"na\",\"status\":\"ok\",\"seed\":$SEED,\"bsz\":$bsz,\"budget\":0,\"steps\":0,\"run_dir\":\"\",\"best_val_loss\":null,\"best_val_tok_acc\":null}"

  run_exp "$task" quick baseline no_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_norm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l8_norm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_norm_node prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_nonorm_detach prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_detach --fs_clip 1.0 >/dev/null
  run_exp "$task" quick scalar_l6_sched_cos prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0 >/dev/null
  run_exp "$task" quick head_l6 prompt_fs "$trainer" "$bsz" "$QUICK_BUDGET" "$QUICK_STEPS" \
    "${base_args[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null

  local promote
  promote=$(promote_from_quick "$task" "$TOPK" "$KEEP_DACC")
  if [[ -z "$promote" ]]; then
    echo "PRUNE task=$task (no quick winner >= $KEEP_DACC)"
    return
  fi
  echo "PROMOTE task=$task configs=$promote"
  run_exp "$task" med baseline no_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" \
    "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null

  IFS=',' read -r -a cfgs <<< "$promote"
  for cfg in "${cfgs[@]}"; do
    case "$cfg" in
      scalar_l6_norm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l8_norm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l6_norm_node)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_clip 1.0 >/dev/null;;
      scalar_l6_nonorm_detach)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_detach --fs_clip 1.0 >/dev/null;;
      scalar_l6_sched_cos)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0 >/dev/null;;
      head_l6)
        run_exp "$task" med "$cfg" prompt_fs "$trainer" "$bsz" "$MED_BUDGET" "$MED_STEPS" "${base_args[@]}" --eval_every 25 --val_batches 8 --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 6 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null;;
    esac
  done
}

run_task_hotpot
run_task_mbpp
run_task_protein_ss
run_task_protein_contact

./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path
from collections import defaultdict

rows=[json.loads(x) for x in Path("runs/_round19_maxutil_prune_records.jsonl").read_text().splitlines() if x.strip()]
by_task_stage=defaultdict(list)
for r in rows:
    by_task_stage[(r["task"], r["stage"])].append(r)

print("="*110)
print("Round19 max-util prune summary (single seed search)")
print("="*110)

tasks=sorted({r["task"] for r in rows if r["stage"] in ("quick","med")})
for task in tasks:
    print(f"[{task}]")
    probes=[r for r in rows if r["task"]==task and r["stage"]=="probe"]
    if probes:
        print(f"  selected_bsz={probes[-1]['bsz']}")

    for stage in ["quick","med"]:
        xs=[r for r in rows if r["task"]==task and r["stage"]==stage and r["status"]=="ok"]
        if not xs:
            print(f"  {stage}: pruned / no successful runs")
            continue
        base=[r for r in xs if r["config"]=="baseline"]
        if not base:
            print(f"  {stage}: missing baseline")
            continue
        base=base[0]
        ba=float(base["best_val_tok_acc"]); bl=float(base["best_val_loss"])
        vals=[]
        for r in xs:
            if r["config"]=="baseline":
                continue
            da=float(r["best_val_tok_acc"])-ba
            dl=float(r["best_val_loss"])-bl
            vals.append((da,dl,r["config"],float(r["best_val_tok_acc"])))
        vals.sort(reverse=True)
        print(f"  {stage} baseline: acc={ba:.4f} loss={bl:.4f}")
        for da,dl,cfg,acc in vals:
            mark="KEEP" if da>=0.001 else ("MAYBE" if da>=0 else "PRUNE")
            print(f"    {cfg:26s} d_acc={da:+.4f} d_loss={dl:+.4f} fs_acc={acc:.4f} [{mark}]")
    print("-"*110)
PY

