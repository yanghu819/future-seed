#!/usr/bin/env bash
set -euo pipefail

# Round38: rapid scene sweep on known real-task-positive domains.
# Search axis: prompt length regime (max/min prompt tokens).
# Datasets limited to known-available tasks to avoid network stalls.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
  cd "$SCRIPT_DIR/.."
else
  cd "$SCRIPT_DIR"
fi

if [[ -f scripts/train_punc_restore_sft.py ]]; then
  PY_PREFIX="scripts/"
else
  PY_PREFIX=""
fi

export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions
export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_transformers
export HF_ENDPOINT=https://huggingface.co
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.9

SEED=0
BSZ=6
Q_BUDGET=70
Q_STEPS=100
M_BUDGET=180
M_STEPS=260
PRUNE=0.010

RUN_REC=runs/_round38_punc_length_sweep_s0_records.jsonl
SUMMARY=runs/_summary_round38_punc_length_sweep_s0.txt
LOG=runs/_log_round38_punc_length_sweep_s0.$(date +%Y%m%d_%H%M%S).log
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
        v=float(r["val_loss"])
        best_loss=v if best_loss is None or v<best_loss else best_loss
    if "val_tok_acc" in r:
        v=float(r["val_tok_acc"])
        best_acc=v if best_acc is None or v>best_acc else best_acc
print(f"{best_loss}\t{best_acc}")
PY
}

record() { printf "%s\n" "$1" >> "$RUN_REC"; }

run_train() {
  local task="$1"; shift
  local scene="$1"; shift
  local stage="$1"; shift
  local cfg="$1"; shift
  local mode="$1"; shift
  local budget="$1"; shift
  local steps="$1"; shift
  local log="runs/_round38_${task}_${scene}_${stage}_${cfg}.log"
  rm -f "$log"
  echo "RUN task=$task scene=$scene stage=$stage cfg=$cfg mode=$mode" >&2
  set +e
  ./.venv/bin/python "${PY_PREFIX}train_punc_restore_sft.py" \
    --train_data_seed "$SEED" --val_data_seed 1234 \
    --model_lr 3e-5 --seed_scale 1.0 \
    --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 \
    "$@" --mode "$mode" --seed "$SEED" --bsz "$BSZ" \
    --time_budget_sec "$budget" --max_steps "$steps" > "$log" 2>&1
  local code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    local err
    err=$(tail -n 3 "$log" | tr '\n' ' ' | sed 's/"/\\"/g')
    record "{\"task\":\"$task\",\"scene\":\"$scene\",\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"fail\",\"bsz\":$BSZ,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\",\"error\":\"$err\"}"
    echo -e "nan\tnan\t"
    return
  fi
  local run_dir
  run_dir=$(tail -n 1 "$log")
  read -r best_loss best_acc < <(metric_best "$run_dir")
  record "{\"task\":\"$task\",\"scene\":\"$scene\",\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"ok\",\"bsz\":$BSZ,\"best_val_loss\":$best_loss,\"best_val_tok_acc\":$best_acc,\"run_dir\":\"$run_dir\"}"
  echo -e "${best_loss}\t${best_acc}\t${run_dir}"
}

run_scene() {
  local task="$1"; shift
  local scene="$1"; shift
  local fs_cfg="$1"; shift
  local common_args=("$@")

  read -r _ b_acc _ < <(run_train "$task" "$scene" quick baseline no_fs "$Q_BUDGET" "$Q_STEPS" "${common_args[@]}")
  [[ "$b_acc" == "nan" ]] && return

  local fs_args=()
  if [[ "$fs_cfg" == "head_l8" ]]; then
    fs_args=(--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0)
  else
    fs_args=(--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0)
  fi

  read -r _ f_acc _ < <(run_train "$task" "$scene" quick "$fs_cfg" prompt_fs "$Q_BUDGET" "$Q_STEPS" "${common_args[@]}" "${fs_args[@]}")
  [[ "$f_acc" == "nan" ]] && return

  local d
  d=$(python3 - <<PY
print(float("$f_acc")-float("$b_acc"))
PY
)
  local keep
  keep=$(python3 - <<PY
print(1 if float("$d") >= float("$PRUNE") else 0)
PY
)
  if [[ "$keep" == "1" ]]; then
    run_train "$task" "$scene" med "$fs_cfg" prompt_fs "$M_BUDGET" "$M_STEPS" "${common_args[@]}" "${fs_args[@]}" >/dev/null
  fi
}

run_task_sweep() {
  local task="$1"; shift
  local fs_cfg="$1"; shift
  local ds="$1"; shift
  local ds_cfg="$1"; shift
  local train_split="$1"; shift
  local val_split="$1"; shift
  local n_train="$1"; shift
  local n_val="$1"; shift
  local min_chars="$1"; shift
  local max_chars="$1"; shift
  local max_answer="$1"; shift

  local base=(--ds "$ds" --ds_cfg "$ds_cfg" --train_split "$train_split" --val_split "$val_split" \
    --n_train "$n_train" --n_val "$n_val" --min_chars "$min_chars" --max_chars "$max_chars" \
    --fill_notes_to_max --note_pool_size 1024 --max_answer_tokens "$max_answer" --eval_every 20 --val_batches 4)

  run_scene "$task" "L768"  "$fs_cfg" "${base[@]}" --max_prompt_tokens 768  --min_prompt_tokens 256
  run_scene "$task" "L1536" "$fs_cfg" "${base[@]}" --max_prompt_tokens 1536 --min_prompt_tokens 512
  run_scene "$task" "L3072" "$fs_cfg" "${base[@]}" --max_prompt_tokens 3072 --min_prompt_tokens 1024
}

{
  echo "Round38 started: $(date)"
  run_task_sweep punc_hotpot head_l8 hotpot_qa distractor train validation 800 160 48 220 128
  run_task_sweep punc_mbpp scalar_l8_sched_cos mbpp '' train test 320 80 32 360 160
  run_task_sweep punc_squad scalar_l8_sched_cos squad '' train validation 900 180 64 260 128

  ./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path
rows=[json.loads(x) for x in Path("runs/_round38_punc_length_sweep_s0_records.jsonl").read_text().splitlines() if x.strip()]
print("="*120)
print("Round38 punc length sweep summary")
print("="*120)
for task in sorted({r["task"] for r in rows}):
    print(f"[{task}]")
    scenes=sorted({r["scene"] for r in rows if r["task"]==task})
    best_scene=None
    best_delta=-1e9
    for sc in scenes:
        b=[r for r in rows if r["task"]==task and r["scene"]==sc and r["stage"]=="quick" and r["config"]=="baseline" and r["status"]=="ok"]
        f=[r for r in rows if r["task"]==task and r["scene"]==sc and r["stage"]=="quick" and r["config"]!="baseline" and r["status"]=="ok"]
        if not b or not f:
            print(f"  {sc}: baseline/fs missing")
            continue
        bacc=float(b[0]["best_val_tok_acc"])
        fbest=max(f,key=lambda x: float(x["best_val_tok_acc"]))
        facc=float(fbest["best_val_tok_acc"])
        d=facc-bacc
        print(f"  {sc}: baseline={bacc*100:.2f}% best_fs={fbest['config']} {facc*100:.2f}% d_acc={d*100:+.2f}pp")
        m=[r for r in rows if r["task"]==task and r["scene"]==sc and r["stage"]=="med" and r["status"]=="ok"]
        if m:
            mbest=max(m,key=lambda x: float(x["best_val_tok_acc"]))
            md=float(mbest["best_val_tok_acc"])-bacc
            print(f"       med={mbest['config']} acc={float(mbest['best_val_tok_acc'])*100:.2f}% d_acc={md*100:+.2f}pp")
        if d>best_delta:
            best_delta=d; best_scene=(sc,fbest["config"],bacc,facc)
    if best_scene is not None:
        sc,c,b,f=best_scene
        print(f"  => best quick scene: {sc} with {c}, gain={((f-b)*100):+.2f}pp")
    print("-"*120)
PY
  echo "Round38 finished: $(date)"
} 2>&1 | tee "$LOG"

