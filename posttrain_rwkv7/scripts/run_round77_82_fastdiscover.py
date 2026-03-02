#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
PYTHON = ROOT / ".venv" / "bin" / "python"

DEFAULT_POLICY: Dict[str, Any] = {
    "model": "gpt-5.3-codex",
    "quick_promote_pp": 0.8,
    "quick_prune_pp": -0.5,
    "strong_negative_quick_pp": -1.0,
    "strong_negative_med_pp": -0.5,
    "cooldown_rounds": 8,
    "report_template_4lines": [
        "当前在跑什么",
        "当前最好结果（任务名 + 提升pp）",
        "当前主要风险",
        "下一步两项实验",
    ],
}
ACTIVE_POLICY: Dict[str, Any] = dict(DEFAULT_POLICY)

PROMOTE_PP = float(DEFAULT_POLICY["quick_promote_pp"])
PRUNE_PP = float(DEFAULT_POLICY["quick_prune_pp"])
STRONG_NEG_QUICK_PP = float(DEFAULT_POLICY["strong_negative_quick_pp"])
STRONG_NEG_MED_PP = float(DEFAULT_POLICY["strong_negative_med_pp"])
COOLDOWN_ROUNDS = int(DEFAULT_POLICY["cooldown_rounds"])


def apply_policy(policy: Dict[str, Any]) -> None:
    global PROMOTE_PP, PRUNE_PP, STRONG_NEG_QUICK_PP, STRONG_NEG_MED_PP, COOLDOWN_ROUNDS, ACTIVE_POLICY
    ACTIVE_POLICY = dict(policy)
    PROMOTE_PP = float(policy["quick_promote_pp"])
    PRUNE_PP = float(policy["quick_prune_pp"])
    STRONG_NEG_QUICK_PP = float(policy["strong_negative_quick_pp"])
    STRONG_NEG_MED_PP = float(policy["strong_negative_med_pp"])
    COOLDOWN_ROUNDS = int(policy["cooldown_rounds"])


def load_policy(policy_path: Optional[Path]) -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    if policy_path is not None and policy_path.exists():
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            policy.update(payload)
    apply_policy(policy)
    return policy


@dataclass
class Candidate:
    name: str
    args: List[str]


@dataclass
class Task:
    name: str
    seed: int
    novelty_tier: str
    trainer: str
    ds: str
    ds_cfg: str
    base_args: List[str]
    cands: List[Candidate]
    quick_budget: int
    quick_steps: int
    quick_eval_every: int
    quick_val_batches: int
    med_budget: int
    med_steps: int
    med_eval_every: int
    med_val_batches: int


@dataclass
class RoundSpec:
    round_id: int
    novelty_tier: str
    tasks: List[Task]


class HistoryIndex:
    def __init__(self) -> None:
        self.med_by_signature: Dict[str, dict] = {}
        self.med_by_signature_canon: Dict[str, dict] = {}
        self.cooldown_until: Dict[str, int] = {}
        self.cooldown_until_canon: Dict[str, int] = {}

    @staticmethod
    def canonical_signature(signature: str) -> str:
        """Canonical key that ignores cosmetic task naming differences."""
        try:
            payload = json.loads(signature)
        except Exception:
            return signature
        if not isinstance(payload, dict):
            return signature
        payload.pop("task", None)
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def update_cooldown(self, signature: str, until_round: int) -> None:
        prev = self.cooldown_until.get(signature, -1)
        if until_round > prev:
            self.cooldown_until[signature] = until_round
        canon = self.canonical_signature(signature)
        prev_c = self.cooldown_until_canon.get(canon, -1)
        if until_round > prev_c:
            self.cooldown_until_canon[canon] = until_round

    def get_cooldown(self, signature: str) -> Optional[int]:
        direct = self.cooldown_until.get(signature)
        canon = self.cooldown_until_canon.get(self.canonical_signature(signature))
        if direct is None:
            return canon
        if canon is None:
            return direct
        return max(direct, canon)

    def add_med_signature(self, signature: str, row: dict) -> None:
        prev = self.med_by_signature.get(signature)
        if prev is None or int(row.get("round", -1)) >= int(prev.get("round", -1)):
            self.med_by_signature[signature] = row
        canon = self.canonical_signature(signature)
        prev_c = self.med_by_signature_canon.get(canon)
        if prev_c is None or int(row.get("round", -1)) >= int(prev_c.get("round", -1)):
            self.med_by_signature_canon[canon] = row

    def has_med_signature(self, signature: str) -> bool:
        if signature in self.med_by_signature:
            return True
        return self.canonical_signature(signature) in self.med_by_signature_canon

    def get_med_signature(self, signature: str) -> Optional[dict]:
        row = self.med_by_signature.get(signature)
        if row is not None:
            return row
        return self.med_by_signature_canon.get(self.canonical_signature(signature))


def env() -> dict:
    e = os.environ.copy()
    e["TORCH_EXTENSIONS_DIR"] = os.environ.get("TORCH_EXTENSIONS_DIR", "/root/autodl-tmp/torch_extensions")
    e["HF_HOME"] = os.environ.get("HF_HOME", "/root/autodl-tmp/hf")
    e["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_datasets")
    e["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_transformers")
    e["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    # Default to offline mode, but allow round-level override via shell env.
    e["HF_DATASETS_OFFLINE"] = os.environ.get("HF_DATASETS_OFFLINE", "1")
    e["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
    e["PYTHONUNBUFFERED"] = "1"
    e["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    e["TORCH_CUDA_ARCH_LIST"] = "8.9"
    return e


def train_script(trainer_name: str) -> Path:
    p = ROOT / "scripts" / trainer_name
    if p.exists():
        return p
    p2 = ROOT / trainer_name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"trainer not found: {trainer_name}")


def extract_arg(base_args: List[str], key: str, default: str = "") -> str:
    for i, tok in enumerate(base_args):
        if tok == key and i + 1 < len(base_args):
            return str(base_args[i + 1])
    return default


def note_pool_size(base_args: List[str]) -> int:
    v = extract_arg(base_args, "--note_pool_size", "-1")
    try:
        return int(v)
    except Exception:
        return -1


def prompt_len(base_args: List[str]) -> int:
    v = extract_arg(base_args, "--max_prompt_tokens", "-1")
    try:
        return int(v)
    except Exception:
        return -1


def base_args_fingerprint(base_args: List[str]) -> str:
    payload = json.dumps(base_args, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def signature_for(task: Task, cfg_name: str) -> str:
    payload = {
        "task": task.name,
        "trainer": task.trainer,
        "seed": int(task.seed),
        "ds": task.ds,
        "ds_cfg": task.ds_cfg,
        "base_args_fp": base_args_fingerprint(task.base_args),
        "cfg_name": cfg_name,
        "quick_budget": int(task.quick_budget),
        "med_budget": int(task.med_budget),
        "note_pool": note_pool_size(task.base_args),
        "prompt_len": prompt_len(task.base_args),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def append_record(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_round_from_filename(name: str) -> Optional[int]:
    m = re.search(r"_round(\d+)_", name)
    if not m:
        return None
    return int(m.group(1))


def best_metrics(run_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return None, None
    best_loss = None
    best_acc = None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "val_loss" in r:
            v = float(r["val_loss"])
            best_loss = v if best_loss is None or v < best_loss else best_loss
        if "val_tok_acc" in r:
            v = float(r["val_tok_acc"])
            best_acc = v if best_acc is None or v > best_acc else best_acc
        elif "val_seq_acc" in r:
            # protein scripts often report sequence-level accuracy as auxiliary.
            v = float(r["val_seq_acc"])
            best_acc = v if best_acc is None or v > best_acc else best_acc
    return best_loss, best_acc


def should_promote(dpp: float) -> bool:
    return dpp >= PROMOTE_PP


def should_quick_prune(dpp: float) -> bool:
    return dpp < PRUNE_PP


def run_one(
    *,
    round_id: int,
    rec_path: Path,
    task: Task,
    stage: str,
    cfg: str,
    mode: str,
    budget: int,
    steps: int,
    eval_every: int,
    val_batches: int,
    extra: List[str],
    gate_reason: str,
    signature: str,
    cooldown_until_round: Optional[int],
    dry_run: bool,
) -> Optional[float]:
    log = RUNS / f"_round{round_id}_{task.name}_{stage}_{cfg}.log"
    cmd = [
        str(PYTHON),
        str(train_script(task.trainer)),
        "--train_data_seed",
        str(task.seed),
        "--val_data_seed",
        "1234",
        "--mode",
        mode,
        "--seed",
        str(task.seed),
        "--time_budget_sec",
        str(budget),
        "--max_steps",
        str(steps),
        "--eval_every",
        str(eval_every),
        "--val_batches",
        str(val_batches),
        "--model_lr",
        "3e-5",
        "--seed_scale",
        "1.0",
        *task.base_args,
        *extra,
    ]

    if dry_run:
        append_record(
            rec_path,
            {
                "round": round_id,
                "task": task.name,
                "novelty_tier": task.novelty_tier,
                "stage": stage,
                "config": cfg,
                "status": "dry_run",
                "best_val_loss": None,
                "best_val_tok_acc": None,
                "run_dir": "",
                "gate_reason": gate_reason,
                "signature": signature,
                "cooldown_until_round": cooldown_until_round,
                "cmd": cmd,
            },
        )
        return None

    with log.open("w", encoding="utf-8") as lf:
        p = subprocess.run(cmd, cwd=ROOT, env=env(), stdout=lf, stderr=subprocess.STDOUT)

    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    if p.returncode != 0:
        append_record(
            rec_path,
            {
                "round": round_id,
                "task": task.name,
                "novelty_tier": task.novelty_tier,
                "stage": stage,
                "config": cfg,
                "status": "fail",
                "best_val_loss": None,
                "best_val_tok_acc": None,
                "run_dir": "",
                "error": "rc_nonzero",
                "gate_reason": gate_reason,
                "signature": signature,
                "cooldown_until_round": cooldown_until_round,
            },
        )
        return None

    run_dir = lines[-1].strip() if lines else ""
    bl, ba = best_metrics(Path(run_dir)) if run_dir else (None, None)
    if ba is None:
        append_record(
            rec_path,
            {
                "round": round_id,
                "task": task.name,
                "novelty_tier": task.novelty_tier,
                "stage": stage,
                "config": cfg,
                "status": "fail",
                "best_val_loss": bl,
                "best_val_tok_acc": None,
                "run_dir": run_dir,
                "error": "no_val_metric",
                "gate_reason": gate_reason,
                "signature": signature,
                "cooldown_until_round": cooldown_until_round,
            },
        )
        return None

    append_record(
        rec_path,
        {
            "round": round_id,
            "task": task.name,
            "novelty_tier": task.novelty_tier,
            "stage": stage,
            "config": cfg,
            "status": "ok",
            "best_val_loss": bl,
            "best_val_tok_acc": ba,
            "run_dir": run_dir,
            "gate_reason": gate_reason,
            "signature": signature,
            "cooldown_until_round": cooldown_until_round,
        },
    )
    return ba


def load_history(runs_dir: Path) -> HistoryIndex:
    hist = HistoryIndex()
    files = sorted(runs_dir.glob("_round*_records.jsonl"))
    for fp in files:
        round_id = parse_round_from_filename(fp.name)
        if round_id is None:
            continue
        rows = []
        for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not ln.strip():
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue

        # direct signature-indexed med results.
        for r in rows:
            sig = r.get("signature")
            if not sig:
                continue
            cur_cd = r.get("cooldown_until_round")
            if isinstance(cur_cd, int):
                hist.update_cooldown(sig, cur_cd)
            if r.get("stage") == "med" and r.get("status") == "ok":
                rec = dict(r)
                rec["round"] = int(rec.get("round", round_id))
                hist.add_med_signature(sig, rec)

        # infer strong negatives from rows that already have signatures.
        by_task_stage: Dict[Tuple[str, str], Dict[str, float]] = {}
        for r in rows:
            if r.get("status") != "ok":
                continue
            if not r.get("signature"):
                continue
            st = str(r.get("stage"))
            if st not in {"quick", "med"}:
                continue
            task = str(r.get("task"))
            cfg = str(r.get("config"))
            key = (task, st)
            if cfg == "baseline":
                by_task_stage.setdefault(key, {})["baseline"] = float(r.get("best_val_tok_acc"))

        for r in rows:
            if r.get("status") != "ok":
                continue
            sig = r.get("signature")
            if not sig:
                continue
            st = str(r.get("stage"))
            if st not in {"quick", "med"}:
                continue
            cfg = str(r.get("config"))
            if cfg == "baseline":
                continue
            task = str(r.get("task"))
            base = by_task_stage.get((task, st), {}).get("baseline")
            if base is None:
                continue
            dpp = (float(r.get("best_val_tok_acc")) - base) * 100.0
            if (st == "quick" and dpp <= STRONG_NEG_QUICK_PP) or (st == "med" and dpp <= STRONG_NEG_MED_PP):
                hist.update_cooldown(sig, round_id + COOLDOWN_ROUNDS)

    return hist


def load_queue(queue_path: Path) -> List[RoundSpec]:
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    out: List[RoundSpec] = []
    for rd in payload["rounds"]:
        tasks = []
        for t in rd["tasks"]:
            cands = [Candidate(name=c["name"], args=list(c["args"])) for c in t["candidates"]]
            tasks.append(
                Task(
                    name=t["name"],
                    seed=int(t["seed"]),
                    novelty_tier=rd["novelty_tier"],
                    trainer=t["trainer"],
                    ds=t["ds"],
                    ds_cfg=t.get("ds_cfg", ""),
                    base_args=list(t["base_args"]),
                    cands=cands,
                    quick_budget=int(t["quick_budget"]),
                    quick_steps=int(t["quick_steps"]),
                    quick_eval_every=int(t["quick_eval_every"]),
                    quick_val_batches=int(t["quick_val_batches"]),
                    med_budget=int(t["med_budget"]),
                    med_steps=int(t["med_steps"]),
                    med_eval_every=int(t["med_eval_every"]),
                    med_val_batches=int(t["med_val_batches"]),
                )
            )
        out.append(RoundSpec(round_id=int(rd["round"]), novelty_tier=rd["novelty_tier"], tasks=tasks))
    return out


def run_task(round_id: int, rec_path: Path, task: Task, hist: HistoryIndex, dry_run: bool) -> None:
    # task-level fast skip if all candidate signatures already have med results.
    cand_sigs = [signature_for(task, c.name) for c in task.cands]
    if all(hist.has_med_signature(sig) for sig in cand_sigs):
        append_record(
            rec_path,
            {
                "round": round_id,
                "task": task.name,
                "novelty_tier": task.novelty_tier,
                "stage": "task_decision",
                "config": "all",
                "status": "skipped",
                "reason": "all_candidates_have_existing_med_signature",
                "gate_reason": "med_skip",
                "signature": "",
                "cooldown_until_round": None,
            },
        )
        return

    sig_base = signature_for(task, "baseline")
    base_quick = run_one(
        round_id=round_id,
        rec_path=rec_path,
        task=task,
        stage="quick",
        cfg="baseline",
        mode="no_fs",
        budget=task.quick_budget,
        steps=task.quick_steps,
        eval_every=task.quick_eval_every,
        val_batches=task.quick_val_batches,
        extra=[],
        gate_reason="promote",
        signature=sig_base,
        cooldown_until_round=hist.get_cooldown(sig_base),
        dry_run=dry_run,
    )
    if base_quick is None:
        append_record(
            rec_path,
            {
                "round": round_id,
                "task": task.name,
                "novelty_tier": task.novelty_tier,
                "stage": "med_decision",
                "config": "all",
                "status": "skipped",
                "reason": "quick_baseline_failed",
                "gate_reason": "med_skip",
                "signature": "",
                "cooldown_until_round": None,
            },
        )
        return

    scored: List[Tuple[Candidate, float, float, str]] = []
    for cand in task.cands:
        sig = signature_for(task, cand.name)
        cd = hist.get_cooldown(sig)
        if cd is not None and round_id <= cd:
            append_record(
                rec_path,
                {
                    "round": round_id,
                    "task": task.name,
                    "novelty_tier": task.novelty_tier,
                    "stage": "quick",
                    "config": cand.name,
                    "status": "skipped",
                    "reason": f"cooldown_active_until_round_{cd}",
                    "gate_reason": "med_skip",
                    "signature": sig,
                    "cooldown_until_round": cd,
                },
            )
            continue

        if hist.has_med_signature(sig):
            append_record(
                rec_path,
                {
                    "round": round_id,
                    "task": task.name,
                    "novelty_tier": task.novelty_tier,
                    "stage": "quick",
                    "config": cand.name,
                    "status": "skipped",
                    "reason": "existing_med_signature",
                    "gate_reason": "med_skip",
                    "signature": sig,
                    "cooldown_until_round": hist.get_cooldown(sig),
                },
            )
            continue

        acc = run_one(
            round_id=round_id,
            rec_path=rec_path,
            task=task,
            stage="quick",
            cfg=cand.name,
            mode="prompt_fs",
            budget=task.quick_budget,
            steps=task.quick_steps,
            eval_every=task.quick_eval_every,
            val_batches=task.quick_val_batches,
            extra=cand.args,
            gate_reason="promote",
            signature=sig,
            cooldown_until_round=hist.get_cooldown(sig),
            dry_run=dry_run,
        )
        if acc is None:
            continue

        dpp = (acc - base_quick) * 100.0
        scored.append((cand, acc, dpp, sig))

        if should_quick_prune(dpp):
            if dpp <= STRONG_NEG_QUICK_PP:
                hist.update_cooldown(sig, round_id + COOLDOWN_ROUNDS)
            append_record(
                rec_path,
                {
                    "round": round_id,
                    "task": task.name,
                    "novelty_tier": task.novelty_tier,
                    "stage": "quick_decision",
                    "config": cand.name,
                    "status": "pruned",
                    "reason": f"quick_drop {dpp:+.2f}pp < {PRUNE_PP:+.2f}pp",
                    "gate_reason": "quick_prune",
                    "signature": sig,
                    "cooldown_until_round": hist.get_cooldown(sig),
                },
            )

    promoted = [x for x in sorted(scored, key=lambda z: z[1], reverse=True) if should_promote(x[2])]
    promoted = promoted[:1]

    if not promoted:
        best = max(scored, key=lambda z: z[1], default=None)
        best_txt = "none" if best is None else f"{best[0].name} {best[2]:+.2f}pp"
        append_record(
            rec_path,
            {
                "round": round_id,
                "task": task.name,
                "novelty_tier": task.novelty_tier,
                "stage": "med_decision",
                "config": best[0].name if best else "none",
                "status": "pruned",
                "reason": f"best_quick {best_txt} < promote_gate {PROMOTE_PP:+.2f}pp",
                "gate_reason": "med_skip",
                "signature": best[3] if best else "",
                "cooldown_until_round": hist.get_cooldown(best[3]) if best else None,
            },
        )
        return

    # med baseline: reuse existing med if exact signature exists.
    base_med_sig = sig_base
    base_med_rec = hist.get_med_signature(base_med_sig)
    if base_med_rec is not None:
        base_med = float(base_med_rec["best_val_tok_acc"])
        append_record(
            rec_path,
            {
                "round": round_id,
                "task": task.name,
                "novelty_tier": task.novelty_tier,
                "stage": "med",
                "config": "baseline",
                "status": "skip_reuse",
                "best_val_loss": base_med_rec.get("best_val_loss"),
                "best_val_tok_acc": base_med,
                "run_dir": base_med_rec.get("run_dir", ""),
                "reason": "existing_med_signature",
                "gate_reason": "med_skip",
                "signature": base_med_sig,
                "cooldown_until_round": hist.get_cooldown(base_med_sig),
            },
        )
    else:
        b = run_one(
            round_id=round_id,
            rec_path=rec_path,
            task=task,
            stage="med",
            cfg="baseline",
            mode="no_fs",
            budget=task.med_budget,
            steps=task.med_steps,
            eval_every=task.med_eval_every,
            val_batches=task.med_val_batches,
            extra=[],
            gate_reason="promote",
            signature=base_med_sig,
            cooldown_until_round=hist.get_cooldown(base_med_sig),
            dry_run=dry_run,
        )
        if b is None:
            append_record(
                rec_path,
                {
                    "round": round_id,
                    "task": task.name,
                    "novelty_tier": task.novelty_tier,
                    "stage": "med_decision",
                    "config": "all",
                    "status": "skipped",
                    "reason": "med_baseline_failed",
                    "gate_reason": "med_skip",
                    "signature": "",
                    "cooldown_until_round": None,
                },
            )
            return
        base_med = b

    for cand, _qa, _qdpp, sig in promoted:
        cd = hist.get_cooldown(sig)
        if cd is not None and round_id <= cd:
            append_record(
                rec_path,
                {
                    "round": round_id,
                    "task": task.name,
                    "novelty_tier": task.novelty_tier,
                    "stage": "med",
                    "config": cand.name,
                    "status": "skipped",
                    "reason": f"cooldown_active_until_round_{cd}",
                    "gate_reason": "med_skip",
                    "signature": sig,
                    "cooldown_until_round": cd,
                },
            )
            continue

        sig_rec = hist.get_med_signature(sig)
        if sig_rec is not None:
            fs_med = float(sig_rec["best_val_tok_acc"])
            append_record(
                rec_path,
                {
                    "round": round_id,
                    "task": task.name,
                    "novelty_tier": task.novelty_tier,
                    "stage": "med",
                    "config": cand.name,
                    "status": "skip_reuse",
                    "best_val_loss": sig_rec.get("best_val_loss"),
                    "best_val_tok_acc": fs_med,
                    "run_dir": sig_rec.get("run_dir", ""),
                    "reason": "existing_med_signature",
                    "gate_reason": "med_skip",
                    "signature": sig,
                    "cooldown_until_round": hist.get_cooldown(sig),
                },
            )
        else:
            fs_med = run_one(
                round_id=round_id,
                rec_path=rec_path,
                task=task,
                stage="med",
                cfg=cand.name,
                mode="prompt_fs",
                budget=task.med_budget,
                steps=task.med_steps,
                eval_every=task.med_eval_every,
                val_batches=task.med_val_batches,
                extra=cand.args,
                gate_reason="promote",
                signature=sig,
                cooldown_until_round=hist.get_cooldown(sig),
                dry_run=dry_run,
            )
            if fs_med is None:
                continue

        dmed = (fs_med - base_med) * 100.0
        if dmed <= STRONG_NEG_MED_PP:
            hist.update_cooldown(sig, round_id + COOLDOWN_ROUNDS)
            append_record(
                rec_path,
                {
                    "round": round_id,
                    "task": task.name,
                    "novelty_tier": task.novelty_tier,
                    "stage": "med_decision",
                    "config": cand.name,
                    "status": "flagged",
                    "reason": f"strong_med_negative {dmed:+.2f}pp <= {STRONG_NEG_MED_PP:+.2f}pp",
                    "gate_reason": "med_skip",
                    "signature": sig,
                    "cooldown_until_round": hist.get_cooldown(sig),
                },
            )


def summarize_round(round_id: int, rec_path: Path, summary_path: Path, policy: Optional[Dict[str, Any]] = None) -> None:
    rows = []
    for line in rec_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    pol = policy or ACTIVE_POLICY
    out = ["=" * 112, f"Round{round_id} fastdiscover summary", "=" * 112]
    out.append(
        f"policy model={pol.get('model', 'gpt-5.3-codex')} "
        f"promote={PROMOTE_PP:+.2f}pp prune={PRUNE_PP:+.2f}pp "
        f"cooldown={COOLDOWN_ROUNDS}"
    )
    out.append("=" * 112)
    rank = []
    tasks = sorted({r["task"] for r in rows if "task" in r})

    for task in tasks:
        out.append(f"[{task}]")
        bq = [r for r in rows if r.get("task") == task and r.get("stage") == "quick" and r.get("config") == "baseline" and r.get("status") in {"ok", "skip_reuse"}]
        if not bq or bq[0].get("best_val_tok_acc") is None:
            out.append("  quick baseline failed")
            out.append("-" * 112)
            continue
        bq_acc = float(bq[0]["best_val_tok_acc"])
        out.append(f"  quick baseline: {bq_acc * 100:.2f}%")

        qfs = [
            r
            for r in rows
            if r.get("task") == task
            and r.get("stage") == "quick"
            and r.get("config") != "baseline"
            and r.get("status") in {"ok", "skip_reuse"}
            and r.get("best_val_tok_acc") is not None
        ]
        for r in sorted(qfs, key=lambda x: float(x["best_val_tok_acc"]), reverse=True):
            dpp = (float(r["best_val_tok_acc"]) - bq_acc) * 100.0
            out.append(f"    {r['config']:28s} d_acc={dpp:+.2f}pp acc={float(r['best_val_tok_acc']) * 100:.2f}%")

        mb = [r for r in rows if r.get("task") == task and r.get("stage") == "med" and r.get("config") == "baseline" and r.get("status") in {"ok", "skip_reuse"} and r.get("best_val_tok_acc") is not None]
        mf = [
            r
            for r in rows
            if r.get("task") == task
            and r.get("stage") == "med"
            and r.get("config") != "baseline"
            and r.get("status") in {"ok", "skip_reuse"}
            and r.get("best_val_tok_acc") is not None
        ]

        if mb and mf:
            bmed = float(mb[0]["best_val_tok_acc"])
            out.append("  med:")
            out.append(f"    baseline                     acc={bmed * 100:.2f}%")
            for r in sorted(mf, key=lambda x: float(x["best_val_tok_acc"]), reverse=True):
                fmed = float(r["best_val_tok_acc"])
                dmed = (fmed - bmed) * 100.0
                out.append(f"    {r['config']:28s} d_acc_vs_med={dmed:+.2f}pp acc={fmed * 100:.2f}%")
                rank.append((task, "med", r["config"], dmed))
        else:
            md = [r for r in rows if r.get("task") == task and r.get("stage") == "med_decision"]
            if md:
                out.append(f"  med skipped: {md[-1].get('reason', 'pruned')}")
            else:
                out.append("  med skipped")
            if qfs:
                best_q = max(qfs, key=lambda x: float(x["best_val_tok_acc"]))
                dq = (float(best_q["best_val_tok_acc"]) - bq_acc) * 100.0
                rank.append((task, "quick", best_q["config"], dq))

        pruned = [r for r in rows if r.get("task") == task and r.get("gate_reason") == "quick_prune"]
        skipped = [r for r in rows if r.get("task") == task and r.get("status") == "skipped"]
        if pruned:
            out.append("  pruned:")
            for p in pruned:
                out.append(f"    {p.get('config', '?'):28s} {p.get('reason', '')}")
        if skipped:
            out.append("  skipped:")
            for s in skipped:
                out.append(f"    {s.get('config', '?'):28s} {s.get('reason', '')}")
        out.append("-" * 112)

    out.append("[useful_task_ranking_this_round]")
    for t, st, cfg, d in sorted(rank, key=lambda x: x[3], reverse=True):
        out.append(f"  {t:32s} stage={st:5s} cfg={cfg:24s} d_acc={d:+.2f}pp")

    summary_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


def run_self_tests() -> None:
    assert should_promote(0.79) is False
    assert should_promote(0.80) is True
    assert should_quick_prune(-0.50) is False
    assert should_quick_prune(-0.51) is True

    h = HistoryIndex()
    sig = "sig-x"
    h.med_by_signature[sig] = {"best_val_tok_acc": 0.2, "round": 12}
    assert sig in h.med_by_signature
    h.update_cooldown(sig, 20)
    h.update_cooldown(sig, 18)
    assert h.cooldown_until[sig] == 20

    # canonical signature ignores "task" field so renamed tasks dedup.
    s1 = json.dumps({"task": "foo", "ds": "x", "seed": 1, "cfg_name": "a"}, sort_keys=True)
    s2 = json.dumps({"task": "bar", "ds": "x", "seed": 1, "cfg_name": "a"}, sort_keys=True)
    rec = {"round": 3, "best_val_tok_acc": 0.1}
    h2 = HistoryIndex()
    h2.add_med_signature(s1, rec)
    assert h2.has_med_signature(s2)
    h2.update_cooldown(s1, 9)
    assert h2.get_cooldown(s2) == 9

    print("self_test_ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=str, default=str(RESULTS / "_search_queue_round77_82.json"))
    ap.add_argument("--round_from", type=int, default=77)
    ap.add_argument("--round_to", type=int, default=82)
    ap.add_argument("--policy", type=str, default=str(RESULTS / "_codex53_team_policy.json"))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        run_self_tests()
        return

    RUNS.mkdir(parents=True, exist_ok=True)
    policy = load_policy(Path(args.policy))
    queue = load_queue(Path(args.queue))
    hist = load_history(RUNS)

    useful_pool_path = RUNS / "_useful_task_pool_fastdiscover.json"
    useful_pool: List[dict] = []
    if useful_pool_path.exists():
        try:
            useful_pool = json.loads(useful_pool_path.read_text(encoding="utf-8"))
        except Exception:
            useful_pool = []

    for rd in queue:
        if rd.round_id < args.round_from or rd.round_id > args.round_to:
            continue

        rec_path = RUNS / f"_round{rd.round_id}_fastdiscover_records.jsonl"
        summary_path = RUNS / f"_summary_round{rd.round_id}_fastdiscover.txt"
        rec_path.write_text("", encoding="utf-8")

        for task in rd.tasks:
            run_task(rd.round_id, rec_path, task, hist, dry_run=args.dry_run)

        summarize_round(rd.round_id, rec_path, summary_path, policy=policy)

        # refresh history index after each round so dedup/cooldown applies to subsequent rounds.
        hist = load_history(RUNS)

        # append useful pool entries from this round.
        rows = [json.loads(x) for x in rec_path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        by_task = {}
        for r in rows:
            by_task.setdefault(r.get("task"), []).append(r)
        for task, arr in by_task.items():
            mb = [r for r in arr if r.get("stage") == "med" and r.get("config") == "baseline" and r.get("status") in {"ok", "skip_reuse"} and r.get("best_val_tok_acc") is not None]
            mf = [r for r in arr if r.get("stage") == "med" and r.get("config") != "baseline" and r.get("status") in {"ok", "skip_reuse"} and r.get("best_val_tok_acc") is not None]
            if not mb or not mf:
                continue
            bmed = float(mb[0]["best_val_tok_acc"])
            best = max(mf, key=lambda x: float(x["best_val_tok_acc"]))
            dpp = (float(best["best_val_tok_acc"]) - bmed) * 100.0
            if dpp > 0:
                useful_pool.append(
                    {
                        "round": rd.round_id,
                        "task": task,
                        "config": best.get("config"),
                        "dpp": dpp,
                        "signature": best.get("signature"),
                    }
                )

        useful_pool_path.write_text(json.dumps(useful_pool, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
