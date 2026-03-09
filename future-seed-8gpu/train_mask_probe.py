#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from future_seed_hf import inject_future_seed, mark_future_seed_trainable


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]


class JsonlRows(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class ExampleDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {"input_ids": ex.input_ids, "labels": ex.labels}


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def distributed_init() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def get_decoder(model):
    base = unwrap_model(model)
    if hasattr(base, "get_decoder"):
        return base.get_decoder()
    if hasattr(base, "model"):
        return base.model
    raise AttributeError("cannot locate decoder")


def get_output_head(model):
    base = unwrap_model(model)
    if hasattr(base, "get_output_embeddings"):
        return base.get_output_embeddings()
    if hasattr(base, "lm_head"):
        return base.lm_head
    raise AttributeError("cannot locate output head")


def mask_embedding_only(model, mask_token_id: int):
    base = unwrap_model(model)
    emb = base.get_input_embeddings().weight
    emb.requires_grad_(True)
    def hook(grad: torch.Tensor):
        keep = torch.zeros_like(grad)
        keep[mask_token_id] = 1
        return grad * keep
    emb.register_hook(hook)


def choose_span(n_tokens: int, rng: random.Random, min_span: int, max_span: int) -> tuple[int, int] | None:
    if n_tokens < min_span + 4:
        return None
    span = min(max(min_span, n_tokens // 5), max_span)
    span = min(span, n_tokens - 2)
    if span < min_span:
        return None
    lo = max(1, n_tokens // 6)
    hi = max(lo + 1, n_tokens - span - 1)
    start = rng.randint(lo, hi - 1)
    return start, start + span


def build_mbpp_examples(rows: Iterable[dict], tokenizer, mask_id: int, max_length: int, seed: int) -> list[Example]:
    out = []
    rng = random.Random(seed)
    for row in rows:
        prefix_ids = tokenizer(f"Problem:\n{row['prompt'].strip()}\n\nCode:\n", add_special_tokens=False).input_ids
        code_ids = tokenizer(row['code'], add_special_tokens=False).input_ids
        span = choose_span(len(code_ids), rng, min_span=4, max_span=32)
        if span is None:
            continue
        s, e = span
        mask_len = e - s
        input_ids = prefix_ids + code_ids[:s] + [mask_id] * mask_len + code_ids[e:]
        labels = [-100] * len(input_ids)
        offset = len(prefix_ids) + s
        for i, tok in enumerate(code_ids[s:e]):
            labels[offset + i] = tok
        if len(input_ids) + 1 > max_length:
            continue
        input_ids = input_ids[:max_length - 1] + [tokenizer.eos_token_id]
        labels = labels[:max_length - 1] + [-100]
        out.append(Example(input_ids=input_ids, labels=labels))
    return out


def single_token_id(text: str, tokenizer) -> int | None:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) == 1:
        return ids[0]
    ids = tokenizer(" " + text, add_special_tokens=False).input_ids
    if len(ids) == 1:
        return ids[0]
    return None


def build_arc_examples(rows: Iterable[dict], tokenizer, mask_id: int, max_length: int) -> list[Example]:
    out = []
    for row in rows:
        answer_id = single_token_id(str(row['answer']).strip(), tokenizer)
        if answer_id is None:
            continue
        options = "\n".join([f"{c['label']}. {c['text']}" for c in row['choices']])
        prefix = f"Question:\n{row['question'].strip()}\nAnswer:"
        suffix = f"\nOptions:\n{options}\n"
        input_ids = tokenizer(prefix, add_special_tokens=False).input_ids + [mask_id] + tokenizer(suffix, add_special_tokens=False).input_ids
        if len(input_ids) + 1 > max_length:
            continue
        labels = [-100] * len(input_ids)
        labels[len(tokenizer(prefix, add_special_tokens=False).input_ids)] = answer_id
        input_ids = input_ids[:max_length - 1] + [tokenizer.eos_token_id]
        labels = labels[:max_length - 1] + [-100]
        out.append(Example(input_ids=input_ids, labels=labels))
    return out


def build_race_examples(rows: Iterable[dict], tokenizer, mask_id: int, max_length: int) -> list[Example]:
    out = []
    for row in rows:
        answer_id = single_token_id(str(row['answer']).strip(), tokenizer)
        if answer_id is None:
            continue
        options = "\n".join([f"{c['label']}. {c['text']}" for c in row['choices']])
        prefix = f"Article:\n{row['article'].strip()}\n\nQuestion:\n{row['question'].strip()}\nAnswer:"
        suffix = f"\nOptions:\n{options}\n"
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        input_ids = prefix_ids + [mask_id] + tokenizer(suffix, add_special_tokens=False).input_ids
        if len(input_ids) + 1 > max_length:
            continue
        labels = [-100] * len(input_ids)
        labels[len(prefix_ids)] = answer_id
        input_ids = input_ids[:max_length - 1] + [tokenizer.eos_token_id]
        labels = labels[:max_length - 1] + [-100]
        out.append(Example(input_ids=input_ids, labels=labels))
    return out


BUILDERS = {
    "mbpp_mask": build_mbpp_examples,
    "arc_mask": build_arc_examples,
    "race_mask": build_race_examples,
}


def collate(batch: list[dict], pad_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        attention_mask.append([1] * len(x["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, rank: int):
    model.eval()
    tok_correct = torch.tensor(0.0, device=device)
    tok_total = torch.tensor(0.0, device=device)
    ex_correct = torch.tensor(0.0, device=device)
    ex_total = torch.tensor(0.0, device=device)
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        decoder = get_decoder(model)
        out = decoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        hidden = out.last_hidden_state
        lm_head = get_output_head(model)
        logits = lm_head(hidden)
        preds = logits.argmax(dim=-1)
        mask = batch["labels"].ne(-100)
        tok_correct += (preds.eq(batch["labels"]) & mask).sum()
        tok_total += mask.sum()
        ex_correct += ((preds.eq(batch["labels"]) | ~mask).all(dim=-1)).sum()
        ex_total += torch.tensor(batch["labels"].shape[0], device=device, dtype=torch.float32)
    tok_correct = reduce_sum(tok_correct)
    tok_total = reduce_sum(tok_total)
    ex_correct = reduce_sum(ex_correct)
    ex_total = reduce_sum(ex_total)
    if tok_total.item() == 0:
        return {"mask_token_acc": 0.0, "mask_exact": 0.0}
    return {
        "mask_token_acc": (tok_correct / tok_total).item(),
        "mask_exact": (ex_correct / ex_total).item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(BUILDERS.keys()), required=True)
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--model_local_dir", type=str, default="")
    ap.add_argument("--train_jsonl", type=Path, required=True)
    ap.add_argument("--eval_jsonl", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--future_seed", action="store_true")
    ap.add_argument("--future_seed_layer_start", type=int, default=1)
    ap.add_argument("--future_seed_alpha_init", type=float, default=-2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--train_limit", type=int, default=2048)
    ap.add_argument("--eval_limit", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    rank, world_size, local_rank = distributed_init()
    set_seed(args.seed + rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if is_main(rank):
        print(json.dumps({
            "event": "start",
            "task": args.task,
            "future_seed": int(args.future_seed),
            "model_id": args.model_id,
            "world_size": world_size,
            "max_steps": args.max_steps,
        }))

    tokenizer_source = args.model_local_dir or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    mask_token = "<fs_mask>"
    if mask_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    quant_cfg = None
    model_kwargs = {"trust_remote_code": True}
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs["device_map"] = {"": local_rank}
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16

    model_source = args.model_local_dir or args.model_id
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    if args.future_seed:
        inject_future_seed(model.get_decoder(), layer_start=args.future_seed_layer_start, alpha_init=args.future_seed_alpha_init)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
    )
    model = get_peft_model(model, lora_config)
    if args.future_seed:
        mark_future_seed_trainable(model)
    mask_embedding_only(model, mask_id)

    train_rows = load_jsonl(args.train_jsonl, args.train_limit)
    eval_rows = load_jsonl(args.eval_jsonl, args.eval_limit)
    train_examples = BUILDERS[args.task](train_rows, tokenizer, mask_id, args.max_length, args.seed)
    eval_examples = BUILDERS[args.task](eval_rows, tokenizer, mask_id, args.max_length, args.seed + 1)
    if not train_examples or not eval_examples:
        raise RuntimeError("empty_examples_after_build")

    if is_main(rank):
        print(json.dumps({
            "event": "data_ready",
            "task": args.task,
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "mask_token_id": mask_id,
        }))

    train_ds = ExampleDataset(train_examples)
    eval_ds = ExampleDataset(eval_examples)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=train_sampler is None, collate_fn=lambda b: collate(b, tokenizer.pad_token_id), drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler, shuffle=False, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    if not args.load_in_4bit:
        model = model.to(device)
    if world_size > 1 and not args.load_in_4bit:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = None
    steps = 0
    best = None
    t0 = time.time()

    train_iter = iter(train_loader)
    while steps < args.max_steps:
        model.train()
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                if train_sampler is not None:
                    train_sampler.set_epoch(steps + args.seed)
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available() and args.bf16 and not args.load_in_4bit):
                decoder = get_decoder(model)
                outputs = decoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state
                logits = get_output_head(model)(hidden)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1), ignore_index=-100)
                loss = loss / args.grad_accum
            loss.backward()
        opt.step()
        steps += 1

        if steps == 1 or steps % args.eval_every == 0 or steps == args.max_steps:
            metrics = evaluate(model, eval_loader, device, rank)
            metrics.update({"step": steps, "elapsed_sec": round(time.time() - t0, 1)})
            if is_main(rank):
                print(json.dumps({"event": "eval", **metrics}))
            if best is None or metrics["mask_exact"] > best["mask_exact"]:
                best = metrics
                if is_main(rank):
                    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
                        json.dump({
                            "task": args.task,
                            "future_seed": int(args.future_seed),
                            "model_id": args.model_id,
                            "train_examples": len(train_examples),
                            "eval_examples": len(eval_examples),
                            "best": best,
                        }, f, ensure_ascii=False, indent=2)

    if is_main(rank):
        summary = {
            "task": args.task,
            "future_seed": int(args.future_seed),
            "model_id": args.model_id,
            "best_mask_token_acc": round(best["mask_token_acc"], 6) if best else 0.0,
            "best_mask_exact": round(best["mask_exact"], 6) if best else 0.0,
            "output_dir": str(args.output_dir),
        }
        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps({"event": "done", **summary}))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
