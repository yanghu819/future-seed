# FS High-Value Task Queue (Rapid Screening)

## Queue Policy
- Single GPU serial runs.
- Quick stage first; med stage only if quick gain passes prune.
- Prune rule: `d_acc >= +0.50pp` for punc tasks, `d_acc >= +0.30pp` for longctx tasks.
- Promote at most top-2 configs per task.

## P0 (Running)
- `punc_hotpot` (real text repair)
- `punc_mbpp` (real code-text repair)
- `punc_squad` (real QA-text repair)
- Script: `scripts/run_round40_highvalue_queue_s0.py`

## P1 (Running after P0 in same script)
- `hotpot_longctx_qafter` (answer after long context)
- `mbpp_longctx_qafter` (code answer after long prompt)
- Script: `scripts/run_round40_highvalue_queue_s0.py`

## P2 (Blocked by network, not model)
- `punc_wikitext`
- `punc_agnews`
- Blocker: intermittent HF connect error (`httpx.ConnectError [Errno 99]`).

## Next Auto Actions
- Take Round40 winners only.
- Run quick seed-check (`s0/s1/s2`) on winners.
- Keep only scenes with positive mean and >=2/3 positive seeds.
