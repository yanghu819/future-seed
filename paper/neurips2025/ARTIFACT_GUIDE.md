# Anonymous Artifact Guide

Intended supplementary contents for the submission:

- `rwkv-diff-future-seed/`: synthetic task code and launch scripts
- `posttrain_rwkv7/scripts/`: post-training trainers and orchestrators
- `posttrain_rwkv7/results/_search_queue_round783_790_realtask_exploit_v3.json`
- `posttrain_rwkv7/results/_search_queue_round799_802_final_confirm.json`
- `posttrain_rwkv7/results/_search_queue_round803_804_mbpp_altconfirm.json`
- `posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json`
- `posttrain_rwkv7/runs/_summary_round783_fastdiscover.txt` ... `_summary_round808_fastdiscover.txt`
- `posttrain_rwkv7/runs/_round783_fastdiscover_records.jsonl` ... `_round808_fastdiscover_records.jsonl`
- `paper/neurips2025/`: the paper package itself

Minimal reproduction steps:

```bash
cd posttrain_rwkv7
python3 scripts/run_round77_82_fastdiscover.py --self_test
python3 scripts/run_round77_82_fastdiscover.py \
  --queue results/_search_queue_round805_808_breadth_roi.json \
  --round_from 805 --round_to 808 --dry_run
```

The paper tables are regenerated with:

```bash
cd paper/neurips2025
python3 render_tables.py
```
