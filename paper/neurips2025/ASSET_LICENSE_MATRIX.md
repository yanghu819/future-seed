# Asset and License Matrix

This table covers the paper-relevant assets referenced by the anonymous snapshot.
It is an audit aid, not a claim that every upstream term has been fully re-verified.
Where the current snapshot cannot support a precise license statement, the status is marked conservatively.

| Asset | Role in paper | Source / citation anchor | License / terms status in this snapshot | Included in anonymous ZIP? | Notes |
|---|---|---|---|---|---|
| `paper/neurips2025/neurips_2025.sty` and style bundle examples | official NeurIPS formatting assets | `https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip` | conference-distributed style bundle; no additional license note is reproduced in this snapshot | Yes | formatting only |
| `rwkv-diff-future-seed/` | synthetic-task code and logs used by the main mechanism evidence | local repository snapshot | no top-level standalone license file is shipped in the current repo snapshot | Yes | anonymous source snapshot only |
| `rwkv7-g1d-0.1b-20260129-ctx8192.pth` | default checkpoint referenced by the post-training scripts | internal checkpoint path referenced in scripts and appendix | not included; no license metadata shipped with the anonymous snapshot | No | this is why reproducibility/open-access checklist answers remain conservative |
| HotpotQA benchmark | source benchmark behind `hotpot_text_restore` | Yang et al. 2018, ACL Anthology: `https://aclanthology.org/D18-1259/` | benchmark license not re-audited in this snapshot | No raw data | only derived task formatting is described |
| SQuAD benchmark | source benchmark behind `squad_text_restore` | Rajpurkar et al. 2016, ACL Anthology: `https://aclanthology.org/D16-1264/` | benchmark license not re-audited in this snapshot | No raw data | only derived task formatting is described |
| ARC benchmark | source benchmark behind `arc_mc_probe` | Clark et al. 2018, arXiv: `https://arxiv.org/abs/1803.05457` | benchmark license not re-audited in this snapshot | No raw data | paper reports token-accuracy probe deltas, not leaderboard scores |
| MBPP benchmark | source benchmark behind `mbpp_longctx_probe` | Austin et al. 2021, arXiv: `https://arxiv.org/abs/2108.07732` | benchmark license not re-audited in this snapshot | No raw data | only probe formatting and shipped logs are included |
| `lamm-mit/protein_secondary_structure_from_PDB` | dataset identifier behind `protein_ss_spot` | exact dataset id appears in `train_protein_ss_spot_sft.py` and shipped signatures | license not re-audited in this anonymous snapshot | No raw data | the paper includes probe semantics and shipped run records, not the raw dataset cache |
| `data/metrics.json` and generated paper tables | curated paper-side metrics artifact | local repository snapshot | new asset authored for this paper package; no separate outbound license statement is attached here | Yes | documented by `METRICS_PROVENANCE.md` |

## Interpretation

Two checklist consequences follow directly from this matrix:

1. The paper can truthfully credit the upstream benchmarks and model families.
2. The paper should still answer the explicit license-coverage checklist item conservatively, because the current anonymous snapshot does not carry a fully verified per-asset license bundle for every upstream dataset, dependency, and checkpoint used during the broader campaign.
