# Complete Audit (round77-569)

- Scope: all `fastdiscover` JSONL records in `results/_round*_fastdiscover_records.jsonl` from round 77 to 569.
- Files parsed: 259
- Quick rows: 495
- Med rows: 237
- Quick prune decisions: 477
- Med skip decisions: 253

- Quick mean/median: +1.48pp / +0.76pp
- Quick positive rate: 63.0%
- Med mean/median: +1.74pp / +1.19pp
- Med positive rate: 69.6%

## Family Stats

| Family | Quick n | Quick mean(pp) | Quick pos% | Med n | Med mean(pp) | Med pos% |
|---|---:|---:|---:|---:|---:|---:|
| arc_mc | 90 | +2.73 | 52.2% | 47 | +3.72 | 55.3% |
| hotpot | 60 | +1.32 | 63.3% | 30 | +0.86 | 73.3% |
| mbpp | 85 | +1.02 | 77.6% | 46 | +0.38 | 63.0% |
| mbpp_longctx | 40 | +2.18 | 77.5% | 25 | +1.32 | 52.0% |
| other | 43 | +0.76 | 72.1% | 19 | +0.65 | 73.7% |
| protein_contact | 32 | +0.00 | 0.0% | 0 | +0.00 | 0.0% |
| protein_ss | 114 | +1.58 | 66.7% | 60 | +2.40 | 88.3% |
| squad | 31 | +0.57 | 74.2% | 10 | +0.45 | 80.0% |

## Top 25 Positive Med Runs

- round157 `arc_mc_seed13_discovery` / `scalar_l8_train8e5`: **+20.83pp**
- round163 `arc_mc_seed16_discovery` / `scalar_l8_sched_cos`: **+16.67pp**
- round136 `arc_mc_seed3_discovery` / `scalar_l8_sched_cos`: **+12.50pp**
- round405 `arc_mc_seed115_discovery` / `scalar_l8_train1e4`: **+12.50pp**
- round417 `arc_mc_seed120_discovery` / `scalar_l8_sched_cos`: **+12.50pp**
- round222 `mbpp_longctx_seed13_repair` / `scalar_l8_train8e5`: **+10.00pp**
- round534 `mbpp_longctx_seed106_repair` / `head_l8`: **+8.69pp**
- round160 `arc_mc_seed14_discovery` / `scalar_l8_train1e4`: **+8.33pp**
- round171 `arc_mc_seed20_discovery` / `scalar_l8_train1e4`: **+8.33pp**
- round193 `arc_mc_seed31_discovery` / `scalar_l8_sched_cos`: **+8.33pp**
- round209 `arc_mc_seed39_discovery` / `scalar_l8_train1e4`: **+8.33pp**
- round425 `arc_mc_seed122_discovery` / `scalar_l8_sched_cos`: **+8.33pp**
- round557 `arc_mc_seed205_discovery` / `scalar_l8_train8e5`: **+8.33pp**
- round569 `arc_mc_seed211_repro` / `scalar_l8_train1e4`: **+8.33pp**
- round168 `arc_mc_seed18_discovery` / `scalar_l8_train1e4`: **+8.33pp**
- round195 `arc_mc_seed32_discovery` / `scalar_l8_sched_cos`: **+8.33pp**
- round528 `protein_ss_seed261_discovery` / `scalar_l8_sched_cos`: **+8.14pp**
- round507 `mbpp_longctx_seed99_repair` / `head_l8`: **+7.44pp**
- round466 `protein_ss_seed239_discovery` / `head_l8`: **+7.33pp**
- round158 `protein_ss_seed25_discovery` / `scalar_l8_train8e5`: **+6.04pp**
- round121 `protein_ss_seed10_discovery` / `scalar_l8_train8e5`: **+5.96pp**
- round526 `mbpp_longctx_seed104_repair` / `head_l8`: **+5.96pp**
- round227 `protein_ss_seed60_discovery` / `head_l8`: **+5.42pp**
- round522 `protein_ss_seed259_discovery` / `scalar_l8_sched_cos`: **+4.98pp**
- round99 `protein_ss_seed4_discovery` / `scalar_l8_train8e5`: **+4.90pp**

## Top 25 Negative Med Runs

- round457 `arc_mc_seed167_discovery` / `scalar_l8_train1e4`: **-12.50pp**
- round116 `mbpp_longctx_seed0_repair` / `scalar_l8_train1e4`: **-6.45pp**
- round172 `mbpp_seed25_anchor` / `scalar_l8_train8e5`: **-4.68pp**
- round197 `arc_mc_seed33_discovery` / `scalar_l8_sched_cos`: **-4.17pp**
- round513 `arc_mc_seed187_discovery` / `scalar_l8_train1e4`: **-4.17pp**
- round189 `arc_mc_seed29_discovery` / `scalar_l8_train1e4`: **-4.17pp**
- round502 `mbpp_longctx_seed98_repair` / `scalar_l8_train1e4`: **-3.89pp**
- round517 `mbpp_seed145_anchor` / `scalar_l8_train1e4`: **-3.28pp**
- round446 `mbpp_seed101_headprobe` / `scalar_l8_train1e4`: **-2.74pp**
- round91 `hotpot_seed2_discovery` / `scalar_l8_train8e5`: **-2.74pp**
- round446 `protein_ss_seed190_discovery` / `scalar_l8_sched_cos`: **-2.26pp**
- round114 `arc_seed7_discovery` / `scalar_l8_train1e4`: **-2.17pp**
- round196 `mbpp_seed34_anchor` / `head_l10_strong`: **-2.16pp**
- round431 `squad_seed55_anchor` / `scalar_l8_sched_cos`: **-2.13pp**
- round568 `protein_ss_seed279_discovery` / `head_l8`: **-2.11pp**
- round205 `hotpot_seed38_discovery` / `head_l8`: **-2.09pp**
- round433 `protein_ss_seed184_discovery` / `scalar_l8_train1e4`: **-2.06pp**
- round402 `mbpp_seed90_headprobe` / `scalar_l8_sched_cos`: **-1.85pp**
- round183 `mbpp_seed29_headprobe` / `scalar_l8_train8e5`: **-1.85pp**
- round556 `protein_ss_seed273_discovery` / `scalar_l8_sched_cos`: **-1.85pp**
- round430 `mbpp_seed97_headprobe` / `scalar_l8_train1e4`: **-1.71pp**
- round427 `hotpot_seed70_discovery` / `scalar_l8_sched_cos`: **-1.69pp**
- round190 `mbpp_longctx_seed9_repair` / `scalar_l8_train8e5`: **-1.59pp**
- round185 `protein_ss_seed39_discovery` / `scalar_l8_train8e5`: **-1.56pp**
- round95 `mbpp_seed1_anchor` / `scalar_l8_train8e5`: **-1.51pp**

## Top 25 Quick→Med Reversals (quick>0, med<=0)

- round457 `arc_mc_seed167_discovery`: quick +4.17pp -> med -12.50pp
- round433 `arc_mc_seed124_discovery`: quick +12.50pp -> med +0.00pp
- round513 `arc_mc_seed187_discovery`: quick +8.33pp -> med -4.17pp
- round543 `arc_mc_seed198_discovery`: quick +12.50pp -> med +0.00pp
- round189 `arc_mc_seed29_discovery`: quick +8.33pp -> med -4.17pp
- round116 `mbpp_longctx_seed0_repair`: quick +2.37pp -> med -6.45pp
- round155 `arc_mc_seed12_discovery`: quick +8.33pp -> med +0.00pp
- round181 `arc_mc_seed25_discovery`: quick +8.33pp -> med +0.00pp
- round429 `arc_mc_seed123_discovery`: quick +8.33pp -> med +0.00pp
- round197 `arc_mc_seed33_discovery`: quick +4.17pp -> med -4.17pp
- round205 `arc_mc_seed37_discovery`: quick +8.33pp -> med +0.00pp
- round416 `arc_mc_seed119_discovery`: quick +8.33pp -> med +0.00pp
- round511 `arc_mc_seed186_discovery`: quick +8.33pp -> med +0.00pp
- round190 `mbpp_longctx_seed9_repair`: quick +5.75pp -> med -1.59pp
- round225 `protein_ss_seed59_discovery`: quick +6.29pp -> med -0.95pp
- round158 `mbpp_longctx_seed5_repair`: quick +5.03pp -> med -1.15pp
- round172 `mbpp_seed25_anchor`: quick +1.40pp -> med -4.68pp
- round502 `mbpp_longctx_seed98_repair`: quick +2.09pp -> med -3.89pp
- round446 `protein_ss_seed190_discovery`: quick +3.58pp -> med -2.26pp
- round185 `protein_ss_seed39_discovery`: quick +4.14pp -> med -1.56pp
- round166 `mbpp_longctx_seed6_repair`: quick +4.40pp -> med -1.14pp
- round456 `mbpp_longctx_seed86_repair`: quick +4.04pp -> med -1.46pp
- round568 `mbpp_longctx_seed117_repair`: quick +5.34pp -> med -0.02pp
- round205 `hotpot_seed38_discovery`: quick +3.06pp -> med -2.09pp
- round167 `mbpp_seed23_headprobe`: quick +3.73pp -> med -1.32pp
