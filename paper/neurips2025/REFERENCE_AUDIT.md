# Reference Audit

This file audits every citation currently used in `main.tex`.

Policy:
- only citations that are actually cited in the paper are listed here
- each entry points to an official venue page, publisher page, DOI page, or arXiv record
- each entry states exactly what role the citation plays in the paper
- citations are used for background, contrast, or task provenance only; no unsupported factual claim should rely on a vague cluster of references

Audit date: `2026-03-10`

## Summary Table

| Key | Role in this paper | Primary source |
|---|---|---|
| `vaswani2017attention` | Transformer baseline family / architectural contrast | https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need |
| `devlin2019bert` | canonical bidirectional masked model | https://aclanthology.org/N19-1423/ |
| `peng2023rwkv` | recurrent backbone family context | https://arxiv.org/abs/2305.13048 |
| `gu2024mamba` | state-space family context | https://arxiv.org/abs/2312.00752 |
| `raffel2020t5` | span corruption / text-to-text transfer background | https://jmlr.org/papers/v21/20-074.html |
| `lee2018iterative` | iterative refinement background | https://aclanthology.org/D18-1149/ |
| `ghazvininejad2019maskpredict` | masked parallel decoding background | https://aclanthology.org/D19-1633/ |
| `gu2019levenshtein` | edit-based non-autoregressive decoding background | https://proceedings.neurips.cc/paper/2019/hash/675f9820626f5bc0afb47b57890b466e-Abstract.html |
| `chang2022maskgit` | iterative masked generation background | https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html |
| `yang2018hotpotqa` | source benchmark behind Hotpot-derived restoration probes | https://aclanthology.org/D18-1259/ |
| `clark2018arc` | source benchmark behind ARC-derived probes | https://arxiv.org/abs/1803.05457 |
| `rajpurkar2016squad` | source benchmark behind SQuAD-derived restoration probes | https://aclanthology.org/D16-1264/ |
| `austin2021program` | source paper for MBPP-style code probes | https://arxiv.org/abs/2108.07732 |
| `rao2019tape` | protein transfer benchmark context | https://proceedings.neurips.cc/paper_files/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html |
| `rives2021biological` | large protein language model context | https://doi.org/10.1073/pnas.2016239118 |

## Detailed Per-Citation Notes

### `vaswani2017attention`
- Canonical title: `Attention Is All You Need`
- Why cited here: to name the Transformer family as the main architectural contrast to recurrent/state-space models in the introduction.
- What claim it supports in our paper: that autoregressive Transformer-style sequence modeling is a standard alternative to recurrent processing.
- What it does **not** support here: it is not cited as evidence for Future-Seed itself, or for any claim about constrained repair.
- Primary source: https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need

### `devlin2019bert`
- Canonical title: `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`
- Why cited here: to represent bidirectional masked modeling as the canonical non-causal/bidirectional baseline family.
- What claim it supports in our paper: that masked bidirectional models provide a different solution to future-context access than recurrent left-to-right scanning.
- What it does **not** support here: it is not used as direct evidence for our synthetic or post-training results.
- Primary source: https://aclanthology.org/N19-1423/

### `peng2023rwkv`
- Canonical title: `RWKV: Reinventing RNNs for the Transformer Era`
- Why cited here: to place Future-Seed inside the recurrent RWKV-style model family.
- What claim it supports in our paper: that recurrent architectures remain relevant for long-context and streaming-friendly settings, and that Future-Seed is a local modification to this family.
- What it does **not** support here: it does not justify any improvement claims for Future-Seed.
- Primary source: https://arxiv.org/abs/2305.13048

### `gu2024mamba`
- Canonical title: `Mamba: Linear-Time Sequence Modeling with Selective State Spaces`
- Why cited here: to broaden the state-space/recurrent context beyond RWKV.
- What claim it supports in our paper: that compact-state sequence models are an active, serious modeling direction, so improving prompt-side future reuse in such models is relevant.
- What it does **not** support here: no direct experimental comparison to Mamba is claimed.
- Primary source: https://arxiv.org/abs/2312.00752

### `raffel2020t5`
- Canonical title: `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer`
- Why cited here: to ground span corruption / text infill style training as prior art for non-left-to-right editing.
- What claim it supports in our paper: that span corruption is a natural background regime for discussing in-place repair and masked restoration.
- What it does **not** support here: it is not presented as a recurrent mechanism or as evidence for prompt-time state feedback.
- Primary source: https://jmlr.org/papers/v21/20-074.html

### `lee2018iterative`
- Canonical title: `Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement`
- Why cited here: to cover iterative refinement as a line of work related to repeated repair.
- What claim it supports in our paper: that refinement-style decoding is an established alternative to one-shot left-to-right generation.
- What it does **not** support here: it is not cited to claim that Future-Seed is itself an iterative decoder.
- Primary source: https://aclanthology.org/D18-1149/

### `ghazvininejad2019maskpredict`
- Canonical title: `Mask-Predict: Parallel Decoding of Conditional Masked Language Models`
- Why cited here: to represent masked parallel decoding in the related-work cluster.
- What claim it supports in our paper: that masked parallel editing/decoding is a well-established design point for sequence generation and repair.
- What it does **not** support here: it is not used as a baseline result in our experiments.
- Primary source: https://aclanthology.org/D19-1633/

### `gu2019levenshtein`
- Canonical title: `Levenshtein Transformer`
- Why cited here: to represent edit-based non-autoregressive generation.
- What claim it supports in our paper: that insertion/deletion style editing is part of the broader design space surrounding in-place repair.
- What it does **not** support here: it does not directly justify our prompt-time state feedback mechanism.
- Primary source: https://proceedings.neurips.cc/paper/2019/hash/675f9820626f5bc0afb47b57890b466e-Abstract.html

### `chang2022maskgit`
- Canonical title: `MaskGIT: Masked Generative Image Transformer`
- Why cited here: to show that iterative masked generation extends beyond text into a broader masked-editing family.
- What claim it supports in our paper: that repeated masked prediction is a recognized paradigm for global-consistency generation.
- What it does **not** support here: it is not used for any claim about recurrent language models specifically.
- Primary source: https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html

### `yang2018hotpotqa`
- Canonical title: `HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering`
- Why cited here: to document the origin of the Hotpot-derived restoration probes in the post-training archive.
- What claim it supports in our paper: that the underlying source benchmark is multi-hop QA with distributed evidence, which motivates restoration-style probe construction.
- What it does **not** support here: it is not cited as evidence that our Hotpot-derived probe preserves official HotpotQA evaluation semantics.
- Primary source: https://aclanthology.org/D18-1259/

### `clark2018arc`
- Canonical title: `Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge`
- Why cited here: to document the origin of ARC-derived multiple-choice probes.
- What claim it supports in our paper: that ARC provides a reasoning-style QA source from which sparse answer probes can be constructed.
- What it does **not** support here: it is not used to claim benchmark-level ARC performance.
- Primary source: https://arxiv.org/abs/1803.05457

### `rajpurkar2016squad`
- Canonical title: `SQuAD: 100,000+ Questions for Machine Comprehension of Text`
- Why cited here: to document the source benchmark behind the SQuAD-derived restoration probes.
- What claim it supports in our paper: that SQuAD is a span-based reading-comprehension benchmark whose evidence distribution motivates restoration probes.
- What it does **not** support here: it is not used to claim EM/F1 benchmark gains.
- Primary source: https://aclanthology.org/D16-1264/

### `austin2021program`
- Canonical title: `Program Synthesis with Large Language Models`
- Why cited here: to ground the MBPP-style code probe family in its source benchmark paper.
- What claim it supports in our paper: that MBPP is an established source for Python programming tasks and therefore a reasonable basis for long-context code probes.
- What it does **not** support here: it is not used to claim pass@k or benchmark-level program-synthesis improvements.
- Primary source: https://arxiv.org/abs/2108.07732

### `rao2019tape`
- Canonical title: `Evaluating Protein Transfer Learning with TAPE`
- Why cited here: to ground the protein-task discussion in a standard protein transfer benchmark line.
- What claim it supports in our paper: that protein-sequence evaluation is a meaningful context for probe-style sequence prediction and transfer analysis.
- What it does **not** support here: it is not the exact same dataset used by the `protein_ss_spot` probe row; it is cited as benchmark context, not as a claimed one-to-one dataset identity.
- Primary source: https://proceedings.neurips.cc/paper_files/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html

### `rives2021biological`
- Canonical title: `Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences`
- Why cited here: to ground the claim that bidirectional or large-context protein language modeling is an important and credible real-task regime.
- What claim it supports in our paper: that protein sequence modeling is a legitimate context in which richer sequence context can matter.
- What it does **not** support here: it is not cited as evidence that our small probe equals large-scale biological structure prediction.
- Primary source: https://doi.org/10.1073/pnas.2016239118

## Update Rule

When adding a new citation:

1. add the citation to `main.tex` first
2. add the BibTeX entry to `references.bib`
3. add the official source link here
4. add a short note explaining exactly what sentence or argument the citation supports
5. rebuild with `./build.sh submission`
