# Reference Audit

This file audits the citations currently used in `main.tex`.

Policy:
- only citations that are actually used in the paper are listed here
- each entry points to an official venue page, publisher page, or arXiv record
- `references.bib` should not contain speculative or unverified entries

Audit date: `2026-03-07`

| Key | Title | Official source |
|---|---|---|
| `vaswani2017attention` | Attention Is All You Need | https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need |
| `devlin2019bert` | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | https://aclanthology.org/N19-1423/ |
| `peng2023rwkv` | RWKV: Reinventing RNNs for the Transformer Era | https://arxiv.org/abs/2305.13048 |
| `gu2024mamba` | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | https://arxiv.org/abs/2312.00752 |
| `raffel2020t5` | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | https://jmlr.org/papers/v21/20-074.html |
| `lee2018iterative` | Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement | https://aclanthology.org/D18-1149/ |
| `ghazvininejad2019maskpredict` | Mask-Predict: Parallel Decoding of Conditional Masked Language Models | https://aclanthology.org/D19-1633/ |
| `gu2019levenshtein` | Levenshtein Transformer | https://proceedings.neurips.cc/paper/2019/hash/675f9820626f5bc0afb47b57890b466e-Abstract.html |
| `chang2022maskgit` | MaskGIT: Masked Generative Image Transformer | https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html |
| `yang2018hotpotqa` | HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering | https://aclanthology.org/D18-1259/ |
| `clark2018arc` | Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge | https://arxiv.org/abs/1803.05457 |
| `rajpurkar2016squad` | SQuAD: 100,000+ Questions for Machine Comprehension of Text | https://aclanthology.org/D16-1264/ |
| `austin2021program` | Program Synthesis with Large Language Models | https://arxiv.org/abs/2108.07732 |
| `rao2019tape` | Evaluating Protein Transfer Learning with TAPE | https://proceedings.neurips.cc/paper_files/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html |
| `rives2021biological` | Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences | https://doi.org/10.1073/pnas.2016239118 |

## Update Rule

When adding a new citation:

1. add the citation to `main.tex` first
2. add the BibTeX entry to `references.bib`
3. add the official source link here
4. rebuild with `./build.sh submission`
