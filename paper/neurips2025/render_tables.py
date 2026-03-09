#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data' / 'metrics.json').read_text())
TABLES = ROOT / 'tables'
TABLES.mkdir(exist_ok=True)

NL = "\\\\"


def latex_escape(s):
    return (str(s)
            .replace('\\', r'\textbackslash{}')
            .replace('&', r'\&')
            .replace('%', r'\%')
            .replace('_', r'\_')
            .replace('#', r'\#'))

def fmt(x):
    return f"{x:.2f}"

def pct(x):
    return f"{x:.2f}\\%"

# Synthetic main table
rows = []
for r in DATA['synthetic']['main_summary']:
    rows.append(f"{latex_escape(r['task'])} & {pct(r['baseline'])} & {pct(r['future_seed'])} & +{fmt(r['delta'])}pp {NL}")
(TABLES / 'synthetic_main.tex').write_text(
"""\\begin{table}[t]
\\centering
\\small
\\caption{Synthetic mechanism evidence. Future-Seed consistently improves future-aware repair tasks and turns KVSORT from failure to exact recovery.}
\\label{tab:synthetic-main}
\\begin{tabular}{lccc}
\\toprule
Task / metric & Baseline & Future-Seed & Delta \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")

# Permfill table
rows = []
for r in DATA['synthetic']['permfill']:
    rows.append(f"{latex_escape(r['weights'])} & {latex_escape(r['anchor'])} & {pct(r['n24'])} & {pct(r['n28'])} & {pct(r['n32'])} & {pct(r['n36'])} {NL}")
(TABLES / 'permfill_main.tex').write_text(
"""\\begin{table}[t]
\\centering
\\small
\\caption{PERMFILL length extrapolation. A tiny anchor in the masked span materially improves out-of-distribution exact match.}
\\label{tab:permfill-main}
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{lccccc}
\\toprule
Weights & Anchor & n=24 & n=28 & n=32 & n=36 \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}}
\\end{table}
""")

# Posttrain main table
rows = []
for r in DATA['posttrain']['main_summary']:
    rows.append(
        f"{latex_escape(r['task'])} & +{fmt(r['best'])}pp & +{fmt(r['median'])}pp & "
        f"{r['pos']}/{r['total']} & {latex_escape(r['judgment'])} & {latex_escape(r['confirm'])} {NL}"
    )
(TABLES / 'posttrain_main.tex').write_text(
"""\\begin{table*}[t]
\\centering
\\small
\\caption{Curated post-training summary used in this paper. Counts aggregate medium-stage Future-Seed vs. baseline comparisons from the internal search archive snapshot behind the paper tables. The final column summarizes whether a family survived the deliberate confirmation queues used near the end of the search.}
\\label{tab:posttrain-main}
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{lccccc}
\\toprule
Probe family & Best gain & Median gain & Positive med count & Current judgment & Confirmation \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}}
\\end{table*}
""")

# Appendix task table
rows = []
for r in DATA['posttrain']['appendix_summary']:
    sign_best = '+' if r['best'] >= 0 else ''
    sign_med = '+' if r['median'] >= 0 else ''
    rows.append(f"{latex_escape(r['task'])} & {sign_best}{fmt(r['best'])}pp & {sign_med}{fmt(r['median'])}pp & {r['pos']}/{r['total']} & {latex_escape(r['judgment'])} {NL}")
(TABLES / 'appendix_negative_tasks.tex').write_text(
"""\\begin{table}[t]
\\centering
\\small
\\caption{Additional task families discussed in appendix.}
\\label{tab:appendix-negative-tasks}
\\begin{tabular}{lcccc}
\\toprule
Task family & Best gain & Median gain & Positive med count & Judgment \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")

# Sudoku phase appendix
rows = []
for r in DATA['synthetic']['sudoku_phase']:
    rows.append(f"{r['holes']} & {pct(r['fs0'])} & {pct(r['fs1'])} {NL}")
(TABLES / 'appendix_sudoku_phase.tex').write_text(
"""\\begin{table}[t]
\\centering
\\small
\caption{4x4 Sudoku solve-rate phase curve. Future-Seed delays the collapse point to harder puzzles.}
\\label{tab:appendix-sudoku-phase}
\\begin{tabular}{rcc}
\\toprule
Holes & FS=0 solve & FS=1 solve \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")

# Sudoku consistency appendix
rows = []
for r in DATA['synthetic']['sudoku_consistency']:
    rows.append(f"{latex_escape(r['setting'])} & {pct(r['holes10'])} & {pct(r['holes12'])} & {pct(r['holes14'])} {NL}")
(TABLES / 'appendix_sudoku_consistency.tex').write_text(
"""\\begin{table}[t]
\\centering
\\small
\\caption{A simple Sudoku consistency regularizer helps at holes=12 but does not solve the hardest setting.}
\\label{tab:appendix-sudoku-consistency}
\\begin{tabular}{lccc}
\\toprule
Setting & holes=10 & holes=12 & holes=14 \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")

# Transformer appendix
rows = []
for r in DATA['synthetic']['transformer_baselines']:
    rows.append(f"{latex_escape(r['model'])} & {latex_escape(r['decode'])} & {pct(r['exact'])} & {pct(r['key_valid'])} & {pct(r['key_order'])} {NL}")
(TABLES / 'appendix_transformer.tex').write_text(
"""\\begin{table*}[t]
\\centering
\\small
\\caption{Transformer baselines on KVSORT. Hungarian decoding restores validity but not ordering.}
\\label{tab:appendix-transformer}
\\begin{tabular}{lcccc}
\\toprule
Model & Decode & Exact & Key-valid & Key-order \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table*}
""")
