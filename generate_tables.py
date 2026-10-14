"""Generate paper tables from archived JSON outputs.

Usage (source repo or release package, run from the tree root):
    python generate_tables.py [table] [input_json]

Tables:
    btos      BTOS 15-sector calibration (Table 6 in the manuscript)
              input: abm/output/industry_calibration/industry_calibration_results.json

Tables are written to tables/*.tex (next to this script) and stdout.
Do NOT hand-edit numbers in the manuscript; regenerate instead.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
# Layout-robust data path: in the release package this script sits at the
# package root (abm/ is a sibling); in the source repository it lives one
# directory down.
_DATA_CANDIDATES = [os.path.join(HERE, 'abm', 'output'),
                    os.path.join(HERE, '..', 'abm', 'output')]
DATA_DIR = next((p for p in _DATA_CANDIDATES if os.path.isdir(p)), _DATA_CANDIDATES[-1])
OUT_DIR = os.path.join(HERE, 'tables')
os.makedirs(OUT_DIR, exist_ok=True)


def generate_btos(input_json=None):
    path = input_json or os.path.join(
        DATA_DIR, 'industry_calibration', 'industry_calibration_results.json')
    with open(path) as f:
        data = json.load(f)

    sectors = data['sectors']
    rows = []
    for name, s in sectors.items():
        btos = s['btos_rate']
        model = s['adoption_rate_mean'] * 100
        std = s['adoption_rate_std'] * 100
        sigma = s['sigma']
        rows.append((name, sigma, btos, model, std, model - btos))

    btos_vals = [r[2] for r in rows]
    model_vals = [r[3] for r in rows]
    rho, rho_p = stats.spearmanr(btos_vals, model_vals)
    r, r_p = stats.pearsonr(btos_vals, model_vals)

    # Sort by BTOS adoption descending (presentation order)
    rows.sort(key=lambda x: -x[2])

    lines = [
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'\textbf{Sector} & $\boldsymbol{\sigma}$ & \textbf{BTOS 11--12/2025 (\%)} & '
        r'\textbf{Model (\%)} & \textbf{Error} \\',
        r'\midrule',
    ]
    for name, sigma, btos, model, std, err in rows:
        lines.append(
            f'{name} & {sigma:.2f} & {btos:.1f} & '
            f'${model:.1f} \\pm {std:.1f}$ & {err:+.1f} \\\\')
    lines += [
        r'\midrule',
        rf'\multicolumn{{2}}{{l}}{{Spearman $\rho$}} & '
        rf'\multicolumn{{2}}{{c}}{{{rho:.3f} ($p = {rho_p:.3f}$)}} & \\',
        rf'\multicolumn{{2}}{{l}}{{Pearson $r$}} & '
        rf'\multicolumn{{2}}{{c}}{{{r:.3f} ($p = {r_p:.3f}$)}} & \\',
        r'\bottomrule',
        r'\end{tabular}',
    ]

    out = '\n'.join(lines) + '\n'
    out_path = os.path.join(OUT_DIR, 'table_btos.tex')
    with open(out_path, 'w') as f:
        f.write(out)
    n_seeds = len(sectors[next(iter(sectors))].get('_per_seed', []))
    print(out)
    print(f'%% sectors={len(rows)}, seeds_per_sector={n_seeds}')
    print(f'%% written to {out_path}')


if __name__ == '__main__':
    table = sys.argv[1] if len(sys.argv) > 1 else 'btos'
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if table == 'btos':
        generate_btos(arg)
    else:
        print(f'unknown table: {table}')
        sys.exit(1)
