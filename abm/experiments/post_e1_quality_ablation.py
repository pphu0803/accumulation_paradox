"""Aggregate the E1 quality-channel ablation into the table quoted in
Section 4.3 (quality channel on vs off, gate-on arm).

Reads the per-cell seed records written by exp_rev_e1_tfp_decomposition.py
from abm/output/revision/e1/cells/ and writes quality_ablation_table.json:
per sigma, seed means of the inclusive (raw) and AI-efficiency (ai) TFP
change and of the adoption rate, with 1.96*SE half-widths for the raw
measure. With the quality channel ablated (theta = 0) delta_tfp_raw equals
delta_tfp_ai per seed, so off_ai must equal off_raw[0] -- asserted as a
built-in consistency check. Pure numpy; no simulation.
"""
import json
import os

import numpy as np

EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXPERIMENTS, '..', 'output', 'revision', 'e1')
SIGMAS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]


def mean_ci(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def run():
    on = json.load(open(os.path.join(OUT, 'cells', 'gate_on_quality_on.json')))
    off = json.load(open(os.path.join(OUT, 'cells', 'gate_on_quality_off.json')))
    table = {}
    for s in SIGMAS:
        key = f'sigma={s}'
        on_raw_m, on_raw_ci = mean_ci([r['delta_tfp_raw'] for r in on[key]])
        off_raw_m, off_raw_ci = mean_ci([r['delta_tfp_raw'] for r in off[key]])
        off_ai_m = float(np.mean([r['delta_tfp_ai'] for r in off[key]]))
        assert abs(off_ai_m - off_raw_m) < 1e-12, 'theta=0: raw must equal ai'
        table[key] = {
            'on_raw': [on_raw_m, on_raw_ci],
            'off_raw': [off_raw_m, off_raw_ci],
            'on_ai': float(np.mean([r['delta_tfp_ai'] for r in on[key]])),
            'off_ai': off_ai_m,
            'on_adopt': float(np.mean([r['adoption_rate'] for r in on[key]])),
            'off_adopt': float(np.mean([r['adoption_rate'] for r in off[key]])),
        }
    path = os.path.join(OUT, 'quality_ablation_table.json')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(table, f, indent=1)
    os.replace(tmp, path)
    print(f'-> {path}')
    for s in (5.0, 8.0):
        k = f'sigma={s}'
        print(f"{k}: quality-off adoption {table[k]['off_adopt']:.3f} "
              f"vs quality-on {table[k]['on_adopt']:.3f}")
    return table


if __name__ == '__main__':
    run()
