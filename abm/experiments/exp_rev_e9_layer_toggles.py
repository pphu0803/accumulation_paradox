"""E9: layer-ablation robustness — do the auxiliary layers drive the results?

Reviewer 2 (omissions i-ii) worried that the undocumented demand-externality
layer and firm entry/exit dynamics might carry the headline results, since
Section 2 described neither.

This experiment reruns the headline sigma comparison under four layer
configurations (config-only toggles, no code changes):

  baseline        : everything on (default)
  no_demand_fb    : demand_shock_coefficient = 0   (layer-2 externality off)
  no_entry_exit   : firm_entry_rate = 0, firm_exit_cash_threshold = -1e6
  neither         : both off

The 500-worker labour-market layer has no clean switch and remains on in all
runs; it is documented as part of the baseline architecture in the revised
Section 2.3.

Design: 4 layers x sigma {2, 5, 8} x 50 seeds, 50+120 ticks.

Usage:
    python abm/experiments/exp_rev_e9_layer_toggles.py [--collect]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_abm_dir, '..'))
for _p in (_project_root, _abm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from abm.src.state import init_simulation
from abm.src.step import run_n_ticks, _make_static_config
from abm.src.config import merge_configs

SAVE_DIR = os.path.join(_abm_dir, 'output', 'revision', 'e9')
CELL_DIR = os.path.join(SAVE_DIR, 'cells')
os.makedirs(CELL_DIR, exist_ok=True)

LAYERS = {
    'baseline': {},
    'no_demand_fb': {'demand_shock_coefficient': 0.0},
    'no_entry_exit': {'firm_entry_rate': 0.0,
                      'firm_exit_cash_threshold': -1e6},
    'neither': {'demand_shock_coefficient': 0.0,
                'firm_entry_rate': 0.0,
                'firm_exit_cash_threshold': -1e6},
}
SIGMAS = [2.0, 5.0, 8.0]
NUM_SEEDS = 50
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120
GAIN = 0.04


def macro_tfp(firms, sc, with_quality=True):
    alpha = float(sc.alpha_CD)
    gain = float(sc.ai_efficiency_gain)
    depth = float(sc.j_curve_depth)
    alive = np.asarray(firms.alive)
    K = np.maximum(np.asarray(firms.K), 0.01)
    L = np.maximum(np.asarray(firms.L), 0.01)
    inputs = K ** alpha * L ** (1 - alpha)
    x = np.asarray(firms.ai_experience)
    mult = np.where(x > 0, 1.0 + gain * ((1 + depth) * x - depth), 1.0)
    if with_quality:
        q = np.maximum(np.asarray(firms.product_quality), 0.1)
        a = np.asarray(firms.A_base) * mult * q ** float(sc.quality_weight)
    else:
        a = np.asarray(firms.A_base) * mult
    m = alive & (inputs > 0.01)
    return float(np.sum(a[m] * inputs[m]) / max(np.sum(inputs[m]), 1e-6))


def adoption_of(firms):
    alive = np.asarray(firms.alive)
    exp = np.asarray(firms.ai_experience)
    return float(np.sum((exp > 0.1) & alive) / max(float(np.sum(alive)), 1))


def run_cell(layer, overrides):
    out_path = os.path.join(CELL_DIR, f'{layer}.json')
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        if all(len(v) >= NUM_SEEDS for v in results.values()):
            print(f'  [{layer}] complete')
            return

    for sigma in SIGMAS:
        key = f'sigma={sigma}'
        rows = results.setdefault(key, [])
        if len(rows) >= NUM_SEEDS:
            continue
        config = merge_configs({
            'sigma_base': sigma,
            'sigma_firm_spread': 0.4,
            'sigma_firm_A_sensitivity': 0.5,
            'ai_efficiency_gain': GAIN,
            **overrides,
        })
        sc = _make_static_config(config)
        for seed_idx in range(len(rows), NUM_SEEDS):
            state = init_simulation(config, seed=seed_idx * 1000 + 42)
            pre = run_n_ticks(state, sc, PRE_TICKS)
            post = run_n_ticks(pre, sc, POST_TICKS)
            pre_r, post_r = macro_tfp(pre.firms, sc), macro_tfp(post.firms, sc)
            pre_a, post_a = (macro_tfp(pre.firms, sc, False),
                             macro_tfp(post.firms, sc, False))
            rows.append({
                'seed': seed_idx,
                'delta_tfp_raw': (post_r - pre_r) / pre_r,
                'delta_tfp_ai': (post_a - pre_a) / pre_a,
                'adoption': adoption_of(post.firms),
                'n_alive': int(np.sum(np.asarray(post.firms.alive))),
            })
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(results, f)
        os.replace(tmp, out_path)
        import jax
        jax.clear_caches()
    print(f'  [{layer}] done')


def collect():
    summary = {}
    for layer, overrides in LAYERS.items():
        p = os.path.join(CELL_DIR, f'{layer}.json')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            cells = json.load(f)
        summary[layer] = {}
        for sig, rows in cells.items():
            summary[layer][sig] = {
                'delta_tfp_raw': float(np.mean([r['delta_tfp_raw'] for r in rows])),
                'delta_tfp_ai': float(np.mean([r['delta_tfp_ai'] for r in rows])),
                'adoption': float(np.mean([r['adoption'] for r in rows])),
                'n_alive': float(np.mean([r['n_alive'] for r in rows])),
            }
    out = os.path.join(SAVE_DIR, 'e9_layers.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'-> {out}')
    print(f"{'layer':<14} {'metric':<14} " + ' '.join(f's={s:<6}' for s in map(str, SIGMAS)))
    for layer in LAYERS:
        if layer not in summary:
            continue
        for metric in ['delta_tfp_raw', 'delta_tfp_ai', 'adoption', 'n_alive']:
            vals = ' '.join(f"{summary[layer][f'sigma={s}'][metric]:<8.3f}"
                            for s in SIGMAS)
            print(f'{layer:<14} {metric:<14} {vals}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args()
    if args.collect:
        collect()
    else:
        t0 = time.time()
        for layer, overrides in LAYERS.items():
            run_cell(layer, overrides)
        print(f'total {time.time()-t0:.0f}s')
