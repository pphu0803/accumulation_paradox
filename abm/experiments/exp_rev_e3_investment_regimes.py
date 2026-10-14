"""E3: investment-regime sweep (w_profit x lambda_comp) — R1 behavioural rules.

Reviewer 1 asks: the investment rule appears to give profitability a dual role
(w_profit rewards being profitable; lambda_comp pushes laggards to invest).
Which dominates, what is the baseline regime, and do the headline patterns
(paradox zone, threshold, reversal) survive alternative regimes?

NOTE on code-paper correspondence (Section 2 rewrite): in the implementation
the "profitability signal" is the aggregate fraction of profitable adopters
(investment_profit_weight) and the "catch-up" term keys on the AI-experience
gap (competitive_pressure_coeff) — see step.py S1. This experiment sweeps
those two implemented coefficients.

Design:
  profit_weight in {0.2, 0.5, 0.8} x pressure in {0.5, 2.0, 5.0} (9 regimes)
  x sigma in {2, 3, 4, 5, 6, 7} x 40 seeds, 50+120 ticks.
  Records adoption, dTFP_raw, dTFP_ai, plus the regime's implied threshold:
  the lowest sigma (in-grid) where mean adoption falls below 50%.

Usage:
    python abm/experiments/exp_rev_e3_investment_regimes.py [--collect]
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

SAVE_DIR = os.path.join(_abm_dir, 'output', 'revision', 'e3')
CELL_DIR = os.path.join(SAVE_DIR, 'cells')
os.makedirs(CELL_DIR, exist_ok=True)

PROFIT_WEIGHTS = [0.2, 0.5, 0.8]
PRESSURES = [0.5, 2.0, 5.0]
SIGMAS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
NUM_SEEDS = 40
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120
GAIN = 0.04


def tfp_metrics(pre_firms, post_firms, sc):
    alpha = float(sc.alpha_CD)
    gain = float(sc.ai_efficiency_gain)
    depth = float(sc.j_curve_depth)

    def macro(firms, with_quality=True):
        alive = np.asarray(firms.alive)
        K = np.maximum(np.asarray(firms.K), 0.01)
        L = np.maximum(np.asarray(firms.L), 0.01)
        inputs = K ** alpha * L ** (1 - alpha)
        x = np.asarray(firms.ai_experience)
        mult = np.where(x > 0, 1.0 + gain * ((1 + depth) * x - depth), 1.0)
        if with_quality:
            q = np.maximum(np.asarray(firms.product_quality), 0.1)
            theta = float(sc.quality_weight)
            a = np.asarray(firms.A_base) * mult * q ** theta
        else:
            a = np.asarray(firms.A_base) * mult
        m = alive & (inputs > 0.01)
        return float(np.sum(a[m] * inputs[m]) / max(np.sum(inputs[m]), 1e-6))

    pre_r, post_r = macro(pre_firms), macro(post_firms)
    pre_a, post_a = macro(pre_firms, False), macro(post_firms, False)
    return ((post_r - pre_r) / pre_r, (post_a - pre_a) / pre_a)


def adoption_of(firms):
    alive = np.asarray(firms.alive)
    exp = np.asarray(firms.ai_experience)
    return float(np.sum((exp > 0.1) & alive) / max(float(np.sum(alive)), 1))


def run_cell(pw, cp):
    out_path = os.path.join(CELL_DIR, f'pw{pw}_cp{cp}.json')
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        done_total = sum(len(v) for v in results.values())
        if done_total >= NUM_SEEDS * len(SIGMAS):
            print(f'  pw={pw} cp={cp}: already complete')
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
            'investment_profit_weight': pw,
            'competitive_pressure_coeff': cp,
        })
        sc = _make_static_config(config)
        for seed_idx in range(len(rows), NUM_SEEDS):
            state = init_simulation(config, seed=seed_idx * 1000 + 42)
            pre = run_n_ticks(state, sc, PRE_TICKS)
            post = run_n_ticks(pre, sc, POST_TICKS)
            d_raw, d_ai = tfp_metrics(pre.firms, post.firms, sc)
            rows.append({'seed': seed_idx, 'adoption': adoption_of(post.firms),
                         'delta_tfp_raw': d_raw, 'delta_tfp_ai': d_ai})
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(results, f)
        os.replace(tmp, out_path)
        import jax
        jax.clear_caches()
    print(f'  pw={pw} cp={cp}: done')


def collect():
    grid = {}
    for pw in PROFIT_WEIGHTS:
        for cp in PRESSURES:
            p = os.path.join(CELL_DIR, f'pw{pw}_cp{cp}.json')
            if not os.path.exists(p):
                continue
            with open(p) as f:
                cells = json.load(f)
            adopt = {s: float(np.mean([r['adoption'] for r in v]))
                     for s, v in cells.items()}
            dtfp = {s: float(np.mean([r['delta_tfp_ai'] for r in v]))
                    for s, v in cells.items()}
            # in-grid threshold: lowest sigma with mean adoption < 50%
            thresh = next(
                (s for s in sorted(adopt,
                                   key=lambda x: float(x.split('=')[1]))
                 if adopt[s] < 0.5), None)
            grid[f'pw={pw},cp={cp}'] = {
                'adoption_by_sigma': adopt,
                'dtfp_ai_by_sigma': dtfp,
                'sigma_star_in_grid': thresh,
            }
    out = os.path.join(SAVE_DIR, 'e3_regimes.json')
    with open(out, 'w') as f:
        json.dump(grid, f, indent=1)
    print(f'collected -> {out}')
    for k, v in grid.items():
        a = ', '.join(f"{s.replace('sigma=', '')}:{val * 100:.0f}%" for s, val in
                      sorted(v['adoption_by_sigma'].items(),
                             key=lambda x: float(x[0].split('=')[1])))
        print(f"{k}: sigma*={v['sigma_star_in_grid']}  {a}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args()
    if args.collect:
        collect()
    else:
        t0 = time.time()
        for pw in PROFIT_WEIGHTS:
            for cp in PRESSURES:
                run_cell(pw, cp)
        print(f'total {time.time()-t0:.0f}s')
