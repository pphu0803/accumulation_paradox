"""Lambda_q (quality conversion elasticity) sensitivity for the revision.

Rationale (Section 4 robustness, "Quality-channel calibration"): lambda_q has
no external anchor, and the inclusive-TFP level gains depend on it. The v1-era
numbers cited in the manuscript came from an unpreserved ad-hoc run; this
script regenerates them under the v3 (post-2026-10-07-audit) engine so every
published number has a first-party source.

Design: sigma in {2, 5} x lambda_q in {0.15, 0.30, 0.60} x 20 seeds,
gate on, quality on, otherwise identical to exp_rev_e1 cells.

Usage:
    PYTHONPATH=. python abm/experiments/exp_rev_lambdaq_sensitivity.py
Output: abm/output/revision/e1/lambdaq_sensitivity.json
"""
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

OUT_PATH = os.path.join(_abm_dir, 'output', 'revision', 'e1', 'lambdaq_sensitivity.json')
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

SIGMA_VALUES = [2.0, 5.0]
LAMBDAQ_VALUES = [0.15, 0.30, 0.60]   # 0.30 = default
GAIN = 0.04
NUM_SEEDS = 20
PRE_TICKS = 250
POST_TICKS = 120
GATE_ON = {
    'adoption_readiness_speed': 0.02,
    'adoption_peer_influence': 0.05,
    'adoption_threshold': 0.8,
}


def ai_tfp_multiplier(x, gain, depth):
    x = np.asarray(x)
    m = 1.0 + gain * ((1.0 + depth) * x - depth)
    return np.where(x > 0, m, 1.0)


def macro_tfp(firms, alpha, theta, gain, depth, quality=True):
    alive = np.asarray(firms.alive)
    K = np.maximum(np.asarray(firms.K), 0.01)
    L = np.maximum(np.asarray(firms.L), 0.01)
    inputs = K ** alpha * L ** (1 - alpha)
    a_eff = np.asarray(firms.A_base) * ai_tfp_multiplier(firms.ai_experience, gain, depth)
    if quality:
        q = np.maximum(np.asarray(firms.product_quality), 0.1)
        num = a_eff * q ** theta * inputs
    else:
        num = a_eff * inputs
    m = alive & (inputs > 0.01)
    return float(np.sum(num[m]) / max(np.sum(inputs[m]), 1e-6))


def main():
    results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f'resuming: {sum(len(v) for v in results.values())} seeds done')

    for sigma in SIGMA_VALUES:
        for lq in LAMBDAQ_VALUES:
            key = f'sigma={sigma}_lq={lq}'
            done = len(results.get(key, []))
            if done >= NUM_SEEDS:
                print(f'[{key}] already complete')
                continue
            config = merge_configs({
                'sigma_base': sigma,
                'sigma_firm_spread': 0.4,
                'sigma_firm_A_sensitivity': 0.5,
                'ai_efficiency_gain': GAIN,
                'ai_quality_elasticity': lq,
                **GATE_ON,
            })
            sc = _make_static_config(config)
            gain, depth, alpha = float(sc.ai_efficiency_gain), float(sc.j_curve_depth), float(sc.alpha_CD)
            theta = float(sc.quality_weight)
            rows = results.setdefault(key, [])
            t0 = time.time()
            for seed_idx in range(done, NUM_SEEDS):
                state = init_simulation(config, seed=seed_idx * 1000 + 42)
                pre = run_n_ticks(state, sc, PRE_TICKS)
                post = run_n_ticks(pre, sc, POST_TICKS)
                pre_raw = macro_tfp(pre.firms, alpha, theta, gain, depth)
                post_raw = macro_tfp(post.firms, alpha, theta, gain, depth)
                pre_ai = macro_tfp(pre.firms, alpha, theta, gain, depth, quality=False)
                post_ai = macro_tfp(post.firms, alpha, theta, gain, depth, quality=False)
                alive = np.asarray(post.firms.alive)
                adoption = float(np.sum(
                    (np.asarray(post.firms.ai_experience) > 0.1) & alive)
                    / max(float(np.sum(alive)), 1))
                rows.append({
                    'seed': seed_idx,
                    'delta_tfp_raw': (post_raw - pre_raw) / pre_raw,
                    'delta_tfp_ai': (post_ai - pre_ai) / pre_ai,
                    'quality_ratio_post': post_raw / post_ai,
                    'adoption_rate': adoption,
                })
                if (seed_idx + 1) % 10 == 0 or seed_idx + 1 == NUM_SEEDS:
                    tmp = OUT_PATH + '.tmp'
                    with open(tmp, 'w') as f:
                        json.dump(results, f)
                    os.replace(tmp, OUT_PATH)
            print(f'[{key}] {NUM_SEEDS} seeds in {time.time()-t0:.0f}s', flush=True)

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f)
    print('saved:', OUT_PATH)


if __name__ == '__main__':
    main()
