"""E11: escape-time distribution for the metastable delay regime (v3.1 deep-dive, W4).

Extends 100 seeds at sigma=5 (the delay-regime centre) to 30 model years with
yearly adoption sampling, and reports:
  - year-10 classification (low <0.3 / intermediate / high >0.7)
  - escape year for year-10-low seeds (first year >=11 with adoption >= 0.5)
  - share of low-classified seeds still low at year 30
Feeds the paper's metastable-delay claim (Sec 4.7) with a distribution rather
than 10+10 centennial trajectories. If a persistent low sub-population appears
at year 30, the centennial subset extension is the follow-up (not run here).

Usage:
    python abm/experiments/exp_rev_e11_escape_times.py
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
from abm.src.step import run_n_ticks, run_n_ticks_collect, _make_static_config
from abm.src.config import merge_configs

SAVE_DIR = os.path.join(_abm_dir, 'output', 'revision', 'e11')
os.makedirs(SAVE_DIR, exist_ok=True)
OUT_PATH = os.path.join(SAVE_DIR, 'e11_escape_times.json')

SIGMA = 5.0
NUM_SEEDS = 100
PRE_TICKS = 250          # matches E2/E10 design (audit 2026-10-07 S2)
POST_TICKS = 360         # 30 model years
SAMPLE_EVERY = 12        # yearly samples; sampled[i] <-> post tick (i+1)*12
GAIN = 0.04
SIGMA_FIRM_SPREAD = 0.4
SIGMA_FIRM_A_SENS = 0.5
ADOPT_EXP = 0.1


def classify(adoption_y10):
    if adoption_y10 < 0.3:
        return 'low'
    if adoption_y10 > 0.7:
        return 'high'
    return 'intermediate'


def run_seed(seed_idx):
    config = merge_configs({
        'sigma_base': SIGMA,
        'sigma_firm_spread': SIGMA_FIRM_SPREAD,
        'sigma_firm_A_sensitivity': SIGMA_FIRM_A_SENS,
        'ai_efficiency_gain': GAIN,
    })
    sc = _make_static_config(config)
    state = init_simulation(config, seed=seed_idx * 1000 + 42)
    pre = run_n_ticks(state, sc, PRE_TICKS)
    final, sampled = run_n_ticks_collect(pre, sc, POST_TICKS,
                                         sample_every=SAMPLE_EVERY)
    n_s = sampled.firms.alive.shape[0]
    adoption, n_alive = [], []
    for i in range(n_s):
        alive = np.asarray(sampled.firms.alive[i])
        adopt = np.sum((np.asarray(sampled.firms.ai_experience[i]) > ADOPT_EXP)
                       & alive) / max(float(alive.sum()), 1.0)
        adoption.append(float(adopt))
        n_alive.append(int(alive.sum()))
    year10 = adoption[9]            # sampled[9] <-> post tick 120
    cls = classify(year10)
    escape_year = None
    if cls == 'low':
        for k in range(10, n_s):    # years 11..30
            if adoption[k] >= 0.5:
                escape_year = k + 1
                break
    del state, pre, final, sampled
    return {'seed': seed_idx, 'adoption': adoption, 'n_alive': n_alive,
            'year10': year10, 'class': cls, 'escape_year': escape_year,
            'still_low_30': bool(cls == 'low' and adoption[-1] < 0.3)}


def main():
    t0 = time.time()
    rows = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            rows = json.load(f)
    done = {r['seed'] for r in rows}
    for seed_idx in range(NUM_SEEDS):
        if seed_idx in done:
            continue
        rows.append(run_seed(seed_idx))
        tmp = OUT_PATH + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(rows, f)
        os.replace(tmp, OUT_PATH)
        r = rows[-1]
        print(f'seed={seed_idx} year10={r["year10"]:.2f} ({r["class"]}) '
              f'esc={r["escape_year"]} ({time.time()-t0:.0f}s)', flush=True)
    # summary
    cls = [r['class'] for r in rows]
    lows = [r for r in rows if r['class'] == 'low']
    esc = sorted(r['escape_year'] for r in lows if r['escape_year'])
    still = sum(1 for r in lows if r['still_low_30'])
    summary = {
        'n_seeds': len(rows),
        'n_low': cls.count('low'), 'n_intermediate': cls.count('intermediate'),
        'n_high': cls.count('high'),
        'escape_years': esc,
        'escape_year_median': float(np.median(esc)) if esc else None,
        'low_still_low_at_30': still,
        'adoption_low_at_30_mean': float(np.mean(
            [r['adoption'][-1] for r in lows])) if lows else None,
    }
    with open(os.path.join(SAVE_DIR, 'e11_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
