"""E7: Bistability stationarity test (R2-8).

Question: are the bimodal outcomes at sigma=5 genuine stationary states of the
dynamics, or a slow-mixing transient of a too-short (370-tick) horizon?

Design:
  Stage 1: classify seeds at sigma=5 (gain=0.04, lr=0.05) by final adoption
           after the standard horizon (250 + 120 ticks): high >60%, low <38%.
  Stage 2: re-run a sample of high and low seeds to 250 + 1200 ticks (100 years),
           recording adoption every 12 ticks.
  Stage 3: drift statistics — OLS slope of adoption over the final 600 ticks
           and mean adoption in years 25-50 vs 50-100. Genuine multistability
           requires |slope| ~ 0 and no convergence between branches.

Usage:
    python abm/experiments/exp_rev_e7_bistability_stationarity.py
"""
import json
import os
import sys

import numpy as np

_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_abm_dir, '..'))
for _p in (_project_root, _abm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from abm.src.state import init_simulation
from abm.src.step import run_n_ticks, _make_static_config
from abm.src.config import merge_configs

SAVE_DIR = os.path.join(_abm_dir, 'output', 'revision', 'e7')
os.makedirs(SAVE_DIR, exist_ok=True)

SIGMA = 5.0
GAIN = 0.04
LR = 0.05
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
SHORT_POST = 120
LONG_POST = 1200
N_CLASSIFY = 100
N_PER_BRANCH = 10
HIGH_T, LOW_T = 0.60, 0.38
SAMPLE_EVERY = 12


def adoption_of(firms):
    alive = np.asarray(firms.alive)
    exp = np.asarray(firms.ai_experience)
    return float(np.sum((exp > 0.1) & alive) / max(float(np.sum(alive)), 1))


def base_config():
    return merge_configs({
        'sigma_base': SIGMA,
        'sigma_firm_spread': 0.4,
        'sigma_firm_A_sensitivity': 0.5,
        'ai_efficiency_gain': GAIN,
        'strategy_learning_rate': LR,
    })


def classify_seeds():
    """Stage 1: short runs -> branch classification."""
    config = base_config()
    sc = _make_static_config(config)
    rows = []
    for seed_idx in range(N_CLASSIFY):
        state = init_simulation(config, seed=seed_idx * 1000 + 42)
        pre = run_n_ticks(state, sc, PRE_TICKS)
        post = run_n_ticks(pre, sc, SHORT_POST)
        a = adoption_of(post.firms)
        branch = 'high' if a > HIGH_T else ('low' if a < LOW_T else 'mid')
        rows.append({'seed': seed_idx, 'adoption': a, 'branch': branch})
    with open(os.path.join(SAVE_DIR, 'classification.json'), 'w') as f:
        json.dump(rows, f)
    n = {'high': 0, 'mid': 0, 'low': 0}
    for r in rows:
        n[r['branch']] += 1
    print(f'classification (n={N_CLASSIFY}): high={n["high"]} '
          f'mid={n["mid"]} low={n["low"]}')
    return rows


def long_run(seed_idx):
    """Stage 2: one seed extended to 100 years with trajectories."""
    config = base_config()
    sc = _make_static_config(config)
    state = init_simulation(config, seed=seed_idx * 1000 + 42)
    pre = run_n_ticks(state, sc, PRE_TICKS)
    post = run_n_ticks(pre, sc, LONG_POST)

    # re-simulate is expensive to instrument via scan; instead sample by
    # re-running in chunks and recording the state after each chunk
    config2 = base_config()
    sc2 = _make_static_config(config2)
    state = init_simulation(config2, seed=seed_idx * 1000 + 42)
    state = run_n_ticks(state, sc2, PRE_TICKS)
    traj = []
    n_chunks = LONG_POST // SAMPLE_EVERY
    for _ in range(n_chunks):
        state = run_n_ticks(state, sc2, SAMPLE_EVERY)
        traj.append(adoption_of(state.firms))
    assert abs(traj[-1] - adoption_of(post.firms)) < 1e-6, \
        'chunked trajectory diverged from single-scan run'
    ticks = [(i + 1) * SAMPLE_EVERY for i in range(n_chunks)]
    return {'seed': seed_idx, 'ticks': ticks, 'adoption': traj,
            'final': traj[-1]}


def drift_stats(traj):
    """Stage 3: slope over the final 600 ticks + non-overlapping window means.

    Audit 2026-10-07: the mid window used to be a[25:75] (post years 26-75)
    while labeled mean_year25_50, and overlapped the late window (years 51-75)
    by 25 points — diluting the drift statistic to ~60% of its true value.
    Both windows are now non-overlapping with honest names (samples are every
    12 ticks = 1 year; sample i is post year i+1).
    """
    a = np.asarray(traj)
    tail = a[-600 // SAMPLE_EVERY:]
    x = np.arange(len(tail))
    # slope per YEAR (samples are 12 ticks = 1 year apart)
    slope_per_year = float(np.polyfit(x, tail, 1)[0])
    mid = float(np.mean(a[25:50]))    # post years 26-50
    late = float(np.mean(a[50:100]))  # post years 51-100
    return {'slope_per_year_tail': slope_per_year,
            'mean_year26_50': mid, 'mean_year51_100': late}


def main():
    rows = classify_seeds()
    high_seeds = [r['seed'] for r in rows if r['branch'] == 'high'][:N_PER_BRANCH]
    low_seeds = [r['seed'] for r in rows if r['branch'] == 'low'][:N_PER_BRANCH]
    print(f'long runs: {len(high_seeds)} high + {len(low_seeds)} low seeds, '
          f'{LONG_POST} ticks each')

    out = []
    for branch, seeds in [('high', high_seeds), ('low', low_seeds)]:
        for s in seeds:
            r = long_run(s)
            r['branch'] = branch
            r['drift'] = drift_stats(r['adoption'])
            out.append(r)
            print(f'  [{branch}] seed={s}: final={r["final"]:.3f} '
                  f'slope={r["drift"]["slope_per_year_tail"]:.2e}/yr '
                  f'y26-50={r["drift"]["mean_year26_50"]:.3f} '
                  f'y51-100={r["drift"]["mean_year51_100"]:.3f}')

    with open(os.path.join(SAVE_DIR, 'long_runs.json'), 'w') as f:
        json.dump(out, f)

    # verdict
    slopes = [abs(r['drift']['slope_per_year_tail']) for r in out]
    shifts = [abs(r['drift']['mean_year51_100'] - r['drift']['mean_year26_50'])
              for r in out]
    print(f'\nverdict: max|slope|={max(slopes):.2e} per tick; '
          f'max|shift(y25-50 vs y50-100)|={max(shifts):.3f}')
    print('stationary' if max(shifts) < 0.05 else
          'NOT stationary — branches still moving; report as slow-mixing')


if __name__ == '__main__':
    main()
