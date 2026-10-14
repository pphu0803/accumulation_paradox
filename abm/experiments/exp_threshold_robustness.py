"""Experiment: Competition Threshold Robustness

Sweeps ai_efficiency_gain × strategy_learning_rate × fine-grained sigma
to test whether the bistability boundary (sigma ≈ 5) is stable.

4 gain × 4 imitation × 14 sigma = 224 point-estimates of bistability boundary.
Each sigma point uses NUM_SEEDS seeds (default 40, overridden via env; min_count=5 absolute for the bistability label).
"""

import sys
import os
import json
import time

import numpy as np

_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_abm_dir, '..'))
for _p in (_project_root, _abm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from abm.src.state import init_simulation
from abm.src.step import _make_static_config, run_n_ticks
from abm.src.config import merge_configs

import jax.numpy as jnp

SAVE_DIR = os.path.join(_abm_dir, 'output', 'robustness')
os.makedirs(SAVE_DIR, exist_ok=True)

GAINS = [0.02, 0.04, 0.06, 0.08]
IMITATION_RATES = [0.0, 0.03, 0.05, 0.08]
SIGMA_VALUES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

NUM_SEEDS = int(os.environ.get('NUM_SEEDS', '40'))
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120
SIGMA_FIRM_SPREAD = 0.4
SIGMA_FIRM_A_SENS = 0.5


def find_bistability_boundary(gain, lr, sigma_values):
    """Find the sigma value where bistability disappears.
    Bistability = both low-adopt (<30%) and high-adopt (>70%) seeds exist
    with at least min_count each.
    Returns the sigma value where this transitions.
    """
    min_count = 5
    boundaries = []

    for sigma in sigma_values:
        config = merge_configs({
            'sigma_base': sigma,
            'sigma_firm_spread': SIGMA_FIRM_SPREAD,
            'sigma_firm_A_sensitivity': SIGMA_FIRM_A_SENS,
            'ai_efficiency_gain': gain,
            'strategy_learning_rate': lr,
        })
        sc = _make_static_config(config)

        adoptions = []
        for seed_idx in range(NUM_SEEDS):
            seed_key = seed_idx * 1000 + 42
            state = init_simulation(config, seed=seed_key)
            pre = run_n_ticks(state, sc, PRE_TICKS)
            final = run_n_ticks(pre, sc, POST_TICKS)

            alive = final.firms.alive
            adoption = float(jnp.sum((final.firms.ai_experience > 0.1) & alive) /
                             max(float(jnp.sum(alive)), 1))
            adoptions.append(adoption)

        n_low = sum(1 for a in adoptions if a < 0.3)
        n_mid = sum(1 for a in adoptions if 0.3 <= a <= 0.7)
        n_high = sum(1 for a in adoptions if a > 0.7)

        is_bistable = (n_low >= min_count and n_high >= min_count)

        # Also compute max gap (from supplementary_light.py logic)
        sorted_a = np.sort(adoptions)
        if len(sorted_a) > 1:
            gaps = np.diff(sorted_a)
            max_gap = float(np.max(gaps))
        else:
            max_gap = 0.0

        boundaries.append({
            'sigma': sigma, 'gain': gain, 'lr': lr,
            'adoption_mean': float(np.mean(adoptions)),
            'adoption_std': float(np.std(adoptions)),
            'adoption_cv': float(np.std(adoptions) / max(np.mean(adoptions), 0.01)),
            'n_low': n_low, 'n_mid': n_mid, 'n_high': n_high,
            'is_bistable': is_bistable,
            'max_gap': max_gap,
        })

        print(f'  sigma={sigma:5.1f}: adopt={np.mean(adoptions)*100:.1f}%±{np.std(adoptions)*100:.1f}%  '
              f'CV={np.std(adoptions)/max(np.mean(adoptions),0.01):.2f}  '
              f'bistable={is_bistable}')

    return boundaries


def estimate_boundary(boundaries):
    """Estimate the critical sigma* where bistability transitions.
    Uses the last sigma with bistable=True.
    """
    bistable_sigmas = [b['sigma'] for b in boundaries if b['is_bistable']]
    if not bistable_sigmas:
        return None

    return max(bistable_sigmas)  # highest sigma still bistable


def main():
    t0 = time.time()

    print('=' * 60)
    print('THRESHOLD ROBUSTNESS: Bistability boundary sigma*')
    print(f'Grid: {len(GAINS)} gains × {len(IMITATION_RATES)} imitation × {len(SIGMA_VALUES)} sigma')
    print(f'Per point: {NUM_SEEDS} seeds, {PRE_TICKS}+{POST_TICKS} ticks')
    print('=' * 60)

    all_boundaries = []
    summary = []

    for gain in GAINS:
        for lr in IMITATION_RATES:
            print(f'\n--- gain={gain}, lr={lr} ---')
            boundaries = find_bistability_boundary(gain, lr, SIGMA_VALUES)
            all_boundaries.extend(boundaries)

            sigma_star = estimate_boundary(boundaries)
            if sigma_star is not None:
                print(f'  >>> Bistability boundary sigma* = {sigma_star:.2f}')
            else:
                print(f'  >>> No bistability detected (boundary < {SIGMA_VALUES[0]})')

            summary.append({
                'gain': gain, 'lr': lr, 'sigma_star': sigma_star,
            })

    # Save raw data
    with open(os.path.join(SAVE_DIR, 'threshold_robustness.json'), 'w') as f:
        json.dump(all_boundaries, f, indent=2, default=float)
    print(f'\nSaved: threshold_robustness.json')

    # Summary analysis
    sigma_stars = [s['sigma_star'] for s in summary if s['sigma_star'] is not None]
    print(f'\n{"="*60}')
    print('SUMMARY')
    if sigma_stars:
        print(f'  sigma* range: {min(sigma_stars):.2f} -- {max(sigma_stars):.2f}')
        print(f'  sigma* mean:  {np.mean(sigma_stars):.2f}')
        print(f'  sigma* std:   {np.std(sigma_stars):.2f}')

        # By gain
        print(f'\n  By gain:')
        for g in GAINS:
            stars = [s['sigma_star'] for s in summary if s['gain'] == g and s['sigma_star'] is not None]
            if stars:
                print(f'    g={g}: sigma* = {np.mean(stars):.2f} (n={len(stars)})')

        # By imitation
        print(f'  By imitation rate:')
        for lr_val in IMITATION_RATES:
            stars = [s['sigma_star'] for s in summary if s['lr'] == lr_val and s['sigma_star'] is not None]
            if stars:
                print(f'    lr={lr_val}: sigma* = {np.mean(stars):.2f} (n={len(stars)})')
    else:
        print('  No bistability detected in any configuration!')

    print(f'\nElapsed: {time.time() - t0:.0f}s ({(time.time()-t0)/60:.1f}min)')


if __name__ == '__main__':
    main()
