"""Experiment: Temporal Reversal Robustness

Sweeps S-curve gating parameters to test whether the 4-year industry
adoption ranking reversal is a structural phenomenon or a parameter artifact.

Grid: adoption_readiness_speed × adoption_peer_influence × adoption_threshold
     [0.01, 0.02, 0.03]  ×  [0.03, 0.05, 0.08]  ×  [0.6, 0.7, 0.8, 0.9]
= 36 configurations × 14 sectors × 10 seeds = 5040 runs
"""

import sys
import os
import json
import time

import numpy as np
from scipy.stats import spearmanr

_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_abm_dir, '..'))
for _p in (_project_root, _abm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from abm.src.state import init_simulation
from abm.src.step import _make_static_config, run_n_ticks, run_n_ticks_collect
from abm.src.config import merge_configs

import jax
import jax.numpy as jnp

BTOS_DATA = {
    'Information': 36.55, 'Professional Services': 33.35,
    'Finance and Insurance': 29.65, 'Educational Services': 27.75,
    'Real Estate and Rental': 22.65, 'Health Care': 19.75,
    'Arts and Recreation': 17.85, 'Wholesale Trade': 13.0,
    'Retail Trade': 12.8, 'Manufacturing': 11.6, 'Other Services': 9.95,
    'Construction': 8.35, 'Accommodation and Food': 8.35, 'Transportation': 7.15,
}

INDUSTRY_MARKUP = {
    'Information': 1.55, 'Professional Services': 1.50,
    'Finance and Insurance': 1.60, 'Educational Services': 1.20,
    'Real Estate and Rental': 1.35, 'Health Care': 1.40,
    'Arts and Recreation': 1.15, 'Wholesale Trade': 1.18,
    'Retail Trade': 1.15, 'Manufacturing': 1.25, 'Other Services': 1.20,
    'Construction': 1.10, 'Accommodation and Food': 1.10, 'Transportation': 1.12,
}

SAVE_DIR = os.path.join(_abm_dir, 'output', 'robustness')
os.makedirs(SAVE_DIR, exist_ok=True)

SPEEDS = [0.01, 0.02, 0.03]
PEERS = [0.03, 0.05, 0.08]
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

NUM_SEEDS = 10
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120
SAMPLE_EVERY = 6

# Use calibrated heterogeneity params (from comprehensive experiment)
SIGMA_FIRM_SPREAD = 0.4
SIGMA_FIRM_A_SENS = 0.5


def run_one_config(speed, peer, threshold):
    """Run reversal analysis for one S-curve configuration.
    Returns mean adoption trajectory per sector and Spearman rho over time."""

    sectors = list(BTOS_DATA.keys())
    btos_arr = np.array([BTOS_DATA[s] for s in sectors])
    sigma_map = {s: INDUSTRY_MARKUP[s] / (INDUSTRY_MARKUP[s] - 1.0) for s in sectors}

    per_sector_mean = {}
    for sector in sectors:
        sigma = sigma_map[sector]
        sector_config = merge_configs({
            'sigma_base': sigma,
            'sigma_firm_spread': SIGMA_FIRM_SPREAD,
            'sigma_firm_A_sensitivity': SIGMA_FIRM_A_SENS,
            'adoption_readiness_speed': speed,
            'adoption_peer_influence': peer,
            'adoption_threshold': threshold,
        })
        sc = _make_static_config(sector_config)

        seed_means = []
        for seed_idx in range(NUM_SEEDS):
            seed_key = seed_idx * 1000 + 42
            state = init_simulation(sector_config, seed=seed_key)
            pre = run_n_ticks(state, sc, PRE_TICKS)
            final, sampled = run_n_ticks_collect(
                pre, sc, POST_TICKS, sample_every=SAMPLE_EVERY
            )

            n_samples = sampled.firms.alive.shape[0]
            traj = []
            for i in range(n_samples):
                alive = sampled.firms.alive[i]
                a = float(jnp.sum(
                    (sampled.firms.ai_experience[i] > 0.1) & alive
                ) / max(float(jnp.sum(alive)), 1))
                traj.append(a)
            seed_means.append(traj)

        per_sector_mean[sector] = np.mean(seed_means, axis=0)
        del seed_means  # free memory

    # Compute Spearman rho at each time point
    n_t = list(per_sector_mean.values())[0].shape[0]
    rho_traj = []
    p_traj = []
    tick_labels = [(i + 1) * SAMPLE_EVERY for i in range(n_t)]

    for t_idx in range(n_t):
        model_at_t = [per_sector_mean[s][t_idx] for s in sectors]
        model_arr = np.array(model_at_t)
        if np.std(model_arr) < 1e-8:
            rho_traj.append(np.nan)
            p_traj.append(np.nan)
        else:
            r, p = spearmanr(model_arr, btos_arr)
            rho_traj.append(r)
            p_traj.append(p)

    return rho_traj, p_traj, tick_labels, per_sector_mean


def main():
    t0 = time.time()

    grid = []
    for speed in SPEEDS:
        for peer in PEERS:
            for threshold in THRESHOLDS:
                grid.append((speed, peer, threshold))

    checkpoint_path = os.path.join(SAVE_DIR, 'reversal_checkpoint.json')
    results = []
    start_idx = 0

    # Restore from checkpoint if available
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            saved = json.load(f)
        results = saved
        start_idx = len(results)
        print(f'Restored {start_idx} configs from checkpoint')

    for idx in range(start_idx, len(grid)):
        speed, peer, threshold = grid[idx]
        tag = f'speed={speed}_peer={peer}_thresh={threshold}'
        print(f'\n[{idx+1}/{len(grid)}] {tag}')

        rho, p, ticks, sector_traj = run_one_config(speed, peer, threshold)

        # Detect reversals
        sign_changes = []
        for i in range(1, len(rho)):
            if not (np.isnan(rho[i]) or np.isnan(rho[i - 1])):
                if rho[i - 1] * rho[i] < 0:
                    sign_changes.append({
                        'tick': ticks[i], 'year': ticks[i] / 12.0,
                        'rho_before': rho[i - 1], 'rho_after': rho[i],
                    })

        early = [x for x in rho[:len(rho) // 2] if not np.isnan(x)]
        late = [x for x in rho[len(rho) // 2:] if not np.isnan(x)]

        entry = {
            'speed': speed, 'peer': peer, 'threshold': threshold,
            'rho_trajectory': rho, 'p_trajectory': p,
            'tick_labels': ticks,
            'has_reversal': len(sign_changes) > 0,
            'n_sign_changes': len(sign_changes),
            'first_reversal_tick': sign_changes[0]['tick'] if sign_changes else None,
            'first_reversal_year': sign_changes[0]['year'] if sign_changes else None,
            'rho_early_mean': float(np.mean(early)) if early else None,
            'rho_late_mean': float(np.mean(late)) if late else None,
            'rho_final': float(rho[-1]) if rho else None,
        }
        results.append(entry)

        rc = 'YES' if entry['has_reversal'] else ' NO'
        yr = f" @ {entry['first_reversal_year']:.1f}yr" if entry['has_reversal'] else ''
        print(f'  Reversal: {rc}{yr}  rho_final={entry["rho_final"]:.3f}')

        # Save checkpoint after each config
        checkpoint = [{'speed': r['speed'], 'peer': r['peer'], 'threshold': r['threshold'],
                       'has_reversal': r['has_reversal'], 'first_reversal_year': r['first_reversal_year'],
                       'rho_final': r['rho_final'], 'rho_early_mean': r['rho_early_mean'],
                       'rho_late_mean': r['rho_late_mean'], 'n_sign_changes': r['n_sign_changes']}
                      for r in results]
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=float)

        # Clear JAX cache to prevent OOM
        jax.clear_caches()
        import gc
        gc.collect()

    # Summary statistics
    n_reversal = sum(1 for r in results if r['has_reversal'])
    reversal_years = [r['first_reversal_year'] for r in results if r['has_reversal'] and r['first_reversal_year'] is not None]
    final_rhos = [r['rho_final'] for r in results if r['rho_final'] is not None]

    print(f'\n{"="*60}')
    print(f'ROBUSTNESS SUMMARY: {n_reversal}/{len(results)} configs show reversal '
          f'({100*n_reversal/len(results):.0f}%)')
    if reversal_years:
        print(f'Reversal year range: {min(reversal_years):.1f} -- {max(reversal_years):.1f}')
        print(f'Reversal year mean:  {np.mean(reversal_years):.1f}')
    print(f'Final rho range:      {min(final_rhos):.3f} -- {max(final_rhos):.3f}')
    print(f'Final rho mean:       {np.mean(final_rhos):.3f}')
    print(f'Final rho median:     {np.median(final_rhos):.3f}')

    # By-parameter breakdown
    print(f'\n--- By threshold ---')
    for t in THRESHOLDS:
        sub = [r for r in results if r['threshold'] == t]
        n_rev = sum(1 for r in sub if r['has_reversal'])
        yrs = [r['first_reversal_year'] for r in sub if r['has_reversal'] and r['first_reversal_year'] is not None]
        rhos = [r['rho_final'] for r in sub if r['rho_final'] is not None]
        print(f'  T={t:.1f}: reversal {n_rev}/{len(sub)}, '
              f'year={np.mean(yrs):.1f}±{np.std(yrs):.1f}, '
              f'rho={np.mean(rhos):.3f}')

    print(f'\n--- By speed ---')
    for s in SPEEDS:
        sub = [r for r in results if r['speed'] == s]
        n_rev = sum(1 for r in sub if r['has_reversal'])
        print(f'  speed={s:.2f}: reversal {n_rev}/{len(sub)}')

    print(f'\n--- By peer influence ---')
    for p in PEERS:
        sub = [r for r in results if r['peer'] == p]
        n_rev = sum(1 for r in sub if r['has_reversal'])
        print(f'  peer={p:.2f}: reversal {n_rev}/{len(sub)}')

    # Save
    output = {
        'grid': [{'speed': r['speed'], 'peer': r['peer'], 'threshold': r['threshold'],
                  'has_reversal': r['has_reversal'], 'first_reversal_year': r['first_reversal_year'],
                  'rho_final': r['rho_final'], 'rho_early_mean': r['rho_early_mean'],
                  'rho_late_mean': r['rho_late_mean'], 'n_sign_changes': r['n_sign_changes']}
                 for r in results],
        'summary': {
            'n_configs': len(results),
            'n_reversal': n_reversal,
            'reversal_rate': n_reversal / len(results),
            'reversal_year_min': float(min(reversal_years)) if reversal_years else None,
            'reversal_year_max': float(max(reversal_years)) if reversal_years else None,
            'reversal_year_mean': float(np.mean(reversal_years)) if reversal_years else None,
            'rho_final_mean': float(np.mean(final_rhos)),
            'rho_final_median': float(np.median(final_rhos)),
        },
    }
    with open(os.path.join(SAVE_DIR, 'reversal_robustness.json'), 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f'\nSaved: reversal_robustness.json')
    print(f'Elapsed: {time.time() - t0:.0f}s ({(time.time()-t0)/60:.1f}min)')


if __name__ == '__main__':
    main()
