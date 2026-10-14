"""Seed-level bootstrap for the E10 headline slopes and ladder ratios.

The frozen-counterfactual ladder (post_e10_frozen) and the aggregation-
convention indices (post_e10_index) report OLS slopes of per-sigma means over
the eight baseline sigma cells (50 seeds each), at the decade horizon. Those
slopes carry no seed uncertainty. This script resamples seeds with
replacement within each sigma cell (percentile bootstrap, B draws, fixed RNG
seed) and reports 95% intervals for

  - the ladder slopes L0..L3 and the surviving fractions L1/L0, L2/L0, L3/L0
  - the index slopes (ii) input-weighted inclusive and (iii) expenditure-
    share Tornquist, and their ratio

Point estimates reproduce the archived post_e10_frozen.json /
post_e10_index.json summaries by construction (same per-seed estimator, same
mean-then-OLS aggregation). Pure numpy; no simulation.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import post_e10_common as C
import post_e10_frozen as F
import post_e10_index as I

B = 2000
RNG_SEED = 20261010
HORIZON = 120
LADDERS = ['L0', 'L1', 'L2', 'L3']
INDEX_KEYS = {'ii': 'ii_simple_sum_inclusive',
              'iii': 'iii_expenditure_tornquist',
              'v': 'v_input_weight_tornquist'}


def collect(cell='baseline'):
    """Per-seed ladder and index values at the decade horizon, per sigma."""
    L = {lv: [] for lv in LADDERS}
    IX = {k: [] for k in INDEX_KEYS}
    n_per = {}
    for sigma in C.CELLS[cell]:
        rows = {lv: [] for lv in LADDERS}
        rix = {k: [] for k in INDEX_KEYS}
        for _, p in C.iter_seeds(cell, sigma):
            k = int(np.where(p['snap_ticks'] - p['snap_ticks'][0]
                             == HORIZON)[0][0])
            lad = F.ladder(p, k)
            idx = I.indices(p, k)
            for lv in LADDERS:
                rows[lv].append(lad[lv])
            for k2, field in INDEX_KEYS.items():
                rix[k2].append(idx[field])
        n_per[f'sigma={sigma}'] = len(rows['L0'])
        for lv in LADDERS:
            L[lv].append(np.array(rows[lv]))
        for k2 in INDEX_KEYS:
            IX[k2].append(np.array(rix[k2]))
    L = {lv: np.array(v) for lv, v in L.items()}        # (8 sigma, n seeds)
    IX = {k: np.array(v) for k, v in IX.items()}
    assert all(v.shape[0] == len(C.CELLS[cell]) for v in list(L.values())
               + list(IX.values()))
    return L, IX, n_per


def boot_slopes(arr, IDX, x):
    """arr (S, n), IDX (B, S, n) -> bootstrap OLS slopes (B,) and point."""
    S, n = arr.shape
    means = np.empty((S, IDX.shape[0]))
    for i in range(S):
        means[i] = arr[i][IDX[:, i, :]].mean(axis=1)
    xc = x - x.mean()
    return (xc @ means) / (xc @ xc)


def summarize(draws, thresh, greater=True):
    draws = np.asarray(draws, float)
    ok = np.isfinite(draws)
    d = draws[ok]
    return {'median': float(np.median(d)),
            'ci_lo': float(np.percentile(d, 2.5)),
            'ci_hi': float(np.percentile(d, 97.5)),
            'p_gt_thresh': float((d > thresh).mean())}


def run(cell='baseline'):
    sigmas = np.array(C.CELLS[cell], float)
    n = None
    L, IX, n_per = collect(cell)
    n = L['L0'].shape[1]

    rng = np.random.default_rng(RNG_SEED)
    IDX = rng.integers(0, n, size=(B, len(sigmas), n))

    point_idx = np.tile(np.arange(n), (1, len(sigmas), 1))  # in-sample (1,S,n)
    pt, draws = {}, {}
    for lv in LADDERS:
        draws[lv] = boot_slopes(L[lv], IDX, sigmas)
        pt[lv] = float(boot_slopes(L[lv], point_idx, sigmas)[0])
    for k in INDEX_KEYS:
        draws[k] = boot_slopes(IX[k], IDX, sigmas)
        pt[k] = float(boot_slopes(IX[k], point_idx, sigmas)[0])

    ratios_pt = {f'{lv}/L0': pt[lv] / pt['L0'] for lv in LADDERS[1:]}
    ratios_draws = {f'{lv}/L0': draws[lv] / draws['L0'] for lv in LADDERS[1:]}
    ratio_ii_draws = draws['iii'] / draws['ii']
    ratio_v_draws = draws['iii'] / draws['v']

    out = {
        'config': {'cell': cell, 'horizon_ticks': HORIZON, 'B': B,
                   'rng_seed': RNG_SEED, 'n_seeds_per_sigma': n_per,
                   'method': 'percentile bootstrap, seeds resampled with '
                             'replacement within each sigma cell; estimator '
                             'identical to post_e10_frozen/post_e10_index '
                             'summaries (per-seed value -> per-sigma mean -> '
                             'OLS slope over sigma)'},
        'point': {'slopes': pt, 'surviving_fraction': ratios_pt,
                  'iii_over_ii': pt['iii'] / pt['ii'],
                  'iii_over_v': pt['iii'] / pt['v']},
        'bootstrap': {
            'slopes': {k: {**summarize(draws[k], 0.0), 'thresh': 0.0}
                       for k in draws},
            'ratios_vs_L0': {k: {**summarize(v, 1.0), 'thresh': 1.0}
                             for k, v in ratios_draws.items()},
            'slope_iii_over_ii': {**summarize(ratio_ii_draws, 0.0),
                                  'thresh': 0.0},
            'slope_iii_over_v': {**summarize(ratio_v_draws, 0.0),
                                 'thresh': 0.0},
        },
    }
    path = C.out_path('post_e10_bootstrap.json')
    C.atomic_json(path, out)
    print(f'-> {path}')
    print(json.dumps(out['point'], indent=1))
    for sec in ['ratios_vs_L0', 'slopes', 'slope_iii_over_ii']:
        print(f'[{sec}]')
        print(json.dumps(out['bootstrap'][sec], indent=1))
    return out


if __name__ == '__main__':
    run()
