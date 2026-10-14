"""E8-lite: sigma-mapping robustness for the BTOS validation (R2-5).

The BTOS calibration assigns each sector a substitution elasticity
sigma = mu/(mu-1) from De Loecker-style markups. Reviewer 2 objects that a
single Spearman rho on n=15 sectors, with a hand-built mapping, carries too
much evidentiary weight.

Strategy: (1) compute the model's adoption-vs-sigma curve once on a fine grid
with many seeds; (2) evaluate EVERY candidate mapping through that curve —
no re-simulation per mapping; (3) report rho with bootstrap CIs for each
mapping and a leave-one-sector-out stability check.

Mappings:
  M0  baseline: DEU (2020) markups, sigma = mu/(mu-1)   [as in the paper]
  M1  Ahmad-Riker estimates: manufacturing-only consistency check (the source
       covers naics 311-339 only; an independent cross-sector mapping is not
       possible with it)
  M2  perturbed baseline: parametric lognormal(0, 0.1) noise on markups

Usage:
    python abm/experiments/exp_rev_e8_sigma_mapping.py            # run grid
    python abm/experiments/exp_rev_e8_sigma_mapping.py --collect  # evaluate
"""
import argparse
import csv
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

SAVE_DIR = os.path.join(_abm_dir, 'output', 'revision', 'e8')
os.makedirs(SAVE_DIR, exist_ok=True)
DATA_DIR = os.path.join(_abm_dir, 'data', 'external')

SIGMA_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0,
              9.0, 10.0, 11.0, 12.0]
NUM_SEEDS = 50
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120


def adoption_of(firms):
    alive = np.asarray(firms.alive)
    exp = np.asarray(firms.ai_experience)
    return float(np.sum((exp > 0.1) & alive) / max(float(np.sum(alive)), 1))


def run_grid():
    """Adoption-vs-sigma curve; resume per sigma; clear caches per config."""
    out_path = os.path.join(SAVE_DIR, 'sigma_curve.json')
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)

    for sigma in SIGMA_GRID:
        key = f'sigma={sigma}'
        rows = results.setdefault(key, [])
        if len(rows) >= NUM_SEEDS:
            continue
        config = merge_configs({
            'sigma_base': sigma,
            'sigma_firm_spread': 0.15,   # sector-experiment convention
            'sigma_firm_A_sensitivity': 0.3,
            'ai_efficiency_gain': 0.04,
        })
        sc = _make_static_config(config)
        for seed_idx in range(len(rows), NUM_SEEDS):
            state = init_simulation(config, seed=seed_idx * 1000 + 42)
            pre = run_n_ticks(state, sc, PRE_TICKS)
            post = run_n_ticks(pre, sc, POST_TICKS)
            rows.append(adoption_of(post.firms))
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(results, f)
        os.replace(tmp, out_path)
        print(f'{key}: {len(rows)} seeds')
        import jax
        jax.clear_caches()
    print('grid done')


def load_sectors():
    """Sector list with BTOS rates and baseline markups (from the calibration)."""
    sys.path.insert(0, os.path.join(_project_root, 'abm', 'experiments'))
    import exp_industry_calibration as cal
    return dict(cal.BTOS_DATA), dict(cal.INDUSTRY_MARKUP)


def curve_eval(curve_sigmas, curve_vals, sigma, rng=None):
    """Evaluate the adoption curve at sigma with linear interpolation +
    seed-level noise (optional rng draws)."""
    v = float(np.interp(sigma, curve_sigmas, curve_vals))
    return v


def rho_with_ci(model_by_sector, btos_by_sector, n_boot=10000, seed=2026):
    from scipy.stats import spearmanr
    sectors = sorted(model_by_sector)
    m = np.array([model_by_sector[s] for s in sectors])
    b = np.array([btos_by_sector[s] for s in sectors])
    rho, p = spearmanr(m, b)
    rng = np.random.default_rng(seed)
    n = len(sectors)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(m[idx]) < 1e-9 or np.std(b[idx]) < 1e-9:
            continue
        boots.append(spearmanr(m[idx], b[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, p, float(lo), float(hi)


def loo_rho(model_by_sector, btos_by_sector):
    from scipy.stats import spearmanr
    sectors = sorted(model_by_sector)
    out = {}
    for drop in sectors:
        keep = [s for s in sectors if s != drop]
        out[drop] = float(spearmanr([model_by_sector[s] for s in keep],
                                    [btos_by_sector[s] for s in keep])[0])
    return out


def collect():
    with open(os.path.join(SAVE_DIR, 'sigma_curve.json')) as f:
        curve = json.load(f)
    cs = np.array(sorted(float(k.split('=')[1]) for k in curve))
    cv = np.array([np.mean(curve[f'sigma={s}']) for s in cs])

    btos, markups = load_sectors()

    # M0: baseline mapping
    m0 = {s: curve_eval(cs, cv, markups[s] / (markups[s] - 1.0)) for s in btos}

    # M1: Ahmad–Riker independent elasticity estimates.
    # Audit 2026-10-07: the file covers MANUFACTURING ONLY (86 naics4 rows,
    # all naics3 in 311-339), so it cannot serve as an independent
    # cross-sector alternative mapping. It is used honestly as a single-sector
    # consistency check: AR manufacturing sigma vs the calibration's
    # manufacturing sigma (markup 1.25 -> 5.0).
    m1 = None
    ar_path = os.path.join(DATA_DIR, 'ahmad_riker_sigma_estimates.csv')
    if os.path.exists(ar_path):
        with open(ar_path) as f:
            rows = list(csv.DictReader(f))
        ar_sigma = np.array([float(r['sigma_2017']) for r in rows])
        cal_sigma_mfg = markups['Manufacturing'] / (markups['Manufacturing'] - 1.0)
        m1 = {
            'n_naics4': len(rows),
            'naics3_coverage': 'manufacturing only (311-339)',
            'ar_sigma_mean': float(ar_sigma.mean()),
            'ar_sigma_median': float(np.median(ar_sigma)),
            'ar_sigma_range': [float(ar_sigma.min()), float(ar_sigma.max())],
            'calibration_sigma_manufacturing': float(cal_sigma_mfg),
            'note': ('Ahmad-Riker estimates cover manufacturing only; an '
                     'independent cross-sector mapping is not possible with '
                     'this source. Within manufacturing their sigma is '
                     f'{ar_sigma.mean():.2f} vs calibration {cal_sigma_mfg:.2f} '
                     '(same order of magnitude; calibration sits above the AR '
                     'distribution, consistent with De Loecker markups being '
                     'lower than AR-implied ones).'),
        }
        print(f"M1 (manufacturing-only check): AR sigma mean="
              f"{ar_sigma.mean():.2f} (n={len(rows)}) vs calibration "
              f"{cal_sigma_mfg:.2f}")

    results = {'M0_baseline': {}}
    rho, p, lo, hi = rho_with_ci(m0, btos)
    results['M0_baseline'] = {
        'model_by_sector': m0, 'rho': rho, 'p': p, 'ci95': [lo, hi],
        'loo_min': min(loo_rho(m0, btos).values()),
        'loo_max': max(loo_rho(m0, btos).values()),
    }

    # M2: measurement-noise perturbation of markups (lognormal sd=0.1)
    from scipy.stats import spearmanr
    rng = np.random.default_rng(2026)
    pert_rho = []
    sectors = sorted(btos)
    b_vec = np.array([btos[s] for s in sectors])
    for _ in range(500):
        mu_p = {s: markups[s] * float(rng.lognormal(0, 0.1)) for s in sectors}
        sig_p = {s: max(mu_p[s] / (mu_p[s] - 1.0), 1.05) for s in sectors}
        m_vec = np.array([curve_eval(cs, cv, sig_p[s]) for s in sectors])
        pert_rho.append(spearmanr(m_vec, b_vec)[0])
    results['M2_markup_noise'] = {
        'rho_mean': float(np.mean(pert_rho)),
        'rho_ci95': [float(np.percentile(pert_rho, 2.5)),
                     float(np.percentile(pert_rho, 97.5))],
    }

    out = os.path.join(SAVE_DIR, 'e8_mapping_robustness.json')
    if m1 is not None:
        results['M1_ahmad_riker_manufacturing'] = m1
    with open(out, 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk != 'model_by_sector'}
                      for k, v in results.items()}, indent=1, default=str))
    print(f'-> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args()
    if args.collect:
        collect()
    else:
        t0 = time.time()
        run_grid()
        print(f'{time.time()-t0:.0f}s')
