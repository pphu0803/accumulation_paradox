"""Experiment: Out-of-Sample Validation with McKinsey 2025 Data

Maps McKinsey global AI adoption rates (2025) to model σ values,
runs the model, and computes correlation with an independent data source.
Also tests time-series prediction: McKinsey 2022→2025 adoption trajectories.
"""

import sys
import os
import json
import time

import numpy as np
from scipy.stats import spearmanr, pearsonr

_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_abm_dir, '..'))
for _p in (_project_root, _abm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from abm.src.state import init_simulation
from abm.src.step import _make_static_config, run_n_ticks, run_n_ticks_collect
from abm.src.config import merge_configs

import jax.numpy as jnp

SAVE_DIR = os.path.join(_abm_dir, 'output', 'robustness')
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Data sources
# ============================================================

# McKinsey 2025 adoption rates (global, % of orgs using AI in >=1 function)
# Source: McKinsey Rewiring + Agents 2025, IBM AI Adoption Index, Stanford AI Index
MCKINSEY_2025 = {
    'Technology': 92.0,
    'Financial Services': 80.0,
    'Professional Services': 82.0,
    'Healthcare': 65.0,
    'Media and Telecom': 74.0,
    'Retail': 68.0,
    'Energy': 66.0,
    'Manufacturing': 65.0,
    'Education': 58.0,
    'Government': 50.0,
    'Agriculture': 42.0,
    'Logistics': 63.0,
    'Environment': 44.0,
    'Legal Services': 52.0,
    'Real Estate': 46.0,
}

# McKinsey time series (2022-2025)
# Columns: [2022, 2023, early 2024, late 2024, 2025]
MCKINSEY_TIMESERIES = {
    'Technology':                [70, 76, 84, 88, 92],
    'Financial Services':        [55, 61, 70, 74, 80],
    'Professional Services':    [50, 57, 72, 76, 82],
    'Healthcare':               [35, 42, 52, 58, 65],
    'Media and Telecom':        [48, 55, 65, 68, 74],
    'Retail':                   [38, 45, 56, 60, 68],
    'Energy':                  [32, 40, 55, 59, 66],
    'Manufacturing':            [40, 46, 55, 59, 65],
    'Education':                [22, 30, 42, 48, 58],
    'Government':               [20, 28, 36, 42, 50],
    'Agriculture':              [15, 20, 29, 34, 42],
    'Logistics':                [30, 38, 50, 55, 63],
    'Environment':              [12, 18, 28, 34, 44],
    'Legal Services':           [15, 22, 35, 42, 52],
    'Real Estate':              [18, 24, 32, 38, 46],
}

# BTOS markup-based sigma mapping — derived from the SAME INDUSTRY_MARKUP
# dict used by exp_industry_calibration (audit 2026-10-07: this table used to
# hold hand-copied values that disagreed with the calibration, e.g. Finance
# 3.13 vs 2.67, Educational Services 4.29 vs 6.00, while claiming
# "from calibration").
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_industry_calibration as _cal
BTOS_SIGMA = {s: m / (m - 1.0) for s, m in _cal.INDUSTRY_MARKUP.items()}

# DEU 2020 markup-based sigma for additional McKinsey sectors
DEU_MARKUP_SIGMA = {
    'Mining Quarrying Oil Gas': 1.49 / (1.49 - 1),   # σ = 3.04
    'Utilities': 1.41 / (1.41 - 1),                    # σ = 3.44
    'Agriculture Forestry': 1.81 / (1.81 - 1),          # σ = 2.23
}

# Mapping: McKinsey sector → σ (using BTOS where possible, DEU for rest)
SECTOR_SIGMA_MAP = {
    'Technology':               BTOS_SIGMA['Information'],          # closest match
    'Financial Services':       BTOS_SIGMA['Finance and Insurance'],
    'Professional Services':     BTOS_SIGMA['Professional Services'],
    'Healthcare':               BTOS_SIGMA['Health Care'],
    'Media and Telecom':        BTOS_SIGMA['Information'],          # subset of Information
    'Retail':                   BTOS_SIGMA['Retail Trade'],
    'Energy':                  DEU_MARKUP_SIGMA['Mining Quarrying Oil Gas'],  # high markup → low σ
    'Manufacturing':            BTOS_SIGMA['Manufacturing'],
    'Education':                BTOS_SIGMA['Educational Services'],
    'Government':               5.5,  # estimate: moderate competition, public sector
    'Agriculture':              DEU_MARKUP_SIGMA['Agriculture Forestry'],
    'Logistics':                BTOS_SIGMA['Transportation'],
    'Environment':              BTOS_SIGMA['Real Estate and Rental'],   # no good match
    'Legal Services':           BTOS_SIGMA['Professional Services'],     # subset
    'Real Estate':              BTOS_SIGMA['Real Estate and Rental'],
}

NUM_SEEDS = 15
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120
SAMPLE_EVERY = 6
SIGMA_FIRM_SPREAD = 0.4
SIGMA_FIRM_A_SENS = 0.5


def run_sector(sigma, n_seeds=NUM_SEEDS):
    """Run model for one sigma value, return mean adoption and stats."""
    config = merge_configs({
        'sigma_base': sigma,
        'sigma_firm_spread': SIGMA_FIRM_SPREAD,
        'sigma_firm_A_sensitivity': SIGMA_FIRM_A_SENS,
    })
    sc = _make_static_config(config)

    adoptions = []
    for seed_idx in range(n_seeds):
        seed_key = seed_idx * 1000 + 42
        state = init_simulation(config, seed=seed_key)
        pre = run_n_ticks(state, sc, PRE_TICKS)
        final = run_n_ticks(pre, sc, POST_TICKS)
        alive = final.firms.alive
        a = float(jnp.sum((final.firms.ai_experience > 0.1) & alive) /
                 max(float(jnp.sum(alive)), 1))
        adoptions.append(a)

    return {
        'mean': float(np.mean(adoptions)),
        'std': float(np.std(adoptions)),
        'cv': float(np.std(adoptions) / max(np.mean(adoptions), 0.01)),
    }


def run_oos_validation():
    """Run out-of-sample validation against McKinsey 2025 data."""

    sectors = list(MCKINSEY_2025.keys())
    sigma_values = [SECTOR_SIGMA_MAP[s] for s in sectors]
    btos_equivalent = [s for s in sectors]
    mck_arr = np.array([MCKINSEY_2025[s] for s in sectors])

    # Compute Spearman with all sectors (including non-BTOS)
    model_adopt = []
    for i, sector in enumerate(sectors):
        print(f'  [{i+1}/{len(sectors)}] {sector:30s} sigma={sigma_values[i]:.2f}', end='', flush=True)
        result = run_sector(sigma_values[i])
        model_adopt.append(result['mean'])
        print(f'  adopt={result["mean"]*100:.1f}±{result["std"]*100:.1f}%  CV={result["cv"]:.2f}')

    model_arr = np.array(model_adopt)
    rho, p = spearmanr(model_arr, mck_arr)
    r_val, p_val = pearsonr(model_arr, mck_arr)

    print(f'\n=== All {len(sectors)} McKinsey sectors ===')
    print(f'  Spearman rho = {rho:.3f} (p = {p:.4f})')
    print(f'  Pearson r  = {r_val:.3f} (p = {p_val:.4f})')

    # Compare: BTOS-mapped sectors only. Audit 2026-10-07: alias sectors
    # (Media and Telecom -> Information, Legal Services -> Professional
    # Services) were dropped — they re-run a σ already covered by another
    # sector, adding tied duplicates rather than independent information.
    BTOS_MAPPED_SECTORS = {
        'Technology', 'Financial Services', 'Professional Services', 'Healthcare',
        'Retail', 'Manufacturing', 'Education', 'Logistics',
    }
    btos_mask = [s in BTOS_MAPPED_SECTORS for s in sectors]
    n_btos = sum(btos_mask)
    rho_b, p_b = None, None

    if n_btos >= 5:
        btos_model = [model_adopt[i] for i in range(len(sectors)) if btos_mask[i]]
        btos_mck = [mck_arr[i] for i in range(len(sectors)) if btos_mask[i]]
        rho_b, p_b = spearmanr(btos_model, btos_mck)
        print(f'\n=== BTOS-mapped sectors only (n={n_btos}) ===')
        print(f'  Spearman rho = {rho_b:.3f} (p = {p_b:.4f})')
    else:
        print(f'\n=== BTOS-mapped sectors: only {n_btos}, skipping ===')

    # Growth comparison — DESCRIPTIVE ONLY (audit 2026-10-07): model values
    # are end-state adoption LEVELS, not growth rates, and McKinsey growth
    # and 2025 level correlate at about -0.90 across sectors, so this mostly
    # restates the level test with a flipped sign. Kept for transparency,
    # not a mechanism test.
    mck_growth = {}
    for s in sectors:
        ts = MCKINSEY_TIMESERIES.get(s)
        if ts and ts[0] > 0:
            mck_growth[s] = (ts[-1] - ts[0]) / ts[0] * 100

    rho_g = r_g = p_g_spearman = p_g_pearson = None
    if mck_growth:
        print(f'\n=== Growth comparison (descriptive, see note in JSON) ===')
        growth_mck = [mck_growth.get(s, 0) for s in sectors]

        valid = [(i, growth_mck[i]) for i in range(len(sectors))
                 if isinstance(growth_mck[i], (int, float)) and growth_mck[i] != 0]
        if len(valid) >= 3:
            g_arr = [g for _, g in valid]
            m_arr = [model_adopt[i] / 100.0 for i, _ in valid]
            rho_g, p_g_spearman = spearmanr(m_arr, g_arr)
            r_g, p_g_pearson = pearsonr(m_arr, g_arr)
            print(f'  Model adoption level vs McKinsey growth: '
                  f'Spearman rho={rho_g:.3f} (p={p_g_spearman:.4f})')
            print(f'  Pearson r={r_g:.3f} (p={p_g_pearson:.4f})')

    return {
        'all_sectors': {
            'n': len(sectors),
            'spearman_rho': rho, 'spearman_p': p,
            'pearson_r': r_val, 'pearson_p': p_val,
            'sectors': sectors,
            'sigma': sigma_values,
            'mckinsey_2025': mck_arr.tolist(),
            'model_adoption': model_arr.tolist(),
        },
        'btos_matched': {
            'n': n_btos,
            'spearman_rho': rho_b, 'spearman_p': p_b,
        } if rho_b is not None else {'n': n_btos, 'spearman_rho': None, 'spearman_p': None},
        'growth_comparison': {
            'spearman_rho': rho_g, 'spearman_p': p_g_spearman,
            'pearson_r': r_g, 'pearson_p': p_g_pearson,
            'note': ('descriptive only: model values are end-state levels; '
                     'McKinsey growth correlates ~-0.90 with the 2025 level, '
                     'so this largely restates the level test with flipped sign'),
        },
    }


def main():
    t0 = time.time()
    print('=' * 60)
    print('OUT-OF-SAMPLE VALIDATION: McKinsey 2025')
    print('=' * 60)

    result = run_oos_validation()

    with open(os.path.join(SAVE_DIR, 'oos_validation.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f'\nSaved: oos_validation.json')
    print(f'Elapsed: {time.time() - t0:.0f}s ({(time.time()-t0)/60:.1f}min)')


if __name__ == '__main__':
    main()
