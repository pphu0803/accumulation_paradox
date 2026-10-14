"""E1: TFP measure separation + three-factor decomposition (gate x sigma x quality).

Motivation (reviewer points R1-TFP / R2-O5): the headline macro TFP metric
    TFP_raw = sum_i Q_i / sum_i (K_i^a L_i^(1-a)),   Q_i = A_eff_i * q_i^theta * inputs_i
conflates the documented AI efficiency channel (A_eff = A_base * ai multiplier)
with an undocumented product-quality channel (q_i^theta). This experiment
reports both measures side by side and ablates the quality channel
(quality_weight theta = 0) to isolate its contribution.

Design: {gate_on, gate_off} x {quality_on, quality_off} x sigma grid x NUM_SEEDS.
Gate on/off and sigma grid follow exp_decomposition.py exactly so numbers are
comparable with the archived decomposition run.

Usage:
    python abm/experiments/exp_rev_e1_tfp_decomposition.py --worker 0..3
    python abm/experiments/exp_rev_e1_tfp_decomposition.py --collect
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

SAVE_DIR = os.path.join(_abm_dir, 'output', 'revision', 'e1')
CELL_DIR = os.path.join(SAVE_DIR, 'cells')
os.makedirs(CELL_DIR, exist_ok=True)

GATE_CONDITIONS = {
    'gate_on': {
        'adoption_readiness_speed': 0.02,
        'adoption_peer_influence': 0.05,
        'adoption_threshold': 0.8,
    },
    'gate_off': {
        'adoption_readiness_speed': 0.0,
        'adoption_peer_influence': 0.0,
        'adoption_threshold': 0.0,
    },
}
QUALITY_CONDITIONS = {
    'quality_on': {'quality_weight': 0.15},   # default theta
    'quality_off': {'quality_weight': 0.0},   # channel ablated
}
SIGMA_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
GAIN = 0.04
NUM_SEEDS = 50
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120
SIGMA_FIRM_SPREAD = 0.4
SIGMA_FIRM_A_SENS = 0.5
COMBOS = [(g, q) for g in GATE_CONDITIONS for q in QUALITY_CONDITIONS]


def ai_tfp_multiplier(ai_experience, gain, j_curve_depth):
    """Firm AI multiplier on A_base; identical formula to step.py::_ai_tfp_multiplier."""
    x = np.asarray(ai_experience)
    m = 1.0 + gain * ((1.0 + j_curve_depth) * x - j_curve_depth)
    return np.where(x > 0, m, 1.0)


def macro_tfp_ai(firms, gain, j_curve_depth, alpha):
    """Quality-free macro TFP: input-weighted average of A_eff.

    TFP_ai = sum_i A_eff_i * w_i,  w_i = inputs_i / sum(inputs),
    so that TFP_raw = TFP_ai * (quality-adjusted rescale) exactly separates
    the AI channel from the quality channel up to the covariance term we
    report separately as q_ratio.
    """
    alive = np.asarray(firms.alive)
    K = np.maximum(np.asarray(firms.K), 0.01)
    L = np.maximum(np.asarray(firms.L), 0.01)
    inputs = K ** alpha * L ** (1 - alpha)
    mult = ai_tfp_multiplier(np.asarray(firms.ai_experience), gain, j_curve_depth)
    a_eff = np.asarray(firms.A_base) * mult
    m = alive & (inputs > 0.01)
    return float(np.sum(a_eff[m] * inputs[m]) / max(np.sum(inputs[m]), 1e-6))


def macro_tfp_raw(firms, alpha, theta, gain, j_curve_depth):
    """Headline (quality-inclusive) macro TFP, recomputed from the final state.

    Audit 2026-10-07: Q is recomputed as A_eff * q^theta * K^alpha L^(1-alpha)
    rather than read from firms.Q. The recorded Q is produced in S3 with the
    within-tick K/L, while the final state's K/L are end-of-tick values (S6/S9
    run after production) and newborn entrants carry Q=0 — mixing vintages
    biased the level by 3.5-6% and made the quality_off zero-check fail by
    up to ~2.4pp. With the recomputed Q the quality_off cell satisfies
    delta_tfp_raw == delta_tfp_ai exactly.
    """
    alive = np.asarray(firms.alive)
    K = np.maximum(np.asarray(firms.K), 0.01)
    L = np.maximum(np.asarray(firms.L), 0.01)
    inputs = K ** alpha * L ** (1 - alpha)
    mult = ai_tfp_multiplier(np.asarray(firms.ai_experience), gain, j_curve_depth)
    a_eff = np.asarray(firms.A_base) * mult
    q = np.maximum(np.asarray(firms.product_quality), 0.1)
    Qhat = a_eff * q ** theta * inputs
    m = alive & (inputs > 0.01)
    return float(np.sum(Qhat[m]) / max(np.sum(inputs[m]), 1e-6))


def run_one_cell(gate_label, quality_label, checkpoint_every=10):
    """Run sigma grid x seeds for one (gate, quality) combo; checkpoint to JSON."""
    out_path = os.path.join(CELL_DIR, f'{gate_label}_{quality_label}.json')
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f'[{gate_label}_{quality_label}] resuming: '
              f'{sum(len(v) for v in results.values())} seeds already done')

    gate_params = GATE_CONDITIONS[gate_label]
    quality_params = QUALITY_CONDITIONS[quality_label]

    for sigma in SIGMA_VALUES:
        key = f'sigma={sigma}'
        done = len(results.get(key, []))
        if done >= NUM_SEEDS:
            continue
        config = merge_configs({
            'sigma_base': sigma,
            'sigma_firm_spread': SIGMA_FIRM_SPREAD,
            'sigma_firm_A_sensitivity': SIGMA_FIRM_A_SENS,
            'ai_efficiency_gain': GAIN,
            **gate_params,
            **quality_params,
        })
        sc = _make_static_config(config)
        gain = float(sc.ai_efficiency_gain)
        depth = float(sc.j_curve_depth)
        alpha = float(sc.alpha_CD)

        rows = results.setdefault(key, [])
        t0 = time.time()
        for seed_idx in range(done, NUM_SEEDS):
            seed_key = seed_idx * 1000 + 42
            state = init_simulation(config, seed=seed_key)
            pre = run_n_ticks(state, sc, PRE_TICKS)
            post = run_n_ticks(pre, sc, POST_TICKS)

            pre_ai = macro_tfp_ai(pre.firms, gain, depth, alpha)
            post_ai = macro_tfp_ai(post.firms, gain, depth, alpha)
            theta = float(sc.quality_weight)
            pre_raw = macro_tfp_raw(pre.firms, alpha, theta, gain, depth)
            post_raw = macro_tfp_raw(post.firms, alpha, theta, gain, depth)

            alive = np.asarray(post.firms.alive)
            markups = np.asarray(post.firms.effective_markup)[alive]
            shares = np.where(alive, np.asarray(post.firms.market_share), 0.0)
            adoption = float(np.sum(
                (np.asarray(post.firms.ai_experience) > 0.1) & alive)
                / max(float(np.sum(alive)), 1))

            delta_raw = (post_raw - pre_raw) / pre_raw
            delta_ai = (post_ai - pre_ai) / pre_ai
            rows.append({
                'seed': seed_idx,
                'delta_tfp_raw': delta_raw,
                'delta_tfp_ai': delta_ai,
                'annualized_tfp_ai': (1 + delta_ai) ** (12.0 / POST_TICKS) - 1,
                'quality_ratio_post': post_raw / post_ai,   # >1 iff quality channel active
                'adoption_rate': adoption,
                'hhi': float(np.sum(shares ** 2)),
                'mean_effective_markup': float(np.mean(markups)) if markups.size else None,
                'min_effective_markup': float(np.min(markups)) if markups.size else None,
                'n_alive': int(np.sum(alive)),
            })
            if (seed_idx + 1) % checkpoint_every == 0 or seed_idx + 1 == NUM_SEEDS:
                _atomic_json(out_path, results)
        print(f'[{gate_label}_{quality_label}] {key}: {NUM_SEEDS} seeds '
              f'in {time.time()-t0:.0f}s')

    _atomic_json(out_path, results)
    return out_path


def _atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def collect():
    """Merge cells and write per-cell summaries with 95% CIs + channel table."""
    from scipy import stats as sps

    merged = {}
    for g in GATE_CONDITIONS:
        for q in QUALITY_CONDITIONS:
            p = os.path.join(CELL_DIR, f'{g}_{q}.json')
            with open(p) as f:
                merged[f'{g}|{q}'] = json.load(f)

    summary = {}
    for combo, cells in merged.items():
        for key, rows in cells.items():
            entry = {}
            for field in ['delta_tfp_raw', 'delta_tfp_ai', 'annualized_tfp_ai',
                          'quality_ratio_post', 'adoption_rate', 'hhi',
                          'mean_effective_markup']:
                vals = [r[field] for r in rows if r.get(field) is not None]
                n = len(vals)
                mean = float(np.mean(vals))
                sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                crit = float(sps.t.ppf(0.975, n - 1)) if n > 1 else 0.0
                entry[field] = {'mean': mean, 'std': float(np.std(vals, ddof=1)),
                                'ci95': [mean - crit * sem, mean + crit * sem],
                                'n': n}
            summary.setdefault(combo, {})[key] = entry

    # Channel attribution per sigma (gate_on, quality_on vs ablations)
    attribution = {}
    for sigma in SIGMA_VALUES:
        key = f'sigma={sigma}'
        row = {}
        for combo, cells in summary.items():
            row[combo] = {
                'delta_tfp_raw': cells[key]['delta_tfp_raw']['mean'],
                'delta_tfp_ai': cells[key]['delta_tfp_ai']['mean'],
                'adoption': cells[key]['adoption_rate']['mean'],
            }
        q_share = {g: (row[f'{g}|quality_on']['delta_tfp_raw']
                       - row[f'{g}|quality_on']['delta_tfp_ai'])
                   for g in GATE_CONDITIONS}
        attribution[key] = {
            'cells': row,
            'quality_channel_pp': q_share,
            'gate_effect_ai_pp': {
                q: row[f'gate_off|{q}']['delta_tfp_ai']
                   - row[f'gate_on|{q}']['delta_tfp_ai'] for q in QUALITY_CONDITIONS},
            'markup_gradient_ai_pp_per_sigma': {
                q: float(np.polyfit(
                    SIGMA_VALUES,
                    [summary[f'gate_on|{q}'][f'sigma={s}']['delta_tfp_ai']['mean']
                     for s in SIGMA_VALUES], 1)[0])
                for q in QUALITY_CONDITIONS},
        }

    out = {
        'meta': {
            'gain': GAIN, 'num_seeds': NUM_SEEDS, 'pre_ticks': PRE_TICKS,
            'post_ticks': POST_TICKS, 'sigma_values': SIGMA_VALUES,
            'sigma_firm_spread': SIGMA_FIRM_SPREAD,
            'sigma_firm_A_sensitivity': SIGMA_FIRM_A_SENS,
            'seed_rule': 'seed_idx*1000+42',
        },
        'summary': summary,
        'attribution': attribution,
    }
    out_path = os.path.join(SAVE_DIR, 'e1_tfp_decomposition.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'collected -> {out_path}')

    # Console digest
    print('\n=== delta TFP: raw vs AI-only (gate_on) ===')
    print('sigma |  raw%   ai%   quality_pp | adopt%')
    for sigma in SIGMA_VALUES:
        a = attribution[f'sigma={sigma}']['cells']['gate_on|quality_on']
        qpp = (a['delta_tfp_raw'] - a['delta_tfp_ai']) * 100
        print(f"{sigma:5.1f} | {a['delta_tfp_raw']*100:5.1f} {a['delta_tfp_ai']*100:5.1f} "
              f"{qpp:8.1f} | {a['adoption']*100:5.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int, default=None,
                    help='run COMBOS[worker] only (0..3)')
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args()

    if args.collect:
        collect()
    else:
        if args.worker is None:
            for i in range(len(COMBOS)):
                run_one_cell(*COMBOS[i])
        else:
            run_one_cell(*COMBOS[args.worker])
