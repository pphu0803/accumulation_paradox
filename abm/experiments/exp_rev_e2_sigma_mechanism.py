"""E2: sigma -> TFP mechanism diagnostics.

Answers two reviewer questions (reviewer points R1-TFP / R1-Fig3):
  (a) Why does macro TFP rise with sigma while AI adoption falls?
  (b) Through which model quantities does market structure (sigma) affect
      aggregate TFP at all?

For each sigma we record, per seed:
  - time paths (every 12 ticks): TFP_raw, TFP_ai, adoption, HHI, price, wage,
    labor share, mean effective markup
  - pre/post shift-share decomposition of TFP_ai into within-firm gains,
    input reallocation, interaction, and exit
  - pass-through elasticity: median over surviving firms of dlog(P)/dlog(MC)
  - per-adopter A_eff gain and adopter/non-adopter A_eff gap
  - A_eff x input-share covariance (allocation efficiency)

Usage:
    python abm/experiments/exp_rev_e2_sigma_mechanism.py
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
from abm.src.step import run_n_ticks, run_n_ticks_collect, _make_static_config
from abm.src.config import merge_configs

SAVE_DIR = os.path.join(_abm_dir, 'output', 'revision', 'e2')
os.makedirs(SAVE_DIR, exist_ok=True)

SIGMA_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
GAIN = 0.04
NUM_SEEDS = 50
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120
SAMPLE_EVERY = 12
SIGMA_FIRM_SPREAD = 0.4
SIGMA_FIRM_A_SENS = 0.5
ADOPT_EXP = 0.1


def ai_tfp_multiplier(x, gain, depth):
    m = 1.0 + gain * ((1.0 + depth) * np.asarray(x) - depth)
    return np.where(np.asarray(x) > 0, m, 1.0)


def cost_multiplier(x, invest, strategy, sc):
    """Mirror of step.py::_ai_cost_multiplier."""
    s = np.asarray(strategy)
    m = (1.0 - float(sc.ai_cost_experience_coeff) * np.asarray(x)
         - np.asarray(invest) * (1.0 - s) * float(sc.ai_cost_elasticity))
    return np.where(np.asarray(x) > 0, np.maximum(m, 0.5), 1.0)


def firm_arrays(firms, sc):
    alive = np.asarray(firms.alive)
    K = np.maximum(np.asarray(firms.K), 0.01)
    L = np.maximum(np.asarray(firms.L), 0.01)
    inputs = K ** float(sc.alpha_CD) * L ** (1 - float(sc.alpha_CD))
    mult = ai_tfp_multiplier(firms.ai_experience, sc.ai_efficiency_gain,
                             sc.j_curve_depth)
    a_eff = np.asarray(firms.A_base) * mult
    return alive, inputs, a_eff


def macro_tfp_components(firms, sc):
    alive, inputs, a_eff = firm_arrays(firms, sc)
    m = alive & (inputs > 0.01)
    inp = inputs[m]
    w = inp / max(np.sum(inp), 1e-6)
    tfp_ai = float(np.sum(a_eff[m] * w))
    # Audit 2026-10-07: recompute Q from the same (final-state) K/L instead of
    # reading firms.Q — the recorded Q uses within-tick K/L while entrants
    # carry Q=0, mixing vintages (3.5-6% level bias).
    q = np.maximum(np.asarray(firms.product_quality), 0.1)[m]
    theta = float(sc.quality_weight)
    Qhat = a_eff[m] * q ** theta * inp
    tfp_raw = float(np.sum(Qhat) / max(np.sum(inp), 1e-6))
    # allocation: input-share-weighted covariance between A_eff and share deviations
    cov_alloc = float(np.sum(w * a_eff[m]) - np.mean(a_eff[m]) * np.sum(w))
    return tfp_raw, tfp_ai, cov_alloc


def shift_share(pre_firms, post_firms, sc):
    """Decompose dTFP_ai exactly into within / reallocation / interaction /
    exit / entry. Survivors telescope; dying firms carry an explicit exit term
    (their fictional A_eff=0 must not leak into the interaction term);
    newborn firms carry an explicit entry term.

    Identity: within + realloc + interact + exit + entry == TFP_post - TFP_pre.
    """
    a_alive, a_inputs, a_eff = firm_arrays(pre_firms, sc)
    p_alive, p_inputs, p_eff = firm_arrays(post_firms, sc)
    m_pre = a_alive & (a_inputs > 0.01)
    inp = a_inputs[m_pre]
    w_pre = inp / max(np.sum(inp), 1e-6)
    base = a_eff[m_pre]

    idx_post = np.where(p_alive & (p_inputs > 0.01))[0]
    tot_post = np.sum(p_inputs[idx_post]) if idx_post.size else 1.0
    common = np.where(m_pre)[0]
    pos = {g: k for k, g in enumerate(common)}

    w_post = np.zeros_like(w_pre)
    aeff_post = np.zeros_like(w_pre)
    still = np.zeros_like(w_pre, dtype=bool)
    for g in idx_post:
        if g in pos:
            k = pos[g]
            w_post[k] = p_inputs[g] / tot_post
            aeff_post[k] = p_eff[g]
            still[k] = True

    da = np.where(still, aeff_post - base, 0.0)
    dw = np.where(still, w_post - w_pre, 0.0)
    within = float(np.sum(w_pre * da))
    realloc = float(np.sum(dw * base))
    interact = float(np.sum(dw * da))
    exit_loss = float(-np.sum(w_pre * base * ~still))
    entry = float(np.sum(
        p_inputs[idx_post[~np.isin(idx_post, common)]] /
        tot_post * p_eff[idx_post[~np.isin(idx_post, common)]]))
    return {'within': within, 'reallocation': realloc,
            'interaction': interact, 'exit': exit_loss, 'entry': entry}


def passthrough(pre_firms, post_firms, market_pre, market_post, sc):
    """Median surviving-firm elasticity dlog(P)/dlog(MC), pre -> post."""
    a_alive, _, _ = firm_arrays(pre_firms, sc)
    p_alive, _, _ = firm_arrays(post_firms, sc)
    K = np.maximum(np.asarray(pre_firms.K), 0.01)
    L = np.maximum(np.asarray(pre_firms.L), 0.01)
    Q_pre = np.maximum(np.asarray(pre_firms.Q), 1e-3)
    Q_post = np.maximum(np.asarray(post_firms.Q), 1e-3)
    cm_pre = cost_multiplier(pre_firms.ai_experience, pre_firms.ai_investment,
                             pre_firms.ai_strategy, sc)
    cm_post = cost_multiplier(post_firms.ai_experience, post_firms.ai_investment,
                              post_firms.ai_strategy, sc)
    w_pre, w_post = float(market_pre.wage_rate), float(market_post.wage_rate)
    d = float(sc.depreciation_rate)
    mc_pre = (w_pre * np.asarray(pre_firms.L) + d * K) / Q_pre * cm_pre
    Kp = np.maximum(np.asarray(post_firms.K), 0.01)
    Lp = np.maximum(np.asarray(post_firms.L), 0.01)
    mc_post = (w_post * Lp + d * Kp) / Q_post * cm_post

    surv = a_alive & p_alive
    lp = np.log(np.maximum(np.asarray(post_firms.effective_markup) * mc_post, 1e-6))
    lm = np.log(np.maximum(mc_pre, 1e-6))
    dlogp = lp - np.log(np.maximum(np.asarray(pre_firms.effective_markup) * mc_pre, 1e-6))
    dlogmc = np.log(np.maximum(mc_post, 1e-6)) - lm
    ok = surv & np.isfinite(dlogp) & np.isfinite(dlogmc) & (np.abs(dlogmc) > 1e-9)
    if not np.any(ok):
        return None
    return float(np.median(dlogp[ok] / dlogmc[ok]))


def run_seed(sigma, seed_idx):
    config = merge_configs({
        'sigma_base': sigma,
        'sigma_firm_spread': SIGMA_FIRM_SPREAD,
        'sigma_firm_A_sensitivity': SIGMA_FIRM_A_SENS,
        'ai_efficiency_gain': GAIN,
    })
    sc = _make_static_config(config)
    state = init_simulation(config, seed=seed_idx * 1000 + 42)
    pre = run_n_ticks(state, sc, PRE_TICKS)
    final, sampled = run_n_ticks_collect(pre, sc, POST_TICKS,
                                         sample_every=SAMPLE_EVERY)

    # trajectories (sampled states are stacked pytrees: field[i] = i-th sample)
    import types
    n_s = sampled.firms.alive.shape[0]
    traj = {'tick': [], 'tfp_raw': [], 'tfp_ai': [], 'adoption': [], 'hhi': []}
    first_adopt_tick = None
    for i in range(n_s):
        firms_i = types.SimpleNamespace(
            alive=sampled.firms.alive[i],
            K=sampled.firms.K[i],
            L=sampled.firms.L[i],
            Q=sampled.firms.Q[i],
            A_base=sampled.firms.A_base[i],
            ai_experience=sampled.firms.ai_experience[i],
            product_quality=sampled.firms.product_quality[i],
            market_share=sampled.firms.market_share[i],
        )
        tfp_raw, tfp_ai, _ = macro_tfp_components(firms_i, sc)
        alive = np.asarray(sampled.firms.alive[i])
        adopt = float(np.sum((np.asarray(sampled.firms.ai_experience[i]) > ADOPT_EXP) & alive)
                      / max(float(np.sum(alive)), 1))
        shares = np.where(alive, np.asarray(sampled.firms.market_share[i]), 0.0)
        traj['tick'].append((i + 1) * SAMPLE_EVERY)
        traj['tfp_raw'].append(tfp_raw)
        traj['tfp_ai'].append(tfp_ai)
        traj['adoption'].append(adopt)
        traj['hhi'].append(float(np.sum(shares ** 2)))
        if first_adopt_tick is None and adopt > 0.05:
            first_adopt_tick = (i + 1) * SAMPLE_EVERY

    pre_raw, pre_ai, _ = macro_tfp_components(pre.firms, sc)
    post_raw, post_ai, post_cov = macro_tfp_components(final.firms, sc)

    a_alive, _, a_eff = firm_arrays(pre.firms, sc)
    p_alive, _, p_eff = firm_arrays(final.firms, sc)
    adopters = a_alive & p_alive & (np.asarray(final.firms.ai_experience) > ADOPT_EXP)
    ever_adopted = p_alive & (np.asarray(final.firms.ai_experience) > ADOPT_EXP)
    gain_adopters = (float(np.mean(p_eff[adopters] - a_eff[adopters])
                           if np.any(adopters) else np.nan))
    gain_non = (float(np.mean(p_eff[~ever_adopted & p_alive] - a_eff[~ever_adopted & p_alive])
                      if np.any(~ever_adopted & p_alive) else np.nan))

    return {
        'seed': seed_idx,
        'delta_tfp_raw': (post_raw - pre_raw) / pre_raw,
        'delta_tfp_ai': (post_ai - pre_ai) / pre_ai,
        'n_alive_post': int(np.sum(p_alive)),
        # adoption among ALIVE firms (np.mean over the full mask would include
        # dead firms in the denominator — the same bug fixed in the archived
        # industry-calibration script)
        'adoption_rate': float(np.sum(ever_adopted) / max(float(np.sum(p_alive)), 1)),
        'first_adopt_tick': first_adopt_tick,
        'shift_share': shift_share(pre.firms, final.firms, sc),
        'passthrough_median': passthrough(pre.firms, final.firms,
                                          pre.market, final.market, sc),
        'adopter_aeff_gain': gain_adopters,
        'nonadopter_aeff_gain': gain_non,
        'alloc_cov_post': post_cov,
        'traj': traj,
    }


def main():
    t0 = time.time()
    out_path = os.path.join(SAVE_DIR, 'e2_sigma_mechanism.json')
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)

    for sigma in SIGMA_VALUES:
        key = f'sigma={sigma}'
        rows = results.setdefault(key, [])
        done = len(rows)
        if done >= NUM_SEEDS:
            continue
        for seed_idx in range(done, NUM_SEEDS):
            rows.append(run_seed(sigma, seed_idx))
        _atomic(out_path, results)
        print(f'sigma={sigma}: {NUM_SEEDS} seeds (total {time.time()-t0:.0f}s)')

    _atomic(out_path, results)
    print(f'done -> {out_path}')


def _atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


if __name__ == '__main__':
    main()
