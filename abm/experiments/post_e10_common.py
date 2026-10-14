"""Shared helpers for the E10 post-processing chain (D1-D4 diagnostics).

Pure numpy (no jax import) — safe to run while the E10 batch is alive.

All identities are computed in float64 (numpy 2.4 NEP-50 keeps float32 under
python-scalar promotion; panels are stored float32, so cast at load).
"""
import json
import os
import types

import numpy as np

_EXPERIMENTS = os.path.abspath(os.path.join(os.path.dirname(__file__)))
_PANEL_DIR = os.path.join(_EXPERIMENTS, '..', 'output', 'revision', 'e10',
                         'panels')

FIRM_FIELDS = ['K', 'L', 'A_base', 'sigma_i', 'product_quality',
               'ai_experience', 'ai_investment', 'ai_strategy',
               'market_share', 'effective_markup', 'cash', 'alive', 'age']

BASELINE_SIGMAS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
GSENS_SIGMAS = [2.0, 5.0, 8.0]
CELLS = {'baseline': BASELINE_SIGMAS,
         'gsens_g0.02': GSENS_SIGMAS, 'gsens_g0.08': GSENS_SIGMAS}

HORIZONS = [12, 36, 60, 120]   # post ticks (model years 1, 3, 5, 10)


def load_panel(cell, sigma, seed_idx):
    """Load one panel npz -> dict with float64 firm arrays + parsed config."""
    path = os.path.join(_PANEL_DIR, cell, f'sigma={sigma}',
                        f'seed{seed_idx:04d}.npz')
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        p = {k: z[k] for k in z.files}
    for f in FIRM_FIELDS:
        p[f] = p[f].astype(np.float64)
    p['alive'] = p['alive'].astype(bool)
    p['age'] = p['age'].astype(np.int64)
    for f in ['market_size_effective', 'total_Q', 'agg_price', 'wage_rate',
              'hhi']:
        p['m_' + f] = p['m_' + f].astype(np.float64)
    cfg = json.loads(str(p['config_json']))
    p['params'] = {
        'alpha': float(cfg['alpha_CD']), 'theta': float(cfg['quality_weight']),
        'gain': float(cfg['ai_efficiency_gain']),
        'depth': float(cfg['j_curve_depth']),
        'dep': float(cfg['depreciation_rate']),
        'cost_exp_coeff': float(cfg['ai_cost_experience_coeff']),
        'cost_elast': float(cfg['ai_cost_elasticity']),
        'sigma_base': float(cfg['sigma_base']),
    }
    return p


def snap(p, k):
    """Snapshot k (0 = pre) as a dict of firm arrays."""
    return {f: p[f][k] for f in FIRM_FIELDS} | {
        'wage': p['m_wage_rate'][k], 'market_size': p['m_market_size_effective'][k],
        'snap_tick': int(p['snap_ticks'][k])}


def derive(s, params):
    """Per-firm derived quantities at one snapshot (float64)."""
    alive = s['alive']
    K = np.maximum(s['K'], 0.01)
    L = np.maximum(s['L'], 0.01)
    inputs = K ** params['alpha'] * L ** (1 - params['alpha'])
    x = s['ai_experience']
    mult = np.where(x > 0, 1 + params['gain'] * ((1 + params['depth']) * x
                                                 - params['depth']), 1.0)
    a_eff = s['A_base'] * mult
    qt = np.maximum(s['product_quality'], 0.1) ** params['theta']
    mask = alive & (inputs > 0.01)
    w = np.zeros_like(inputs)
    tot = inputs[mask].sum()
    if tot > 0:
        w[mask] = inputs[mask] / tot
    strat = s['ai_strategy']
    cm = (1 - params['cost_exp_coeff'] * x
          - s['ai_investment'] * (1 - strat) * params['cost_elast'])
    cm = np.where(x > 0, np.maximum(cm, 0.5), 1.0)
    return {'inputs': inputs, 'a_eff': a_eff, 'qt': qt, 'y': a_eff * qt,
            'mask': mask, 'w': w, 'cost_mult': cm, 'alive': alive}


def roster(snap_pre, snap_post, elapsed):
    """Age-rule roster: survivors / exiters / entrants / reborn slots.

    survivor = alive_pre & alive_post & (age_post - age_pre == elapsed)
    reborn slots count their PRE firm in exiters and their POST firm in
    entrants (slot recycled, step.py:606-653).
    """
    pre, post = snap_pre['alive'], snap_post['alive']
    d = snap_post['age'] - snap_pre['age']
    both = pre & post
    surv = both & (d == elapsed)
    reborn = both & (d != elapsed)
    exiters = (pre & ~post) | reborn          # pre-firms gone (incl. reborn slots)
    entrants = (~pre & post) | reborn         # post-firms new (incl. reborn slots)
    return {'S': surv, 'E': exiters, 'N': entrants, 'R': reborn,
            'n_reborn': int(reborn.sum())}


def ss_general(w_pre, w_post, y_pre, y_post, S):
    """5-term shift-share over a survivor set S (any measure y).

    within + realloc + interact - exit + entry, with exit/entry carried by
    callers. Here: within/realloc/interact over S; returns also the S-side
    telescoping sums needed for closure. Identity:
      TFP_post - TFP_pre = within + realloc + interact
          + sum_{N} w_post y_post - sum_{E} w_pre y_pre
    """
    dw = np.where(S, w_post - w_pre, 0.0)
    dy = np.where(S, y_post - y_pre, 0.0)
    within = float(np.sum(np.where(S, w_pre, 0.0) * dy))
    realloc = float(np.sum(dw * np.where(S, y_pre, 0.0)))
    interact = float(np.sum(dw * dy))
    return {'within': within, 'reallocation': realloc, 'interaction': interact}


def decompose(p, k, measure='y'):
    """Full 5-term decomposition pre -> snapshot k for measure in
    {'y' (inclusive), 'a' (AI-efficiency), 'q' (quality factor)}.

    For 'y' additionally splits `within` into within_A and within_q via the
    exact telescoping: d(a q) = da * q_pre + a_post * dq.
    Returns dict of terms + resid (must be ~1e-15 in float64).
    """
    params = p['params']
    sp, sq = snap(p, 0), snap(p, k)
    dp, dq = derive(sp, params), derive(sq, params)
    r = roster(sp, sq, int(p['snap_ticks'][k] - p['snap_ticks'][0]))
    S, E, N = r['S'], r['E'], r['N']

    def masked_y(d, m):
        return np.where(m, d['y'] if measure == 'y'
                        else d['a_eff'] if measure == 'a'
                        else d['qt'], 0.0)

    y_pre, y_post = masked_y(dp, dp['mask']), masked_y(dq, dq['mask'])
    w_pre, w_post = dp['w'], dq['w']
    terms = ss_general(w_pre, w_post, y_pre, y_post, S & dp['mask']
                       & dq['mask'])
    terms['exit'] = float(-np.sum(w_pre * y_pre * (E & dp['mask'])))
    terms['entry'] = float(np.sum(w_post * y_post * (N & dq['mask'])))
    tfp_pre = float(np.sum(w_pre * y_pre))
    tfp_post = float(np.sum(w_post * y_post))
    terms['tfp_pre'], terms['tfp_post'] = tfp_pre, tfp_post
    resid = (terms['within'] + terms['reallocation'] + terms['interaction']
             + terms['exit'] + terms['entry']) - (tfp_post - tfp_pre)
    terms['resid'] = float(resid)
    terms['n_reborn'] = r['n_reborn']

    if measure == 'y':
        # exact within split (telescoping; sum is convention-invariant)
        Sm = S & dp['mask'] & dq['mask']
        da = np.where(Sm, dq['a_eff'] - dp['a_eff'], 0.0)
        dqt = np.where(Sm, dq['qt'] - dp['qt'], 0.0)
        terms['within_A'] = float(np.sum(np.where(Sm, dp['w'], 0.0)
                                         * da * np.where(Sm, dp['qt'], 0.0)))
        terms['within_q'] = float(np.sum(np.where(Sm, dp['w'], 0.0)
                                         * np.where(Sm, dq['a_eff'], 0.0) * dqt))
    return terms


def tfp_of(p, k, theta=None):
    """Aggregate TFP at snapshot k; theta=None uses model theta, 0 strips q."""
    params = dict(p['params'])
    if theta is not None:
        params['theta'] = theta
    d = derive(snap(p, k), params)
    m = d['mask']
    y = d['a_eff'] * (d['qt'] if theta is None or theta > 0 else 1.0)
    return float(np.sum(d['w'][m] * y[m]))


def iter_seeds(cell, sigma):
    """Yield (seed_idx, panel) for every completed npz in a cell/sigma."""
    d = os.path.join(_PANEL_DIR, cell, f'sigma={sigma}')
    if not os.path.isdir(d):
        return
    for fn in sorted(os.listdir(d)):
        if fn.startswith('seed') and fn.endswith('.npz'):
            idx = int(fn[4:8])
            p = load_panel(cell, sigma, idx)
            if p is not None:
                yield idx, p


def mean_ci(values, axis=0):
    v = np.asarray(values, float)
    m = v.mean(axis=axis)
    se = v.std(axis=axis, ddof=1) / np.sqrt(v.shape[axis]) \
        if v.shape[axis] > 1 else np.zeros_like(m)
    return m, 1.96 * se


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = np.polyfit(x, y, 1)[0]
    return float(b)


def out_path(name):
    d = os.path.join(_EXPERIMENTS, '..', 'output', 'revision', 'e10')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
