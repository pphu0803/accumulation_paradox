"""D3: index robustness — does the sigma-gradient of "competitive markets
transmit more" survive alternative aggregation rules?

Indices (growth over pre -> h, survivor-set Tornquist where noted):
  (i)   AI-efficiency-only   dln TFP_ai                     (= E2 headline)
  (ii)  simple-sum inclusive dln (sum Q_hat / sum inputs)    (= paper headline)
  (iii) expenditure-share-weighted real output net of input growth (Tornquist)
  (iv)  hedonic-deflated: quality treated as price decline
        P_hed = P / q^{theta/(sigma_i-1)}, real quantity = Q_hat * q^{theta/(sigma_i-1)}
  (v)   input-share-weighted Tornquist on the same survivor set as (iii):
        the pure reweighting of (iii) (same roster, same formula, input
        weights instead of expenditure weights), so (iii)/(v) isolates the
        weighting convention.
Deliverable: sign of the gradient (OLS slope over sigma + per-sigma means)
under each index.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import post_e10_common as C


def indices(p, k):
    params = p['params']
    sp, sq = C.snap(p, 0), C.snap(p, k)
    dp, dq = C.derive(sp, params), C.derive(sq, params)
    r = C.roster(sp, sq, int(p['snap_ticks'][k] - p['snap_ticks'][0]))
    S = r['S'] & dp['mask'] & dq['mask']

    def ln_growth(vals_pre, vals_post, w_pre, w_post):
        dl = np.log(np.maximum(vals_post, 1e-12) / np.maximum(vals_pre, 1e-12))
        wbar = 0.5 * (w_pre + w_post)
        return float(np.sum(wbar * dl))

    # (i) AI-only and (ii) simple-sum inclusive
    def tfp(d, y):
        m = d['mask']
        return float(np.sum(d['w'][m] * y[m]))

    i_ai = np.log(tfp(dq, dq['a_eff']) / tfp(dp, dp['a_eff']))
    Qh_pre = dp['y'] * dp['inputs']
    Qh_post = dq['y'] * dq['inputs']
    # Audit 2026-10-10: the post ratio below used PRE inputs, which cancelled
    # against the pre ratio and reduced (ii) to gross-output growth (sigma=2
    # gave -13.8% while inclusive TFP rises ~19%). Post inputs restored.
    i_inc = np.log(Qh_post[dq['mask']].sum() / dq['inputs'][dq['mask']].sum()
                   / (Qh_pre[dp['mask']].sum() / dp['inputs'][dp['mask']].sum()))

    # survivor-set weights for Tornquist indices
    s_pre = np.where(S, sp['market_share'], 0.0)
    s_post = np.where(S, sq['market_share'], 0.0)
    s_pre = s_pre / max(s_pre.sum(), 1e-12)
    s_post = s_post / max(s_post.sum(), 1e-12)
    wS_pre = dp['inputs'][S] / dp['inputs'][S].sum()
    wS_post = dq['inputs'][S] / dq['inputs'][S].sum()
    # expand to full arrays for ln_growth's flat weighting
    wpre_full = np.zeros_like(s_pre)
    wpost_full = np.zeros_like(s_post)
    wpre_full[S], wpost_full[S] = wS_pre, wS_post

    # (iii) expenditure-weighted real output net of input growth
    y_exp = ln_growth(Qh_pre, Qh_post, s_pre, s_post)
    x_exp = ln_growth(dp['inputs'], dq['inputs'], wpre_full, wpost_full)
    iii = y_exp - x_exp

    # (v) input-weighted Tornqvist on the same survivor set: identical to
    # (iii) except the output side is weighted by input shares, not
    # expenditure shares -- the pure weighting-convention comparison.
    y_inw = ln_growth(Qh_pre, Qh_post, wpre_full, wpost_full)
    v_inw = y_inw - x_exp

    # (iv) hedonic-deflated: real quantity Q~ = Q_hat * q^{theta/(sigma_i-1)}
    expo_pre = np.maximum(sp['product_quality'], 0.1) ** (
        params['theta'] / np.maximum(sp['sigma_i'] - 1.0, 0.5))
    expo_post = np.maximum(sq['product_quality'], 0.1) ** (
        params['theta'] / np.maximum(sq['sigma_i'] - 1.0, 0.5))
    Qt_pre = Qh_pre * expo_pre
    Qt_post = Qh_post * expo_post
    iv = ln_growth(Qt_pre, Qt_post, s_pre, s_post) - x_exp

    return {'i_ai_efficiency_only': float(i_ai),
            'ii_simple_sum_inclusive': float(i_inc),
            'iii_expenditure_tornquist': float(iii),
            'iv_hedonic_deflated': float(iv),
            'v_input_weight_tornquist': float(v_inw),
            'n_S': int(S.sum())}


def run(cells=('baseline',), h=120):
    out = {}
    for cell in cells:
        for sigma in C.CELLS[cell]:
            rows = []
            for _, p in C.iter_seeds(cell, sigma):
                k = int(np.where(p['snap_ticks'] - p['snap_ticks'][0] == h)[0][0])
                rows.append(indices(p, k))
            out.setdefault(cell, {})[f'sigma={sigma}'] = rows

    summary = {}
    for cell, sigmas in out.items():
        have = [s for s in C.CELLS[cell] if sigmas.get(f'sigma={s}')]
        keys = ['i_ai_efficiency_only', 'ii_simple_sum_inclusive',
                'iii_expenditure_tornquist', 'iv_hedonic_deflated',
                'v_input_weight_tornquist']
        tab, slopes, signs = {}, {}, {}
        for s in have:
            tab[f'sigma={s}'] = {}
            for key in keys:
                m, ci = C.mean_ci([r[key] for r in sigmas[f'sigma={s}']])
                tab[f'sigma={s}'][key] = [float(m), float(ci)]
        for key in keys:
            means = [float(np.mean([r[key] for r in sigmas[f'sigma={s}']]))
                     for s in have]
            slopes[key] = C.ols_slope(have, means)
            signs[key] = {str(s): float(np.sign(np.mean(
                [r[key] for r in sigmas[f'sigma={s}']]))) for s in have}
        summary[cell] = {'means': tab, 'slopes_vs_sigma': slopes,
                         'sign_table': signs}
    path = C.out_path('post_e10_index.json')
    C.atomic_json(path, {'summary': summary, 'per_seed': out})
    print(f'-> {path}')
    return {'summary': summary}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='all',
                    help="comma-separated cells, or 'all' to process every cell "
                         "present under the panels directory (the default; a bare "
                         "'baseline' run would write a partial file over the archive)")
    a = ap.parse_args()
    if a.cells == 'all':
        import os as _os
        _pd = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            '..', 'output', 'revision', 'e10', 'panels')
        a.cells = ','.join(sorted(d for d in _os.listdir(_pd)
                                  if _os.path.isdir(_os.path.join(_pd, d))))
    res = run(tuple(a.cells.split(',')))
    print(json.dumps(res['summary'].get('baseline', {}).get('slopes_vs_sigma',
                                                            {}), indent=1))
