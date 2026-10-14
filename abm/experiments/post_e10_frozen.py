"""D1: frozen-counterfactual ladder on realized paths.

L0 realized          -> L1 roster fixed (age-rule survivors, renormalized)
   -> L2 input shares frozen at pre -> L3 quality stripped (theta=0).

Rung differences are each margin's contribution on the restricted set; the
ladder answers how much of the sigma-gradient of measured TFP survives
without turnover, without reallocation, without quality. Pure accounting on
realized paths (no re-simulation, no equilibrium change).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import post_e10_common as C


def ladder(p, k):
    params = p['params']
    sp, sq = C.snap(p, 0), C.snap(p, k)
    dp, dq = C.derive(sp, params), C.derive(sq, params)
    r = C.roster(sp, sq, int(p['snap_ticks'][k] - p['snap_ticks'][0]))
    S = r['S'] & dp['mask'] & dq['mask']

    def tfp_S(d):
        return float(np.sum(d['inputs'][S] * d['y'][S]) / np.sum(d['inputs'][S]))

    L0 = float(np.sum(dq['w'][dq['mask']] * dq['y'][dq['mask']])
               - np.sum(dp['w'][dp['mask']] * dp['y'][dp['mask']]))
    L1 = tfp_S(dq) - tfp_S(dp)
    omega = dp['inputs'][S] / np.sum(dp['inputs'][S])
    L2 = float(np.sum(omega * (dq['y'][S] - dp['y'][S])))
    L3 = float(np.sum(omega * (dq['a_eff'][S] - dp['a_eff'][S])))
    L0a = float(np.sum(dq['w'][dq['mask']] * dq['a_eff'][dq['mask']])
                - np.sum(dp['w'][dp['mask']] * dp['a_eff'][dp['mask']]))

    # closure: 3-term shift-share over S with renormalized weights == dL1
    wS_pre = dp['inputs'][S] / np.sum(dp['inputs'][S])
    wS_post = dq['inputs'][S] / np.sum(dq['inputs'][S])
    ypre, ypost = dp['y'][S], dq['y'][S]
    dw, dy = wS_post - wS_pre, ypost - ypre
    resid = (np.sum(wS_pre * dy) + np.sum(dw * ypre) + np.sum(dw * dy)) - L1
    checks = {
        'omega_sum': float(abs(omega.sum() - 1.0)),
        'frozen_pre_reproduces': float(abs(np.sum(omega * ypre) - tfp_S(dp))),
        'three_term_resid': float(abs(resid)),
    }
    return {'L0': L0, 'L1': L1, 'L2': L2, 'L3': L3, 'L0a': L0a,
            'n_S': int(S.sum()), 'n_reborn': r['n_reborn'], 'checks': checks}


def run(cells=('baseline',), horizons=C.HORIZONS):
    out = {}
    for cell in cells:
        for sigma in C.CELLS[cell]:
            per_h = {str(h): [] for h in horizons}
            for _, p in C.iter_seeds(cell, sigma):
                for h in horizons:
                    k = int(np.where(p['snap_ticks'] - p['snap_ticks'][0] == h)[0][0])
                    per_h[str(h)].append(ladder(p, k))
            out.setdefault(cell, {})[f'sigma={sigma}'] = per_h

    # summarize: means + CIs, gradients, surviving fractions
    summary = {}
    for cell, sigmas in out.items():
        cs = {}
        for h in C.HORIZONS:
            rows = {}
            for lv in ['L0', 'L1', 'L2', 'L3', 'L0a']:
                means = [np.mean([r[lv] for r in sigmas[f'sigma={s}'][str(h)]])
                         for s in C.CELLS[cell]]
                rows[lv] = [float(m) for m in means]
            x = C.CELLS[cell]
            grad = {lv: C.ols_slope(x, v) for lv, v in rows.items()}
            cs[str(h)] = {'levels': rows, 'gradients': grad,
                          'surviving_fraction': {
                              lv: float(grad[lv] / grad['L0'])
                              for lv in ['L1', 'L2', 'L3'] if grad['L0'] != 0}}
        summary[cell] = cs

    resid_max = 0.0
    for cell, sigmas in out.items():
        for sk, per_h in sigmas.items():
            for h, rows in per_h.items():
                for r in rows:
                    resid_max = max(resid_max, max(r['checks'].values()))
    result = {'summary': summary, 'residual_max': float(resid_max),
              'per_seed': out}
    path = C.out_path('post_e10_frozen.json')
    C.atomic_json(path, result)
    print(f'-> {path} (residual_max={resid_max:.2e})')
    return result


def figure(result, cell='baseline', h='120'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = result['summary'][cell][h]['levels']
    x = C.CELLS[cell]
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.2
    for j, lv in enumerate(['L0', 'L1', 'L2', 'L3']):
        ax.bar(np.arange(len(x)) + (j - 1.5) * w, s[lv], w, label=lv)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$\Delta$TFP (inclusive)')
    ax.legend(title='ladder')
    fig.tight_layout()
    fig.savefig(C.out_path('e10_fig_frozen_ladder.png'), dpi=150)
    print('figure -> e10_fig_frozen_ladder.png')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='all',
                    help="comma-separated cells, or 'all' to process every cell "
                         "present under the panels directory (the default; a bare "
                         "'baseline' run would write a partial file over the archive)")
    ap.add_argument('--no-fig', action='store_true')
    a = ap.parse_args()
    if a.cells == 'all':
        import os as _os
        _pd = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            '..', 'output', 'revision', 'e10', 'panels')
        a.cells = ','.join(sorted(d for d in _os.listdir(_pd)
                                  if _os.path.isdir(_os.path.join(_pd, d))))
    res = run(tuple(a.cells.split(',')))
    if not a.no_fig:
        figure(res)
