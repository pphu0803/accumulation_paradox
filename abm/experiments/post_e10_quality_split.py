"""D2: quality-term split — is the inclusive-measure sigma-gradient carried by
reallocation/composition (market-structure margin) or within-firm quality
accumulation?

Exact decomposition of the inclusive measure (y = a_eff * q^theta):
  within_A + within_q + reallocation + interaction + exit + entry == dTFP_raw
with the within term split by telescoping. Also replicates the A-efficiency
5-term decomposition (E2 comparison) and reports n_reborn everywhere.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import post_e10_common as C

INC_KEYS = ['within_A', 'within_q', 'reallocation', 'interaction', 'exit',
            'entry']
AI_KEYS = ['within', 'reallocation', 'interaction', 'exit', 'entry']


def run(cells=('baseline',), horizons=C.HORIZONS):
    out = {}
    resid_max = 0.0
    for cell in cells:
        for sigma in C.CELLS[cell]:
            per_h = {str(h): {'inc': [], 'ai': []} for h in horizons}
            for _, p in C.iter_seeds(cell, sigma):
                for h in horizons:
                    k = int(np.where(p['snap_ticks'] - p['snap_ticks'][0] == h)[0][0])
                    t_inc = C.decompose(p, k, measure='y')
                    t_ai = C.decompose(p, k, measure='a')
                    resid_max = max(resid_max, abs(t_inc['resid']),
                                    abs(t_ai['resid']))
                    per_h[str(h)]['inc'].append(t_inc)
                    per_h[str(h)]['ai'].append(t_ai)
            out.setdefault(cell, {})[f'sigma={sigma}'] = per_h

    summary = {}
    for cell, sigmas in out.items():
        cell_sigs = C.CELLS[cell]
        cs = {}
        for h in horizons:
            tab = {}
            for s in cell_sigs:
                if f'sigma={s}' not in sigmas:
                    continue
                rows = sigmas[f'sigma={s}'][str(h)]
                for key in INC_KEYS:
                    m, ci = C.mean_ci([r[key] for r in rows['inc']])
                    tab.setdefault(f'sigma={s}', {})[f'inc.{key}'] = \
                        [float(m), float(ci)]
                for key in AI_KEYS:
                    m, ci = C.mean_ci([r[key] for r in rows['ai']])
                    tab[f'sigma={s}'][f'ai.{key}'] = [float(m), float(ci)]
                d = [r['tfp_post'] - r['tfp_pre'] for r in rows['inc']]
                m, ci = C.mean_ci(d)
                tab[f'sigma={s}']['inc.dTFP'] = [float(m), float(ci)]
            cs[str(h)] = tab

        # who carries the sigma-gradient? (h=120, means over seeds)
        have = [s for s in cell_sigs
                if sigmas.get(f'sigma={s}', {}).get('120', {}).get('inc')]
        grad = {}
        for key in INC_KEYS:
            grad[f'inc.{key}'] = C.ols_slope(
                have, [float(np.mean([r[key] for r in
                                      sigmas[f'sigma={s}']['120']['inc']]))
                       for s in have])
        grad['inc.dTFP'] = C.ols_slope(
            have, [float(np.mean([r['tfp_post'] - r['tfp_pre'] for r in
                                  sigmas[f'sigma={s}']['120']['inc']]))
                   for s in have])
        realloc_side = (grad['inc.reallocation'] + grad['inc.interaction']
                        + grad['inc.entry'])
        within_side = grad['inc.within_A'] + grad['inc.within_q']
        grad['realloc_entry_interact'] = float(realloc_side)
        grad['within_A_plus_q'] = float(within_side)
        grad['realloc_share_of_gradient'] = (
            float(realloc_side / grad['inc.dTFP'])
            if len(have) > 2 and grad['inc.dTFP'] != 0 else None)
        n_reborn = [r['n_reborn'] for sk in sigmas.values()
                    for r in sk['120']['inc']]
        grad['n_reborn_mean'] = float(np.mean(n_reborn))
        grad['n_reborn_max'] = int(np.max(n_reborn))
        cs['gradient_carriers'] = grad
        summary[cell] = cs

    path = C.out_path('post_e10_quality_split.json')
    C.atomic_json(path, {'summary': summary, 'residual_max': float(resid_max),
                         'per_seed': out})
    print(f'-> {path} (residual_max={resid_max:.2e})')
    return {'summary': summary, 'per_seed': out, 'residual_max': resid_max}


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
    gc = res['summary'].get('baseline', {}).get('gradient_carriers')
    if gc:
        print(json.dumps(gc, indent=1))
