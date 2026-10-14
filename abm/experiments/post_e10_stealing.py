"""D4: business stealing vs market expansion — is the aggregate gain created
surplus or redistributed position?

Per (sigma, seed, horizon):
  expansion   = d ln(market_size_effective)          (scalar, demand layer)
  dilution    = d ln(sum_S s_i)                      survivors' combined share
  identity    d ln(sum_S rev) = expansion + dilution (asserted)
  redistribution: var/cov stats of survivor share changes, cov(dln s, dln y)
  passthrough_survivors: median dlnP/dlnMC over AGE-RULE survivors (recomputes
      E2's passthrough with uncontaminated survivor labels; E2's endpoint
      labels included reborn slots).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import post_e10_common as C


def stats(p, k):
    params = p['params']
    sp, sq = C.snap(p, 0), C.snap(p, k)
    dp, dq = C.derive(sp, params), C.derive(sq, params)
    r = C.roster(sp, sq, int(p['snap_ticks'][k] - p['snap_ticks'][0]))
    S = r['S'] & dp['mask'] & dq['mask']

    M_pre, M_post = sp['market_size'], sq['market_size']
    expansion = float(np.log(max(M_post, 1e-9) / max(M_pre, 1e-9)))

    s_pre_raw = np.where(S, sp['market_share'], 0.0)
    s_post_raw = np.where(S, sq['market_share'], 0.0)
    dilution = float(np.log(max(s_post_raw.sum(), 1e-12)
                            / max(s_pre_raw.sum(), 1e-12)))
    # identity check: sum_S rev reconstructed both ways
    rev_pre = float((s_pre_raw * M_pre).sum())
    rev_post = float((s_post_raw * M_post).sum())
    ident = abs(np.log(max(rev_post, 1e-12) / max(rev_pre, 1e-12))
                - (expansion + dilution))

    # redistribution among survivors (renormalized shares)
    st_pre = s_pre_raw / max(s_pre_raw.sum(), 1e-12)
    st_post = s_post_raw / max(s_post_raw.sum(), 1e-12)
    dln_s = np.log(np.maximum(st_post, 1e-12) / np.maximum(st_pre, 1e-12))
    dln_y = np.log(np.maximum(dq['y'], 1e-12) / np.maximum(dp['y'], 1e-12))
    cov_sy = float(np.cov(dln_s[S], dln_y[S])[0, 1]) if S.sum() > 2 else None

    # passthrough over true survivors (P = effective_markup * MC_hat,
    # MC_hat = (wage L + dep K)/Q_hat * cost_mult)
    def mc(s, d):
        Qh = np.maximum(d['y'] * d['inputs'], 1e-6)
        return ((s['wage'] * np.maximum(s['L'], 0.01)
                 + params['dep'] * np.maximum(s['K'], 0.01)) / Qh * d['cost_mult'])

    mc_pre, mc_post = mc(sp, dp), mc(sq, dq)
    P_pre = sp['effective_markup'] * mc_pre
    P_post = sq['effective_markup'] * mc_post
    dlogp = np.log(np.maximum(P_post, 1e-9) / np.maximum(P_pre, 1e-9))
    dlogmc = np.log(np.maximum(mc_post, 1e-9) / np.maximum(mc_pre, 1e-9))
    ok = S & np.isfinite(dlogp) & np.isfinite(dlogmc) & (np.abs(dlogmc) > 1e-9)
    passthrough = float(np.median(dlogp[ok] / dlogmc[ok])) if ok.any() else None

    return {'expansion': expansion, 'dilution': dilution,
            'identity_err': float(ident),
            'var_dln_s': float(np.var(dln_s[S])) if S.sum() > 1 else None,
            'cov_dln_s_dln_y': cov_sy,
            'passthrough_survivors': passthrough, 'n_S': int(S.sum()),
            'n_reborn': r['n_reborn'],
            'n_entrants': int(r['N'].sum()), 'n_exiters': int(r['E'].sum())}


def run(cells=('baseline',), horizons=C.HORIZONS):
    out = {}
    ident_max = 0.0
    for cell in cells:
        for sigma in C.CELLS[cell]:
            per_h = {str(h): [] for h in horizons}
            for _, p in C.iter_seeds(cell, sigma):
                for h in horizons:
                    k = int(np.where(p['snap_ticks'] - p['snap_ticks'][0] == h)[0][0])
                    r = stats(p, k)
                    ident_max = max(ident_max, r['identity_err'])
                    per_h[str(h)].append(r)
            out.setdefault(cell, {})[f'sigma={sigma}'] = per_h
    summary = {}
    for cell, sigmas in out.items():
        cs = {}
        for h in C.HORIZONS:
            tab = {}
            for s in C.CELLS[cell]:
                if f'sigma={s}' not in sigmas:
                    continue
                rows = sigmas[f'sigma={s}'][str(h)]
                for key in ['expansion', 'dilution', 'var_dln_s',
                            'cov_dln_s_dln_y', 'passthrough_survivors']:
                    # audit 2026-10-10: skip None (a sigma=10 seed with n_S=2
                    # trips the cov guard; None otherwise poisons the mean)
                    vals = [r[key] for r in rows if r[key] is not None]
                    m, ci = C.mean_ci(vals)
                    tab.setdefault(f'sigma={s}', {})[key] = [float(m), float(ci)]
            cs[str(h)] = tab
        summary[cell] = cs
    path = C.out_path('post_e10_stealing.json')
    C.atomic_json(path, {'summary': summary, 'identity_err_max': float(ident_max),
                         'per_seed': out})
    print(f'-> {path} (identity_err_max={ident_max:.2e})')
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
    run(tuple(a.cells.split(',')))
