"""Adopter composition at the decade horizon (committed generator for
e10/adopter_composition.json — audit 2026-10-10 source-gap fix).

Per (sigma, baseline panels, h=120):
  adopt_share    adopters (ai_experience > 0.1, alive at post) / alive firms
  w_pre_share    adopters' share of PRE-AI inputs (input weights)
  invest_per_adopt  mean per-period ai_investment among post adopters
  adshare_pre/post  adopters' expenditure share (market_share sum), pre -> post
                   (source of the paper's "surviving adopters' expenditure
                    share rises" sentence, Sec 4.9)

Usage:
    python abm/experiments/post_e10_adopters.py                # baseline cell
    python abm/experiments/post_e10_adopters.py --cells all

The baseline cell writes the canonical adopter_composition.json (the
archived artifact behind the manuscript numbers); every other cell writes
its own adopter_composition_<cell>.json, so no run can clobber the
baseline file.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import post_e10_common as C

ADOPT_EXP = 0.1


def run(cell='baseline', h=120):
    out = {}
    for sigma in C.CELLS[cell]:
        rows = []
        for _, p in C.iter_seeds(cell, sigma):
            k = int(np.where(p['snap_ticks'] - p['snap_ticks'][0] == h)[0][0])
            sp, sq = C.snap(p, 0), C.snap(p, k)
            dp, dq = C.derive(sp, p['params']), C.derive(sq, p['params'])
            ad = dq['alive'] & (sq['ai_experience'] > ADOPT_EXP)
            rows.append({
                'adopt_share': float(ad.sum() / max(dq['alive'].sum(), 1)),
                'w_pre_share': float(dp['inputs'][ad & dp['mask']].sum()
                                     / max(dp['inputs'][dp['mask']].sum(), 1e-9)),
                'invest_per_adopt': float(sq['ai_investment'][ad].mean())
                                    if ad.any() else 0.0,
                'adshare_pre': float(sp['market_share'][ad & dp['mask']].sum()),
                'adshare_post': float(sq['market_share'][ad & dq['mask']].sum()),
                'n_adopt': float(ad.sum()),
            })
        m = {kk: float(np.mean([r[kk] for r in rows])) for kk in rows[0]}
        m['n_adopt'] = float(np.mean([r['n_adopt'] for r in rows]))
        out[f'sigma={sigma}'] = m
    first, last = C.CELLS[cell][0], C.CELLS[cell][-1]
    out['ratio_invest_last_vs_first'] = (
        out[f'sigma={last}']['invest_per_adopt']
        / out[f'sigma={first}']['invest_per_adopt'])
    name = ('adopter_composition.json' if cell == 'baseline'
            else f'adopter_composition_{cell}.json')
    path = C.out_path(name)
    C.atomic_json(path, out)
    print(f'-> {path}')
    print(f"sigma={last}: adopt={out[f'sigma={last}']['adopt_share']:.3f} "
          f"w_pre={out[f'sigma={last}']['w_pre_share']:.3f} "
          f"invest_ratio={out['ratio_invest_last_vs_first']:.1f}")
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='baseline',
                    help="comma-separated cells, or 'all' for every cell under "
                         "the panels directory. Default 'baseline' writes the "
                         "canonical adopter_composition.json; other cells write "
                         "adopter_composition_<cell>.json")
    a = ap.parse_args()
    if a.cells == 'all':
        import os as _os
        _pd = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            '..', 'output', 'revision', 'e10', 'panels')
        a.cells = ','.join(sorted(d for d in _os.listdir(_pd)
                                  if _os.path.isdir(_os.path.join(_pd, d))))
    for cell in a.cells.split(','):
        run(cell)
