"""Rebuild the sector sigma assignment from the packaged De Loecker et al.
(2020) sector file and re-evaluate the BTOS rank correlation.

The calibration's headline mapping (manuscript Secs 3.2/4.6) uses a
synthesized sector-markup table (reproduced in the manuscript appendix);
this script quantifies how the external validation changes if the sigma
assignment is instead rebuilt mechanically from the packaged firm-level
DEU sector averages (deu_markups_by_sector_2020.csv).  Split NAICS
sectors (retail 44/45, manufacturing 31-33, transportation 48/49) are
aggregated to a BTOS sector markup either by 2016 sales share (falling
back to the unweighted mean where the file reports zero shares) or by the
unweighted mean; sigma = mu / (mu - 1).  Model adoption at the rebuilt
sigma is read off the archived adoption-sigma curve by monotone-cubic
(PCHIP) interpolation with flat extrapolation outside the calibrated
range.  Pure pandas/scipy on packaged inputs; no simulation.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import PchipInterpolator

EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(EXPERIMENTS, '..', 'output', 'industry_calibration')
CAL = os.path.join(OUTDIR, 'industry_calibration_results.json')
DEU = os.path.join(EXPERIMENTS, '..', 'data', 'external',
                   'deu_markups_by_sector_2020.csv')

# BTOS sector -> naics2 rows of the DEU file (multi-unit companies: none)
CSV_MAP = {
    'Information': [51], 'Professional Services': [54],
    'Finance and Insurance': [52], 'Educational Services': [61],
    'Real Estate and Rental': [53], 'Health Care': [62],
    'Arts and Recreation': [71], 'Wholesale Trade': [42],
    'Retail Trade': [44, 45], 'Manufacturing': [31, 32, 33],
    'Other Services': [81], 'Construction': [23],
    'Accommodation and Food': [72], 'Transportation': [48, 49],
}


def run():
    cal = json.load(open(CAL))
    sectors = cal['sectors']
    deu = pd.read_csv(DEU)

    # archived adoption-sigma curve (one point per distinct sigma)
    pts = {}
    for v in sectors.values():
        pts.setdefault(round(v['sigma'], 6), v['adoption_rate_mean'])
    xs = np.array(sorted(pts))
    ys = np.array([pts[round(s, 6)] for s in xs])
    curve = PchipInterpolator(xs, ys)

    def model_adopt(sig):
        return float(curve(np.clip(sig, xs[0], xs[-1])))

    rows = {}
    for sec, naics in CSV_MAP.items():
        sub = deu[deu['naics2'].isin(naics)]
        w = sub['sales_share_2016'].values
        if w.sum() > 0:
            mu_sales = float((sub['markup_2016'].values * w).sum() / w.sum())
            rule = 'sales_share'
        else:
            mu_sales = float(sub['markup_2016'].mean())
            rule = 'mean_fallback_zero_share'
        mu_mean = float(sub['markup_2016'].mean())
        sig_sales = mu_sales / (mu_sales - 1)
        sig_mean = mu_mean / (mu_mean - 1)
        rows[sec] = {
            'btos_rate': sectors[sec]['btos_rate'],
            'used_sigma': sectors[sec]['sigma'],
            'used_adoption': sectors[sec]['adoption_rate_mean'],
            'aggregation_sales': rule,
            'markup_sales': mu_sales, 'sigma_sales': sig_sales,
            'adoption_sales': model_adopt(sig_sales),
            'markup_mean': mu_mean, 'sigma_mean': sig_mean,
            'adoption_mean': model_adopt(sig_mean),
        }
    df = pd.DataFrame(rows).T
    base = stats.spearmanr(df.used_adoption, df.btos_rate)
    r_sales = stats.spearmanr(df.adoption_sales, df.btos_rate)
    r_mean = stats.spearmanr(df.adoption_mean, df.btos_rate)
    out = {
        'source': 'deu_markups_by_sector_2020.csv (packaged) + archived '
                  'adoption-sigma curve (PCHIP interpolation, flat '
                  'extrapolation outside the calibrated sigma range)',
        'n_sectors': len(df),
        'baseline_spearman': float(base.statistic),
        'spearman_sales_share_aggregation': float(r_sales.statistic),
        'spearman_sales_share_aggregation_p': float(r_sales.pvalue),
        'spearman_row_mean_aggregation': float(r_mean.statistic),
        'spearman_row_mean_aggregation_p': float(r_mean.pvalue),
        'sectors': rows,
    }
    path = os.path.join(OUTDIR, 'deu_sigma_robustness.json')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, path)
    print(f'-> {path}')
    print(f"baseline (synthesized sigmas, same {len(df)} sectors): rho={out['baseline_spearman']:.3f}")
    print(f"DEU sales-share aggregation: rho={out['spearman_sales_share_aggregation']:.3f} "
          f"(p={out['spearman_sales_share_aggregation_p']:.4f})")
    print(f"DEU row-mean aggregation:    rho={out['spearman_row_mean_aggregation']:.3f} "
          f"(p={out['spearman_row_mean_aggregation_p']:.4f})")
    return out


if __name__ == '__main__':
    run()
