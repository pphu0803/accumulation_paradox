"""Cross-check the sector sigma mapping against AI occupational-exposure
scores (Felten, Raj & Seamans 2021, AIIE by 4-digit NAICS).

The manuscript (Secs 3.2/4.6) states that the markup-based sector ordering
is cross-checked against AIIE industry exposure.  This script is that
check: it aggregates Data Appendix B of the AIOE data appendix (industry-
level AIIE, 4-digit NAICS) to the fourteen BTOS calibration sectors with a
NAICS mapping (multi-unit companies has none), and correlates the model's
archived adoption prediction and the assigned sigma with the sector mean
exposure score.

The raw Felten appendix is NOT redistributed with the package; set
AIOE_XLSX to its path (default: <repo>/data/raw/exposure/
AIOE_DataAppendix.xlsx, downloadable from the authors' public repository).
The output JSON records the fourteen sector aggregates, so the rank
comparison is auditable without the raw file.  Pure pandas/scipy; no
simulation.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(EXPERIMENTS))
CAL = os.path.join(EXPERIMENTS, '..', 'output', 'industry_calibration',
                   'industry_calibration_results.json')

XLSX = os.environ.get(
    'AIOE_XLSX',
    os.path.join(ROOT, 'data', 'raw', 'exposure', 'AIOE_DataAppendix.xlsx'))
if not os.path.isfile(XLSX):
    raise SystemExit(
        f'AIOE input not found: {XLSX}\n'
        'The Felten, Raj & Seamans (2021) AIOE Data Appendix is not '
        'redistributed with this package.\nSet the AIOE_XLSX environment '
        'variable to your copy (see the module docstring and README). The '
        '14 sector-level aggregates this script computes are archived in '
        'felten_crosscheck.json, so the check can be audited without the '
        'raw file.')

# BTOS calibration sector -> 2-digit NAICS prefixes of its industries
# (multi-unit companies is excluded: no NAICS definition).
NAICS_MAP = {
    'Information': [51],
    'Professional Services': [54],
    'Finance and Insurance': [52],
    'Educational Services': [61],
    'Real Estate and Rental': [53],
    'Health Care': [62],
    'Arts and Recreation': [71],
    'Wholesale Trade': [42],
    'Retail Trade': [44, 45],
    'Manufacturing': [31, 32, 33],
    'Other Services': [81],
    'Construction': [23],
    'Accommodation and Food': [72],
    'Transportation': [48, 49],
}


def run():
    cal = json.load(open(CAL))
    sectors = cal['sectors']
    aiie = pd.read_excel(XLSX, sheet_name='Appendix B')
    aiie['naics2'] = aiie['NAICS'].astype(str).str[:2].astype(int)

    rows = {}
    for sec, prefixes in NAICS_MAP.items():
        sub = aiie[aiie['naics2'].isin(prefixes)]
        if len(sub) == 0:
            raise RuntimeError(f'no AIIE industries matched for {sec}')
        rows[sec] = {
            'n_industries': int(len(sub)),
            'aiie_mean': float(sub['AIIE'].mean()),
            'sigma': sectors[sec]['sigma'],
            'btos_rate': sectors[sec]['btos_rate'],
            'model_adoption': sectors[sec]['adoption_rate_mean'],
        }
    df = pd.DataFrame(rows).T
    out = {
        'source': 'Felten, Raj & Seamans (2021), AIOE Data Appendix B '
                  '(AIIE by 4-digit NAICS), aggregated as unweighted means '
                  'of 4-digit industries within each sector; not '
                  'redistributed, obtainable from the authors',
        'n_sectors': len(df),
        'spearman_model_adoption_vs_aiie': float(stats.spearmanr(df.model_adoption, df.aiie_mean).statistic),
        'spearman_model_adoption_vs_aiie_p': float(stats.spearmanr(df.model_adoption, df.aiie_mean).pvalue),
        'spearman_sigma_vs_aiie': float(stats.spearmanr(df.sigma, df.aiie_mean).statistic),
        'spearman_btos_vs_aiie': float(stats.spearmanr(df.btos_rate, df.aiie_mean).statistic),
        'sectors': rows,
    }
    path = os.path.join(EXPERIMENTS, '..', 'output', 'industry_calibration',
                        'felten_crosscheck.json')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, path)
    print(f'-> {path}')
    print(f"model adoption vs AIIE: rho={out['spearman_model_adoption_vs_aiie']:.3f} "
          f"(p={out['spearman_model_adoption_vs_aiie_p']:.4f}, n={out['n_sectors']})")
    print(f"sigma vs AIIE:         rho={out['spearman_sigma_vs_aiie']:.3f}")
    return out


if __name__ == '__main__':
    run()
