"""BTOS sector-panel analysis (paper Sec 4.6, two-wave paragraph).

Builds the consistent-wording sector panel (original AI question, qid=7,
2023-09 through 2025-10, pre-rewording) from the archived BTOS sector file,
then computes the rank correlations reported in the paper:
  - adoption level vs sigma, early and late windows
  - change (pp) vs sigma; growth ratio vs sigma (n.s.)
  - model early-window adoption gains (E2 trajectories, y1->3) vs sigma
  - model-implied gains vs observed sector changes

Inputs (repository):
  data/raw/download/ai-btos-main/data/clean/Sector_20251208_clean.dta
  abm/output/revision/e2/e2_sigma_mechanism.json
Outputs:
  abm/data/external/btos_sector_panel_two_wave.csv
  abm/data/external/btos_panel_stats.json

The survey's AI question was reworded in late 2025 (biweek 24 of 2025); the
pre-change series is used throughout, so levels before/after the rewording
are never compared.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_abm_dir, '..'))
for _p in (_project_root, _abm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DTA = os.path.join(_project_root, 'data', 'raw', 'download', 'ai-btos-main',
                   'data', 'clean', 'Sector_20251208_clean.dta')
E2 = os.path.join(_abm_dir, 'output', 'revision', 'e2',
                  'e2_sigma_mechanism.json')
OUT_CSV = os.path.join(_abm_dir, 'data', 'external',
                       'btos_sector_panel_two_wave.csv')
OUT_JSON = os.path.join(_abm_dir, 'data', 'external', 'btos_panel_stats.json')

# sector -> DEU-based markup (identical to exp_industry_calibration.py)
MARKUP = {'Information': 1.55,
          'Professional, Scientific, and Technical Services': 1.50,
          'Finance and Insurance': 1.60, 'Educational Services': 1.20,
          'Multi-unit companies operating in multiple sectors': 1.35,
          'Real Estate and Rental and Leasing': 1.35,
          'Health Care and Social Assistance': 1.40,
          'Arts, Entertainment, and Recreation': 1.15,
          'Wholesale Trade': 1.18, 'Retail Trade': 1.15,
          'Manufacturing': 1.25,
          'Other Services (except Public Administration)': 1.20,
          'Construction': 1.10, 'Accommodation and Food Services': 1.10,
          'Transportation and Warehousing': 1.12}

EARLY_END = '2024-01-15'
LATE_START = '2025-09-01'
# Audit 2026-10-10: qid=7 goes missing for four biweeks after 2025-10-01 and
# returns on 2025-11-26 with a uniform +2.9..+16.1pp jump across all mapped
# sectors (largest week-over-week move in the series) — the first fielding of
# the reworded question. Truncate the pre-change series there.
PANEL_END = '2025-11-01'


def main():
    df = pd.read_stata(DTA)
    ai = df[df.questionid == '7'].copy()
    ai['wdate'] = pd.to_datetime(ai['wdate'])
    panel = ai.pivot_table(index='wdate', columns='sector', values='d1')
    panel = panel[(panel.index >= '2023-09-01') & (panel.index < PANEL_END)]

    early = panel[panel.index <= EARLY_END].mean()
    late = panel[panel.index >= LATE_START].mean()

    rows = []
    for sec in panel.columns:
        if sec not in MARKUP:
            continue
        sig = MARKUP[sec] / (MARKUP[sec] - 1)
        rows.append({'sector': sec, 'sigma': sig,
                     'early': float(early[sec]), 'late': float(late[sec]),
                     'delta_pp': float(late[sec] - early[sec]),
                     'growth_ratio': float(late[sec] / early[sec])})
    T = pd.DataFrame(rows).sort_values('sigma').reset_index(drop=True)

    # model early-window adoption gains (E2 yearly trajectories)
    e2 = json.load(open(E2))
    md = {}
    for k in e2:
        s = float(k.split('=')[1])
        tr = np.array([r['traj']['adoption'] for r in e2[k]])
        md[s] = tr.mean(axis=0) * 100
    T['model_d13_pp'] = [
        np.interp(t.sigma, sorted(md), [md[s][2] - md[s][0] for s in sorted(md)])
        for t in T.itertuples()]
    # statistics on unrounded values; rounding is display-only
    T.round({'sigma': 2, 'early': 2, 'late': 2, 'delta_pp': 2,
             'growth_ratio': 2, 'model_d13_pp': 1}).to_csv(OUT_CSV, index=False)

    def sp(a, b):
        r = stats.spearmanr(a, b)
        return float(r[0]), float(r[1])

    # per-biweek pointwise rank correlation of adoption level with sigma
    # (pairwise-complete over mapped sectors per biweek; supports the paper's
    # "concentrated sectors lead at every biweekly reading")
    mapped = [s for s in panel.columns if s in MARKUP]
    sig_vec = np.array([MARKUP[s] / (MARKUP[s] - 1) for s in mapped])
    pt_rho = []
    for w, row in panel[mapped].iterrows():
        obs = row.notna().values
        if obs.sum() >= 3:
            pt_rho.append(sp(sig_vec[obs], row.values[obs])[0])
    pt_rho = np.array(pt_rho)

    out = {
        'source': 'Sector_20251208_clean.dta qid=7 (original AI question, '
                  'pre-rewording)',
        'window_early': f'2023-09-17..{EARLY_END}',
        'window_late': f'{LATE_START}..2025-10-01',
        'n_biweeks': int(panel.shape[0]), 'n_sectors': int(T.shape[0]),
        'n_biweeks_pointwise': int(len(pt_rho)),
        'pointwise_rho_adoption_sigma_all_negative': bool((pt_rho < 0).all()),
        'pointwise_rho_adoption_sigma_min': float(pt_rho.min()),
        'pointwise_rho_adoption_sigma_max': float(pt_rho.max()),
        'spearman_sigma_early': sp(T.sigma, T.early)[0],
        'spearman_sigma_late': sp(T.sigma, T.late)[0],
        'spearman_sigma_delta_pp': sp(T.sigma, T.delta_pp)[0],
        'spearman_sigma_delta_pp_p': sp(T.sigma, T.delta_pp)[1],
        'spearman_sigma_growth_ratio': sp(T.sigma, T.growth_ratio)[0],
        'spearman_sigma_growth_ratio_p': sp(T.sigma, T.growth_ratio)[1],
        'model_d13_vs_sigma': sp(T.sigma, T.model_d13_pp)[0],
        'model_d13_vs_sigma_p': sp(T.sigma, T.model_d13_pp)[1],
        'model_vs_data_delta': sp(T.delta_pp, T.model_d13_pp)[0],
        'model_vs_data_delta_p': sp(T.delta_pp, T.model_d13_pp)[1],
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'-> {OUT_CSV}\n-> {OUT_JSON}')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
