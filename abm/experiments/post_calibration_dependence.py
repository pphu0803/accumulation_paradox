"""Dependence and vintage robustness of the BTOS calibration statistics.

Four checks quoted in manuscript Secs 4.6/4.9, all pure post-processing on
packaged inputs (archived calibration results, the two-wave sector panel,
the sector adoption CSV, and the McKinsey OOS output):

(a) duplicate-sigma deduplication: the 15 BTOS sectors map to only 11
    distinct model elasticities (four duplicate pairs).  For every one of
    the 16 ways of keeping one sector per pair, recompute (i) the
    cross-section model-vs-BTOS Spearman and (ii) the panel model-vs-delta
    Spearman; report the range.  Also a fixed-seed Monte-Carlo permutation
    p-value for the cross-section at the weakest resolution (n=11).
(b) question-rewording vintage: the cross-section targets the mean of
    the 2025-11-26 and 2025-12-10 waves (the November-December 2025
    fieldings, the first two after the AI question was reworded); the
    panel uses the pre-rewording series.  Recompute the cross-section
    Spearman, the model-data level gap, and the level jump against the
    pre-rewording column (yes2wk_old).
(c) McKinsey boundary case: report the correlation under every admissible
    subset construction -- the strictest (n=6: only the BTOS-mapped sectors
    whose elasticity is unique in the cross-check), the reported 8-sector
    subset (keeps the primary of each shared pair, drops the aliases and the
    real-estate/environment pair), and the 8+real-estate / 8+both variants.
(d) leave-one-out on the Table-6 design itself (drop each of the 15
    sectors in turn, Spearman on the remaining 14), quoted in Secs 4.6
    and 4.9 alongside the grid-design range of exp_rev_e8.
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(EXPERIMENTS, '..', 'output')
CAL = os.path.join(OUTDIR, 'industry_calibration',
                   'industry_calibration_results.json')
OOS = os.path.join(OUTDIR, 'robustness', 'oos_validation.json')
PANEL = os.path.join(EXPERIMENTS, '..', 'data', 'external',
                     'btos_sector_panel_two_wave.csv')
ADOPT = os.path.join(EXPERIMENTS, '..', 'data', 'external',
                     'btos_ai_adoption_by_sector.csv')

NAME = {  # long BTOS names -> calibration sector keys
    'Information': 'Information',
    'Professional, Scientific, and Technical Services': 'Professional Services',
    'Finance and Insurance': 'Finance and Insurance',
    'Educational Services': 'Educational Services',
    'Multi-unit companies operating in multiple sectors': 'Multi-sector Companies',
    'Real Estate and Rental and Leasing': 'Real Estate and Rental',
    'Health Care and Social Assistance': 'Health Care',
    'Arts, Entertainment, and Recreation': 'Arts and Recreation',
    'Wholesale Trade': 'Wholesale Trade',
    'Retail Trade': 'Retail Trade',
    'Manufacturing': 'Manufacturing',
    'Other Services (except Public Administration)': 'Other Services',
    'Construction': 'Construction',
    'Accommodation and Food Services': 'Accommodation and Food',
    'Transportation and Warehousing': 'Transportation',
}

MC_DRAWS = 200_000
MC_SEED = 20261011


def resolutions(keys, dup_groups):
    """Yield the 16 keep-lists (one sector per duplicate pair)."""
    for combo in itertools.product(*[[0, 1]] * len(dup_groups)):
        keep = set(keys)
        for gi, take in enumerate(combo):
            for j, s in enumerate(dup_groups[gi]):
                if j != take:
                    keep.discard(s)
        yield [k for k in keys if k in keep]


def run():
    cal = json.load(open(CAL))
    sectors = cal['sectors']
    dup = cal['correlation']['duplicate_sigma_groups']
    keys = list(sectors)

    # (a) duplicate-sigma resolutions
    cs = [stats.spearmanr([sectors[k]['adoption_rate_mean'] for k in keep],
                          [sectors[k]['btos_rate'] for k in keep])
          for keep in resolutions(keys, dup)]
    panel = pd.read_csv(PANEL)
    panel['key'] = panel.sector.map(NAME)
    assert panel.key.notna().all()
    pn = [stats.spearmanr(panel[panel.key.isin(keep)].model_d13_pp,
                          panel[panel.key.isin(keep)].delta_pp)
          for keep in resolutions(keys, dup)]
    # MC permutation p at the weakest cross-section resolution
    weakest = min(cs, key=lambda r: r.statistic)
    wk = min(resolutions(keys, dup),
             key=lambda keep: stats.spearmanr(
                 [sectors[k]['adoption_rate_mean'] for k in keep],
                 [sectors[k]['btos_rate'] for k in keep]).statistic)
    x = np.array([sectors[k]['adoption_rate_mean'] for k in wk])
    y = np.array([sectors[k]['btos_rate'] for k in wk])
    rng = np.random.default_rng(MC_SEED)
    hit = sum(abs(stats.spearmanr(rng.permutation(x), y).statistic)
              >= abs(weakest.statistic) - 1e-12 for _ in range(MC_DRAWS))

    # (b) vintage: pre-rewording column
    adopt = pd.read_csv(ADOPT)
    adopt['key'] = adopt.sector.map(NAME)
    assert adopt.key.notna().all()
    old = stats.spearmanr([sectors[k]['adoption_rate_mean'] for k in adopt.key],
                          adopt.yes2wk_old)
    new = stats.spearmanr([sectors[k]['adoption_rate_mean'] for k in adopt.key],
                          adopt.yes2wk_new)
    jump = (adopt.yes2wk_new - adopt.yes2wk_old)
    gaps_new = sorted(sectors[k]['adoption_rate_mean'] * 100 - v
                      for k, v in zip(adopt.key, adopt.yes2wk_new))
    gaps_old = sorted(sectors[k]['adoption_rate_mean'] * 100 - v
                      for k, v in zip(adopt.key, adopt.yes2wk_old))

    # (d) leave-one-out on the Table-6 design itself
    loo = []
    for drop in keys:
        keep_ = [k for k in keys if k != drop]
        r_ = stats.spearmanr([sectors[k]['adoption_rate_mean'] for k in keep_],
                             [sectors[k]['btos_rate'] for k in keep_])
        loo.append((float(r_.statistic), drop))
    loo.sort()

    # (c) McKinsey boundary case
    oos = json.load(open(OOS))
    a = oos['all_sectors']
    rec = {s: (sig, mck, mad) for s, sig, mck, mad
           in zip(a['sectors'], a['sigma'], a['mckinsey_2025'], a['model_adoption'])}
    S8 = ['Technology', 'Financial Services', 'Professional Services',
          'Healthcare', 'Retail', 'Manufacturing', 'Education', 'Logistics']
    def mck(secs):
        return stats.spearmanr([rec[s][2] for s in secs], [rec[s][1] for s in secs])
    m8, m9re, m10 = mck(S8), mck(S8 + ['Real Estate']), mck(S8 + ['Real Estate', 'Environment'])
    # Strictest literal construction: among the twelve BTOS-mapped McKinsey
    # sectors, keep only those whose elasticity is unique within the cross-check
    # (drops both members of each shared pair: technology/media-telecom,
    # professional-services/legal, real-estate/environment).
    S6 = ['Financial Services', 'Healthcare', 'Retail', 'Manufacturing',
          'Education', 'Logistics']
    m6 = mck(S6)

    out = {
        'duplicate_sigma_deduplication': {
            'n_resolutions': len(cs),
            'cross_section_rho_range': [float(min(r.statistic for r in cs)),
                                        float(max(r.statistic for r in cs))],
            'panel_model_vs_delta_rho_range': [float(min(r.statistic for r in pn)),
                                               float(max(r.statistic for r in pn))],
            'panel_model_vs_delta_p_range': [float(min(r.pvalue for r in pn)),
                                             float(max(r.pvalue for r in pn))],
            'cross_section_weakest_rho': float(weakest.statistic),
            'cross_section_weakest_mc_permutation_p': float(hit / MC_DRAWS),
            'mc_draws': MC_DRAWS, 'mc_seed': MC_SEED,
        },
        'question_rewording_vintage': {
            'cross_section_rho_pre_rewording': float(old.statistic),
            'cross_section_p_pre_rewording': float(old.pvalue),
            'cross_section_rho_post_rewording': float(new.statistic),
            'level_jump_pp_range': [float(jump.min()), float(jump.max())],
            'level_jump_pp_mean': float(jump.mean()),
            'model_minus_data_gap_pp_post': [gaps_new[0], gaps_new[-1]],
            'model_minus_data_gap_pp_pre': [gaps_old[0], gaps_old[-1]],
        },
        'leave_one_out_table6_design': {
            'loo_rho_range': [loo[0][0], loo[-1][0]],
            'dropped_at_min': loo[0][1],
            'dropped_at_max': loo[-1][1],
        },
        'mckinsey_boundary': {
            'strict6_unique_elasticity_rho': float(m6.statistic),
            'strict6_unique_elasticity_p': float(m6.pvalue),
            'matched8_rho': float(m8.statistic), 'matched8_p': float(m8.pvalue),
            'plus_real_estate_rho': float(m9re.statistic),
            'plus_real_estate_p': float(m9re.pvalue),
            'plus_real_estate_and_environment_rho': float(m10.statistic),
            'plus_real_estate_and_environment_p': float(m10.pvalue),
        },
    }
    path = os.path.join(OUTDIR, 'industry_calibration',
                        'calibration_dependence.json')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, path)
    print(f'-> {path}')
    for k, v in out.items():
        print(k, json.dumps(v))
    return out


if __name__ == '__main__':
    run()
