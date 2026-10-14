"""Counterfactual strategy lock experiment (Insight 1 verification).

Tests whether strategy imitation learning substitutes for demand externality.

Design: 2x5 factorial
  - strategy_learning_rate: 0.0 (locked) vs 0.05 (default)
  - demand_shock_coefficient: 0.0, 0.04, 0.08, 0.12, 0.20

If demand_shock_coeff becomes significant when imitation is off,
this proves "imitation substitutes for demand externality."
"""

import sys
import os
import json
import time

import numpy as np

_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_abm_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _abm_dir not in sys.path:
    sys.path.insert(0, _abm_dir)

from abm.src.state import init_simulation
from abm.src.step import _make_static_config, run_n_ticks
from abm.src.config import merge_configs

import jax.numpy as jnp

STRATEGY_RATES = [0.0, 0.05]
DEMAND_COEFFS = [0.0, 0.04, 0.08, 0.12, 0.20]
NUM_SEEDS = 20
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = int(os.environ.get('CF_POST_TICKS', 120))  # default: v3.1 unified 120-tick window

SAVE_DIR = os.path.join(_abm_dir, 'output', 'counterfactual_strategy')
os.makedirs(SAVE_DIR, exist_ok=True)

# Output filename: the default run archives as counterfactual_strategy_results.json
# (the manuscript's Table 4 source); a CF_POST_TICKS override writes to a
# suffixed file so the archived outputs can never be clobbered.  Verified
# 2026-10-11: CF_POST_TICKS=250 under this engine reproduces every
# manuscript-quoted double-horizon value of counterfactual_double_horizon.json
# at reporting precision (investment 0.135 in both arms; dTFP +35.2/+34.6%);
# unreported seed-dispersion fields differ in the last significant digits
# (max ~2.5% relative).
OUT_NAME = ('counterfactual_strategy_results.json' if POST_TICKS == 120
            else f'counterfactual_strategy_results_post{POST_TICKS}.json')


def compute_metrics(pre_firms, post_firms, post_market, sc):
    alpha = float(sc.alpha_CD)
    alive = post_firms.alive
    alive_pre = pre_firms.alive

    # Macro TFP — Q recomputed from the state (vintage-consistent; see
    # exp_comprehensive.py, audit 2026-10-07)
    theta = float(sc.quality_weight)
    gain = float(sc.ai_efficiency_gain)
    depth = float(sc.j_curve_depth)

    def tfp_of(firms, alive_mask):
        K = jnp.maximum(firms.K, 0.01)
        L = jnp.maximum(firms.L, 0.01)
        eff = jnp.power(K, alpha) * jnp.power(L, 1.0 - alpha)
        mult = jnp.where(
            firms.ai_experience > 0,
            1.0 + gain * ((1.0 + depth) * firms.ai_experience - depth),
            1.0)
        a_eff = firms.A_base * mult
        q = jnp.maximum(firms.product_quality, 0.1)
        Qhat = a_eff * jnp.power(q, theta) * eff
        return float(jnp.sum(Qhat[alive_mask]) /
                     max(float(jnp.sum(eff[alive_mask])), 0.01))

    post_tfp = tfp_of(post_firms, alive)
    pre_tfp = tfp_of(pre_firms, alive_pre)

    shares = jnp.where(alive, post_firms.market_share, 0)
    hhi = float(jnp.sum(shares ** 2))

    adoption = float(jnp.sum((post_firms.ai_experience > 0.1) & alive) /
                     max(float(jnp.sum(alive)), 1))

    # Strategy concentration: std of ai_strategy across firms
    # Low std = all firms converged to same strategy (herding)
    strategy_std = float(jnp.std(jnp.where(alive, post_firms.ai_strategy, 0.5)))
    strategy_mean = float(jnp.mean(jnp.where(alive, post_firms.ai_strategy, 0.5)))

    # AI investment arms race metric: mean investment rate
    mean_invest = float(jnp.mean(jnp.where(alive, post_firms.ai_investment, 0)))

    return {
        'delta_tfp': (post_tfp - pre_tfp) / pre_tfp,
        'hhi': hhi,
        'adoption_rate': adoption,
        'strategy_std': strategy_std,
        'strategy_mean': strategy_mean,
        'mean_ai_investment': mean_invest,
        'demand_shock': float(post_market.demand_externalit_shock),
        'labor_share': float(post_market.labor_share),
    }


def main():
    t0 = time.time()
    all_results = {}

    print('Counterfactual strategy lock experiment')
    print(f'  {len(STRATEGY_RATES)} strategy rates x {len(DEMAND_COEFFS)} demand coeffs '
          f'x {NUM_SEEDS} seeds')
    print(f'  {PRE_TICKS} pre + {POST_TICKS} post ticks')
    print()

    for sr in STRATEGY_RATES:
        for dsc in DEMAND_COEFFS:
            # Audit 2026-10-07: lr=0 previously only disabled the imitation
            # shift while the engine's exploration noise kept random-walking
            # strategies — "locked" was a misnomer. The sr=0 arm now sets
            # strategy_freeze=True, a true lock (no imitation AND no noise).
            config = merge_configs({
                'strategy_learning_rate': sr,
                'demand_shock_coefficient': dsc,
                'strategy_freeze': sr == 0.0,
            })
            sc = _make_static_config(config)

            key = f'lr={sr}_demand={dsc}'
            per_seed = []

            for seed_idx in range(NUM_SEEDS):
                seed_key = seed_idx * 1000 + 42
                state = init_simulation(config, seed=seed_key)
                pre = run_n_ticks(state, sc, PRE_TICKS)
                final = run_n_ticks(pre, sc, POST_TICKS)
                m = compute_metrics(pre.firms, final.firms, final.market, sc)
                per_seed.append(m)

            agg = {'strategy_lr': sr, 'demand_coeff': dsc}
            for k in ['delta_tfp', 'hhi', 'adoption_rate', 'strategy_std',
                      'strategy_mean', 'mean_ai_investment', 'demand_shock',
                      'labor_share']:
                vals = [s[k] for s in per_seed]
                agg[f'{k}_mean'] = float(np.mean(vals))
                agg[f'{k}_std'] = float(np.std(vals))

            agg['paradox_rate'] = sum(1 for s in per_seed
                                       if s['demand_shock'] > 0.01
                                       and abs(s['delta_tfp']) < 0.05) / NUM_SEEDS

            all_results[key] = agg

            print(f'  lr={sr} demand={dsc}: '
                  f'dTFP={agg["delta_tfp_mean"]*100:+6.1f}% '
                  f'strat_std={agg["strategy_std_mean"]:.3f} '
                  f'mean_invest={agg["mean_ai_investment_mean"]:.4f} '
                  f'HHI={agg["hhi_mean"]:.4f} '
                  f'paradox={agg["paradox_rate"]:.0%}',
                  flush=True)

    elapsed = time.time() - t0
    print(f'\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)')

    # --- Analysis: does demand sensitivity change with strategy lock? ---
    print('\n' + '='*60)
    print('KEY COMPARISON: demand sensitivity with vs without imitation')
    print('='*60)

    for sr in STRATEGY_RATES:
        dtfps = [all_results[f'lr={sr}_demand={d}']['delta_tfp_mean']
                 for d in DEMAND_COEFFS]
        base = dtfps[0]  # demand=0 baseline
        sensitivity = [(d - base) / max(abs(base), 0.01) * 100 for d in dtfps]
        print(f'\n  strategy_lr={sr}:')
        print(f'    dTFP at demand=0: {base*100:+.1f}%')
        print(f'    Relative sensitivity at demand=0.20: {sensitivity[-1]:+.1f}%')
        for i, d in enumerate(DEMAND_COEFFS):
            r = all_results[f'lr={sr}_demand={d}']
            print(f'    demand={d}: dTFP={dtfps[i]*100:+.1f}% '
                  f'rel={sensitivity[i]:+.1f}% '
                  f'paradox={r["paradox_rate"]:.0%} '
                  f'strat_std={r["strategy_std_mean"]:.3f} '
                  f'mean_invest={r["mean_ai_investment_mean"]:.4f}')

    # T-test between lr=0 and lr=0.05 at each demand level
    from scipy.stats import ttest_ind
    print(f'\n  Statistical comparison (lr=0 vs lr=0.05):')
    print(f'  {"Demand":>7s} {"dTFP diff":>10s} {"p-value":>10s} {"Signif":>7s}')
    print(f'  {"-"*38}')
    for dsc in DEMAND_COEFFS:
        # Reload per-seed data... actually we don't store it in agg.
        # Use mean/std for approximate z-test
        m0 = all_results[f'lr=0.0_demand={dsc}']['delta_tfp_mean']
        s0 = all_results[f'lr=0.0_demand={dsc}']['delta_tfp_std']
        m1 = all_results[f'lr=0.05_demand={dsc}']['delta_tfp_mean']
        s1 = all_results[f'lr=0.05_demand={dsc}']['delta_tfp_std']
        # Approximate two-sample t-test
        se = np.sqrt(s0**2/NUM_SEEDS + s1**2/NUM_SEEDS)
        t_stat = (m1 - m0) / max(se, 1e-8)
        # Approximate p-value (two-tailed)
        from scipy.stats import t as t_dist
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=2*NUM_SEEDS-2))
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        print(f'  {dsc:7.2f} {(m1-m0)*100:+9.1f}% {p_val:10.4f} {sig:>7s}')

    # Arms race metric comparison
    print(f'\n  AI investment arms race (mean_ai_investment):')
    print(f'  {"Demand":>7s} {"lr=0":>8s} {"lr=0.05":>8s} {"Ratio":>7s}')
    print(f'  {"-"*32}')
    for dsc in DEMAND_COEFFS:
        inv0 = all_results[f'lr=0.0_demand={dsc}']['mean_ai_investment_mean']
        inv1 = all_results[f'lr=0.05_demand={dsc}']['mean_ai_investment_mean']
        ratio = inv1 / max(inv0, 1e-8)
        print(f'  {dsc:7.2f} {inv0:8.4f} {inv1:8.4f} {ratio:7.2f}x')

    raw_path = os.path.join(SAVE_DIR, OUT_NAME)
    with open(raw_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f'\nSaved: {raw_path}')

    generate_figures(all_results)
    return all_results


def generate_figures(all_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(SAVE_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'font.family': 'serif', 'mathtext.fontset': 'cm',
    })

    coeffs = np.array(DEMAND_COEFFS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: dTFP vs demand coefficient (with/without imitation)
    ax = axes[0, 0]
    for sr, color, label in [(0.0, '#9E9E9E', 'No imitation (lr=0)'),
                              (0.05, '#2196F3', 'With imitation (lr=0.05)')]:
        means = [all_results[f'lr={sr}_demand={d}']['delta_tfp_mean'] * 100 for d in DEMAND_COEFFS]
        stds = [all_results[f'lr={sr}_demand={d}']['delta_tfp_std'] * 100 for d in DEMAND_COEFFS]
        ax.errorbar(coeffs, means, yerr=stds, fmt='o-' if sr > 0 else 's--',
                    color=color, capsize=4, linewidth=2, markersize=6, label=label)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
    ax.set_xlabel('Demand shock coefficient')
    ax.set_ylabel(r'$\Delta$ macro TFP (%)')
    ax.set_title(r'(a) TFP Response: Imitation Lock vs Free')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: Paradox rate
    ax = axes[0, 1]
    x = np.arange(len(DEMAND_COEFFS))
    width = 0.35
    p0 = [all_results[f'lr=0.0_demand={d}']['paradox_rate'] * 100 for d in DEMAND_COEFFS]
    p1 = [all_results[f'lr=0.05_demand={d}']['paradox_rate'] * 100 for d in DEMAND_COEFFS]
    ax.bar(x - width/2, p0, width, label='No imitation', color='#9E9E9E', alpha=0.8)
    ax.bar(x + width/2, p1, width, label='With imitation', color='#2196F3', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in DEMAND_COEFFS])
    ax.set_xlabel('Demand shock coefficient')
    ax.set_ylabel('Paradox rate (%)')
    ax.set_title('(b) Paradox Rate Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: AI investment arms race
    ax = axes[1, 0]
    for sr, color, label in [(0.0, '#9E9E9E', 'No imitation'),
                              (0.05, '#2196F3', 'With imitation')]:
        means = [all_results[f'lr={sr}_demand={d}']['mean_ai_investment_mean']
                 for d in DEMAND_COEFFS]
        stds = [all_results[f'lr={sr}_demand={d}']['mean_ai_investment_std']
                for d in DEMAND_COEFFS]
        ax.errorbar(coeffs, means, yerr=stds, fmt='o-' if sr > 0 else 's--',
                    color=color, capsize=4, linewidth=2, markersize=6, label=label)
    ax.set_xlabel('Demand shock coefficient')
    ax.set_ylabel('Mean AI investment rate')
    ax.set_title('(c) AI Investment Arms Race')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: Strategy diversity (std)
    ax = axes[1, 1]
    for sr, color, label in [(0.0, '#9E9E9E', 'No imitation'),
                              (0.05, '#2196F3', 'With imitation')]:
        means = [all_results[f'lr={sr}_demand={d}']['strategy_std_mean']
                 for d in DEMAND_COEFFS]
        stds = [all_results[f'lr={sr}_demand={d}']['strategy_std_std']
                for d in DEMAND_COEFFS]
        ax.errorbar(coeffs, means, yerr=stds, fmt='o-' if sr > 0 else 's--',
                    color=color, capsize=4, linewidth=2, markersize=6, label=label)
    ax.set_xlabel('Demand shock coefficient')
    ax.set_ylabel(r'Strategy $\sigma$ (diversity)')
    ax.set_title(r'(d) Strategy Diversity ($\sigma$)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Counterfactual: Does Imitation Learning Substitute for Demand Externality?',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'S4_counterfactual_strategy.png'))
    print(f'Saved: S4_counterfactual_strategy.png')
    plt.close(fig)

    print(f'\nAll figures saved to: {fig_dir}/')


if __name__ == '__main__':
    all_results = main()
    print('\nCounterfactual strategy lock experiment complete.')
