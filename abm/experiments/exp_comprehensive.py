"""Comprehensive experiment: all 4 improvements active, full parameter sweep.

Sweeps:
  - demand_shock_coefficient: 0.0, 0.04, 0.08, 0.12, 0.20
  - ai_efficiency_gain: 0.02, 0.04, 0.06, 0.08

With all improvements enabled:
  1. Per-firm sigma_i heterogeneity (markup distribution)
  2. S-curve adoption gating (short-run TFP calibration)
  3. Demand feedback A+B (paradox mechanism)
  4. Industry-calibrated markup range

Produces the main results table and figures for the paper.
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
from abm.src.step import _make_static_config, run_n_ticks, run_n_ticks_collect
from abm.src.config import merge_configs

import jax.numpy as jnp

# Parameter grid
DEMAND_COEFFS = [0.0, 0.04, 0.08, 0.12, 0.20]
AI_GAINS = [0.02, 0.04, 0.06, 0.08]

NUM_SEEDS = 20
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120  # v3.1 (2026-10-09): AI window unified to 120 ticks (10 yr) across all experiments
# (TRAJ_INTERVAL removed in v3.1 review: no trajectory collection in this script)

SAVE_DIR = os.path.join(_abm_dir, 'output', 'comprehensive')
os.makedirs(SAVE_DIR, exist_ok=True)


def compute_all_metrics(pre_firms, post_firms, post_market, sc):
    alpha = float(sc.alpha_CD)
    alive = post_firms.alive
    alive_pre = pre_firms.alive

    # Macro TFP — Q recomputed from the state (A_eff * q^theta * K^alpha
    # L^(1-alpha)); audit 2026-10-07: recorded firms.Q mixes within-tick K/L
    # with end-of-tick state and is 0 for newborn entrants.
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

    delta_tfp = (post_tfp - pre_tfp) / pre_tfp

    # Markup distribution over SURVIVING firms only.
    # Audit 2026-10-07: the old numerator sorted all 200 slots (dead firms
    # imputed with firm 0's mu) while the denominator used n_alive^2 —
    # inflating the Gini ~5x. Both sides now use the alive-only set.
    sigma_alive = post_firms.sigma_i[alive]
    mu_alive = sigma_alive / (sigma_alive - jnp.float32(1.0))
    mu_vals = jnp.sort(mu_alive)
    n_alive = int(jnp.sum(alive))
    if n_alive > 2:
        # Gini coefficient of markup
        mu_diff = jnp.abs(mu_vals[:, None] - mu_vals[None, :])
        gini = float(jnp.sum(mu_diff) / (2.0 * n_alive * n_alive * jnp.mean(mu_vals)))
    else:
        gini = 0.0
    markup_mean = float(jnp.mean(mu_alive))
    markup_std = float(jnp.std(mu_alive))

    # Market structure
    shares = jnp.where(alive, post_firms.market_share, 0)
    hhi = float(jnp.sum(shares ** 2))

    # AI adoption (deep adoption dropped 2026-10-07: ai_experience is capped
    # at 1.0 by the engine, so "> 1.0" was identically zero)
    adoption = float(jnp.sum((post_firms.ai_experience > 0.1) & alive) /
                     max(float(jnp.sum(alive)), 1))

    # Labor
    labor_share = float(post_market.labor_share)

    # Demand shock
    demand_shock = float(post_market.demand_externalit_shock)

    return {
        'delta_tfp': delta_tfp,
        'pre_tfp': pre_tfp,
        'post_tfp': post_tfp,
        'markup_mean': markup_mean,
        'markup_std': markup_std,
        'markup_gini': gini,
        'hhi': hhi,
        'adoption_rate': adoption,
        'labor_share': labor_share,
        'demand_shock': demand_shock,
    }


def run_one(config, seed_idx):
    sc = _make_static_config(config)
    seed_key = seed_idx * 1000 + 42
    state = init_simulation(config, seed=seed_key)
    pre = run_n_ticks(state, sc, PRE_TICKS)
    final = run_n_ticks(pre, sc, POST_TICKS)
    return compute_all_metrics(pre.firms, final.firms, final.market, sc)


def main():
    t0 = time.time()

    n_configs = len(AI_GAINS) * len(DEMAND_COEFFS)
    total_runs = n_configs * NUM_SEEDS
    print(f'Comprehensive experiment: all improvements active')
    print(f'  {len(AI_GAINS)} AI gains x {len(DEMAND_COEFFS)} demand coeffs '
          f'= {n_configs} configs x {NUM_SEEDS} seeds = {total_runs} runs')
    print(f'  {PRE_TICKS} pre-AI + {POST_TICKS} post-AI ticks')
    print()

    all_results = {}
    run_idx = 0

    for gain in AI_GAINS:
        for dsc in DEMAND_COEFFS:
            config = merge_configs({
                'ai_efficiency_gain': gain,
                'demand_shock_coefficient': dsc,
                # All improvements use defaults (already in config.py)
            })

            key = f'gain={gain}_demand={dsc}'
            per_seed = []

            for seed_idx in range(NUM_SEEDS):
                m = run_one(config, seed_idx)
                per_seed.append(m)
                run_idx += 1

            # Aggregate
            agg = {'ai_gain': gain, 'demand_coeff': dsc}
            for k in ['delta_tfp', 'pre_tfp', 'post_tfp', 'markup_mean', 'markup_std',
                      'markup_gini', 'hhi', 'adoption_rate',
                      'labor_share', 'demand_shock']:
                vals = [s[k] for s in per_seed]
                agg[f'{k}_mean'] = float(np.mean(vals))
                agg[f'{k}_std'] = float(np.std(vals))

            # Annualized TFP growth
            years = POST_TICKS / 12.0
            agg['annualized_tfp_mean'] = ((1 + agg['delta_tfp_mean']) ** (1/years) - 1)
            agg['annualized_tfp_std'] = float(np.std(
                [(1 + s['delta_tfp']) ** (1/years) - 1 for s in per_seed]))

            # Paradox rate: positive demand shock AND small TFP change
            agg['paradox_rate'] = sum(1 for s in per_seed
                                       if s['demand_shock'] > 0.01
                                       and abs(s['delta_tfp']) < 0.05) / NUM_SEEDS

            # Strong paradox: positive demand shock AND negative TFP
            agg['strong_paradox_rate'] = sum(1 for s in per_seed
                                              if s['demand_shock'] > 0.01
                                              and s['delta_tfp'] < 0) / NUM_SEEDS

            all_results[key] = agg

            pct = run_idx / total_runs * 100
            print(f'  [{pct:5.1f}%] gain={gain} demand={dsc}: '
                  f'dTFP={agg["delta_tfp_mean"]*100:+5.1f}% '
                  f'ann={agg["annualized_tfp_mean"]*100:+4.1f}%/yr '
                  f'adopt={agg["adoption_rate_mean"]:.0%} '
                  f'Gini={agg["markup_gini_mean"]:.3f} '
                  f'HHI={agg["hhi_mean"]:.4f} '
                  f'paradox={agg["paradox_rate"]:.0%}',
                  flush=True)

    elapsed = time.time() - t0
    print(f'\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}min)')

    raw_path = os.path.join(SAVE_DIR, 'comprehensive_results.json')
    with open(raw_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f'Saved: {raw_path}')

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
    gains = np.array(AI_GAINS)
    colors_gain = ['#FFC107', '#2196F3', '#4CAF50', '#F44336']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel A: TFP heatmap (gain x demand)
    ax = axes[0, 0]
    tfp_matrix = np.array([[all_results[f'gain={g}_demand={d}']['delta_tfp_mean'] * 100
                            for d in DEMAND_COEFFS] for g in AI_GAINS])
    im = ax.imshow(tfp_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-5, vmax=15)
    ax.set_xticks(range(len(DEMAND_COEFFS)))
    ax.set_xticklabels([str(d) for d in DEMAND_COEFFS])
    ax.set_yticks(range(len(AI_GAINS)))
    ax.set_yticklabels([f'{g}' for g in AI_GAINS])
    ax.set_xlabel('Demand shock coefficient')
    ax.set_ylabel('AI efficiency gain')
    ax.set_title('(a) Macro TFP Change (%)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    # Annotate cells
    for i in range(len(AI_GAINS)):
        for j in range(len(DEMAND_COEFFS)):
            v = tfp_matrix[i, j]
            color = 'white' if abs(v) > 7 else 'black'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8, color=color)

    # Panel B: Paradox rate heatmap
    ax = axes[0, 1]
    paradox_matrix = np.array([[all_results[f'gain={g}_demand={d}']['paradox_rate'] * 100
                                for d in DEMAND_COEFFS] for g in AI_GAINS])
    im = ax.imshow(paradox_matrix, cmap='Oranges', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(DEMAND_COEFFS)))
    ax.set_xticklabels([str(d) for d in DEMAND_COEFFS])
    ax.set_yticks(range(len(AI_GAINS)))
    ax.set_yticklabels([f'{g}' for g in AI_GAINS])
    ax.set_xlabel('Demand shock coefficient')
    ax.set_ylabel('AI efficiency gain')
    ax.set_title('(b) Paradox Rate (%)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(AI_GAINS)):
        for j in range(len(DEMAND_COEFFS)):
            v = paradox_matrix[i, j]
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8)

    # Panel C: Annualized TFP vs demand coefficient
    ax = axes[0, 2]
    for gi, gain in enumerate(AI_GAINS):
        means = [all_results[f'gain={gain}_demand={d}']['annualized_tfp_mean'] * 100
                 for d in DEMAND_COEFFS]
        stds = [all_results[f'gain={gain}_demand={d}']['annualized_tfp_std'] * 100
                for d in DEMAND_COEFFS]
        ax.errorbar(coeffs, means, yerr=stds, fmt='o-', color=colors_gain[gi],
                    capsize=3, linewidth=1.5, markersize=5, label=f'gain={gain}')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='US empirical')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
    ax.set_xlabel('Demand shock coefficient')
    ax.set_ylabel('Annualized TFP growth (%/yr)')
    ax.set_title('(c) TFP Growth Rate')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel D: AI adoption rate vs gain
    ax = axes[1, 0]
    for gi, gain in enumerate(AI_GAINS):
        means = [all_results[f'gain={gain}_demand={d}']['adoption_rate_mean'] * 100
                 for d in DEMAND_COEFFS]
        stds = [all_results[f'gain={gain}_demand={d}']['adoption_rate_std'] * 100
                for d in DEMAND_COEFFS]
        ax.errorbar(coeffs, means, yerr=stds, fmt='s-', color=colors_gain[gi],
                    capsize=3, linewidth=1.5, markersize=5, label=f'gain={gain}')
    ax.set_xlabel('Demand shock coefficient')
    ax.set_ylabel('AI adoption rate (%)')
    ax.set_title('(d) AI Adoption Rate')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel E: HHI vs demand coefficient
    ax = axes[1, 1]
    for gi, gain in enumerate(AI_GAINS):
        means = [all_results[f'gain={gain}_demand={d}']['hhi_mean']
                 for d in DEMAND_COEFFS]
        stds = [all_results[f'gain={gain}_demand={d}']['hhi_std']
                for d in DEMAND_COEFFS]
        ax.errorbar(coeffs, means, yerr=stds, fmt='^-', color=colors_gain[gi],
                    capsize=3, linewidth=1.5, markersize=5, label=f'gain={gain}')
    ax.set_xlabel('Demand shock coefficient')
    ax.set_ylabel('HHI')
    ax.set_title('(e) Market Concentration')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel F: Labor share vs demand coefficient
    ax = axes[1, 2]
    for gi, gain in enumerate(AI_GAINS):
        means = [all_results[f'gain={gain}_demand={d}']['labor_share_mean']
                 for d in DEMAND_COEFFS]
        stds = [all_results[f'gain={gain}_demand={d}']['labor_share_std']
                for d in DEMAND_COEFFS]
        ax.errorbar(coeffs, means, yerr=stds, fmt='D-', color=colors_gain[gi],
                    capsize=3, linewidth=1.5, markersize=5, label=f'gain={gain}')
    ax.set_xlabel('Demand shock coefficient')
    ax.set_ylabel('Labor share')
    ax.set_title('(f) Labor Share')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Comprehensive Results: AI Productivity Paradox ABM (All Improvements Active)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'F1_comprehensive_main.png'))
    print(f'Saved: F1_comprehensive_main.png')
    plt.close(fig)

    # Figure 2: Markup distribution summary
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes2[0]
    gini_means = [all_results[f'gain={g}_demand=0.08']['markup_gini_mean'] for g in AI_GAINS]
    gini_stds = [all_results[f'gain={g}_demand=0.08']['markup_gini_std'] for g in AI_GAINS]
    mu_means = [all_results[f'gain={g}_demand=0.08']['markup_mean_mean'] for g in AI_GAINS]
    ax.bar(range(len(AI_GAINS)), gini_means, yerr=gini_stds, capsize=3,
           color='#2196F3', alpha=0.8)
    ax.axhline(0.03, color='red', linestyle='--', linewidth=1, label='Target Gini>0.03')
    ax.set_xticks(range(len(AI_GAINS)))
    ax.set_xticklabels([f'{g}' for g in AI_GAINS])
    ax.set_xlabel('AI efficiency gain')
    ax.set_ylabel('Markup Gini coefficient')
    ax.set_title('(a) Markup Distribution Inequality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes2[1]
    markup_means = [all_results[f'gain={g}_demand=0.08']['markup_mean_mean'] for g in AI_GAINS]
    markup_stds_plot = [all_results[f'gain={g}_demand=0.08']['markup_std_mean'] for g in AI_GAINS]
    x_pos = np.arange(len(AI_GAINS))
    ax.bar(x_pos, markup_stds_plot, color='#FF9800', alpha=0.8, label='Markup std')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g}' for g in AI_GAINS])
    ax.set_xlabel('AI efficiency gain')
    ax.set_ylabel('Markup standard deviation')
    ax.set_title('(b) Markup Heterogeneity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig2.suptitle('Markup Distribution (demand_coeff=0.08)', fontsize=12)
    plt.tight_layout()
    fig2.savefig(os.path.join(fig_dir, 'F2_markup_distribution.png'))
    print(f'Saved: F2_markup_distribution.png')
    plt.close(fig2)

    print(f'\nAll figures saved to: {fig_dir}/')


if __name__ == '__main__':
    all_results = main()
    print('\nComprehensive experiment complete.')
