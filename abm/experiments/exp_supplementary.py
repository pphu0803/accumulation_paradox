"""Supplementary analyses: interesting findings from ABM v2.0.

Three analyses:
  1. Time-horizon crossover: finer-grained BTOS correlation trajectory
  2. Bistability in high-sigma sectors: path dependence and critical mass
  3. Markup inequality -> measured-TFP transmission: does Gini reduce TFP transmission?
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

BTOS_DATA = {
    'Information': 36.55, 'Professional Services': 33.35,
    'Finance and Insurance': 29.65, 'Educational Services': 27.75,
    'Multi-sector Companies': 23.0, 'Real Estate and Rental': 22.65,
    'Health Care': 19.75, 'Arts and Recreation': 17.85,
    'Wholesale Trade': 13.0, 'Retail Trade': 12.8,
    'Manufacturing': 11.6, 'Other Services': 9.95,
    'Construction': 8.35, 'Accommodation and Food': 8.35,
    'Transportation': 7.15,
}

INDUSTRY_MARKUP = {
    'Information': 1.55, 'Professional Services': 1.50,
    'Finance and Insurance': 1.60, 'Educational Services': 1.20,
    'Multi-sector Companies': 1.35, 'Real Estate and Rental': 1.35,
    'Health Care': 1.40, 'Arts and Recreation': 1.15,
    'Wholesale Trade': 1.18, 'Retail Trade': 1.15,
    'Manufacturing': 1.25, 'Other Services': 1.20,
    'Construction': 1.10, 'Accommodation and Food': 1.10,
    'Transportation': 1.12,
}

SAVE_DIR = os.path.join(_abm_dir, 'output', 'supplementary')
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Analysis 1: Time-horizon crossover (finer-grained)
# ============================================================

def analysis1_time_crossover():
    from scipy.stats import spearmanr

    print('\n' + '='*60)
    print('ANALYSIS 1: Time-horizon BTOS correlation crossover')
    print('='*60)

    sectors = list(BTOS_DATA.keys())
    btos_arr = np.array([BTOS_DATA[s] for s in sectors])
    sigma_map = {s: INDUSTRY_MARKUP[s] / (INDUSTRY_MARKUP[s] - 1.0) for s in sectors}

    NUM_SEEDS = 15
    PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
    POST_TICKS = 120
    SAMPLE_EVERY = 3  # every quarter-year for fine resolution

    per_sector_seed = {}
    for sector in sectors:
        sigma = sigma_map[sector]
        per_sector_seed[sector] = []
        for seed_idx in range(NUM_SEEDS):
            config = merge_configs({
                'sigma_base': sigma, 'sigma_firm_spread': 0.15, 'sigma_firm_A_sensitivity': 0.3,
            })
            sc = _make_static_config(config)
            seed_key = seed_idx * 1000 + 42
            state = init_simulation(config, seed=seed_key)
            pre = run_n_ticks(state, sc, PRE_TICKS)
            final, sampled = run_n_ticks_collect(pre, sc, POST_TICKS, sample_every=SAMPLE_EVERY)

            n_samples = sampled.firms.alive.shape[0]
            traj = []
            for i in range(n_samples):
                alive = sampled.firms.alive[i]
                a = float(jnp.sum((sampled.firms.ai_experience[i] > 0.1) & alive) /
                         max(float(jnp.sum(alive)), 1))
                traj.append(a)
            per_sector_seed[sector].append(traj)

    # Compute correlation at each time point
    n_t = len(per_sector_seed[sectors[0]][0])
    rho_traj = []
    p_traj = []
    tick_labels = []

    for t_idx in range(n_t):
        model_at_t = []
        for sector in sectors:
            vals = [per_sector_seed[sector][s][t_idx] for s in range(NUM_SEEDS)]
            model_at_t.append(np.mean(vals))
        model_arr = np.array(model_at_t)
        if np.std(model_arr) < 1e-8:
            rho_traj.append(np.nan)
            p_traj.append(np.nan)
        else:
            r, p = spearmanr(model_arr, btos_arr)
            rho_traj.append(r)
            p_traj.append(p)
        tick_labels.append((t_idx + 1) * SAMPLE_EVERY)

    # Find crossover point
    sign_changes = []
    for i in range(1, len(rho_traj)):
        if not (np.isnan(rho_traj[i]) or np.isnan(rho_traj[i-1])):
            if rho_traj[i-1] * rho_traj[i] < 0:
                sign_changes.append((tick_labels[i], rho_traj[i-1], rho_traj[i]))

    print(f'  Fine-grained: every {SAMPLE_EVERY} ticks, {n_t} samples')
    print(f'  Sign changes: {len(sign_changes)}')
    for tc in sign_changes:
        print(f'    tick {tc[0]} ({tc[0]/12:.1f}yr): rho {tc[1]:.3f} -> {tc[2]:.3f}')

    # Also compute adoption rate differential (high-BTOS vs low-BTOS sectors)
    high_btos = ['Information', 'Professional Services', 'Finance and Insurance']
    low_btos = ['Construction', 'Accommodation and Food', 'Transportation']

    high_adopt_traj = []
    low_adopt_traj = []
    for t_idx in range(n_t):
        h = np.mean([np.mean([per_sector_seed[s][si][t_idx] for si in range(NUM_SEEDS)])
                     for s in high_btos])
        l = np.mean([np.mean([per_sector_seed[s][si][t_idx] for si in range(NUM_SEEDS)])
                     for s in low_btos])
        high_adopt_traj.append(h)
        low_adopt_traj.append(l)

    results = {
        'tick_labels': tick_labels,
        'rho': [None if np.isnan(v) else float(v) for v in rho_traj],
        'p_values': [None if np.isnan(v) else float(v) for v in p_traj],
        'sign_changes': sign_changes,
        'high_btos_adoption': high_adopt_traj,
        'low_btos_adoption': low_adopt_traj,
    }

    with open(os.path.join(SAVE_DIR, 'analysis1_crossover.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float, allow_nan=False)
    print(f'  Saved: analysis1_crossover.json')

    generate_fig1(results)
    return results


def generate_fig1(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'font.family': 'serif', 'mathtext.fontset': 'cm',
    })

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    years = np.array(results['tick_labels']) / 12.0
    rho = np.array(results['rho'])

    # Panel A: Spearman rho over time
    ax = axes[0]
    valid = ~np.isnan(rho)
    ax.plot(years[valid], rho[valid], 'o-', color='#2196F3', linewidth=2, markersize=4)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.fill_between(years[valid], 0, rho[valid],
                    where=rho[valid] > 0, alpha=0.15, color='green', label='Positive (concentrated sectors lead)')
    ax.fill_between(years[valid], 0, rho[valid],
                    where=rho[valid] < 0, alpha=0.15, color='red', label='Negative (competitive sectors lead)')
    ax.set_xlabel('Years after AI introduction')
    ax.set_ylabel("Spearman $\\rho$ (model vs BTOS)")
    ax.set_title('(a) Time-Horizon BTOS Correlation Crossover')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    # Annotate crossover
    for sc_data in results['sign_changes']:
        yr = sc_data[0] / 12.0
        ax.axvline(yr, color='purple', linestyle=':', linewidth=1, alpha=0.7)
        ax.annotate(f'crossover\n@{yr:.1f}yr', xy=(yr, 0), fontsize=7,
                   ha='center', va='bottom', xytext=(0, 10), textcoords='offset points',
                   color='purple')

    # Panel B: Adoption trajectories by sector group
    ax = axes[1]
    high = np.array(results['high_btos_adoption']) * 100
    low = np.array(results['low_btos_adoption']) * 100
    ax.plot(years, high, '-', color='#2196F3', linewidth=2, label='High BTOS sectors (low $\\sigma$)')
    ax.plot(years, low, '-', color='#F44336', linewidth=2, label='Low BTOS sectors (high $\\sigma$)')
    ax.fill_between(years, high, low, alpha=0.1, color='gray')
    ax.set_xlabel('Years after AI introduction')
    ax.set_ylabel('AI adoption rate (%)')
    ax.set_title('(b) Adoption Trajectories: High vs Low BTOS Sectors')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Supplementary Analysis 1: Time-Horizon Correlation Crossover', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'figures', 'S1_time_crossover.png'))
    print(f'  Saved: S1_time_crossover.png')
    plt.close(fig)


# ============================================================
# Analysis 2: Bistability in high-sigma sectors
# ============================================================

def analysis2_bistability():
    print('\n' + '='*60)
    print('ANALYSIS 2: Bistability in high-sigma sectors')
    print('='*60)

    # Three-sigma comparison (audit 2026-10-07): the committed archive
    # (2026-06-18) contains High/Mid/Low x 40 seeds and the bimodality lives in
    # the Mid-sigma (Manufacturing, sigma=5) group — the Mid arm was lost from
    # the script at some point, making fig3 un-regenerable. Restored here.
    sigma_high = 11.0
    sigma_mid = 5.0    # Manufacturing
    sigma_low = 2.8    # Information

    NUM_SEEDS = 40
    PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
    POST_TICKS = 120
    SAMPLE_EVERY = 6

    results = {}

    for label, sigma_val in [('High-sigma (Construction)', sigma_high),
                             ('Mid-sigma (Manufacturing)', sigma_mid),
                             ('Low-sigma (Information)', sigma_low)]:
        config = merge_configs({
            'sigma_base': sigma_val, 'sigma_firm_spread': 0.15, 'sigma_firm_A_sensitivity': 0.3,
        })
        sc = _make_static_config(config)

        final_adoptions = []
        trajectories = []
        final_hhi = []
        final_top4 = []

        for seed_idx in range(NUM_SEEDS):
            seed_key = seed_idx * 1000 + 42
            state = init_simulation(config, seed=seed_key)
            pre = run_n_ticks(state, sc, PRE_TICKS)
            final, sampled = run_n_ticks_collect(pre, sc, POST_TICKS, sample_every=SAMPLE_EVERY)

            alive = final.firms.alive
            adoption = float(jnp.sum((final.firms.ai_experience > 0.1) & alive) /
                             max(float(jnp.sum(alive)), 1))
            final_adoptions.append(adoption)

            # Trajectory
            traj = []
            for i in range(sampled.firms.alive.shape[0]):
                a = float(jnp.sum((sampled.firms.ai_experience[i] > 0.1) & sampled.firms.alive[i]) /
                         max(float(jnp.sum(sampled.firms.alive[i])), 1))
                traj.append(a)
            trajectories.append(traj)

            shares = jnp.where(alive, final.firms.market_share, 0)
            final_hhi.append(float(jnp.sum(shares**2)))
            top4 = jnp.sort(shares)[-4:]
            final_top4.append(float(jnp.sum(top4)))

        final_adoptions = np.array(final_adoptions)
        final_hhi = np.array(final_hhi)
        final_top4 = np.array(final_top4)
        trajectories = np.array(trajectories)

        # Bimodality test: histogram
        n_low = np.sum(final_adoptions < 0.3)
        n_mid = np.sum((final_adoptions >= 0.3) & (final_adoptions <= 0.7))
        n_high = np.sum(final_adoptions > 0.7)

        # Check if distribution is bimodal (large gap between modes)
        sorted_a = np.sort(final_adoptions)
        gaps = np.diff(sorted_a)
        max_gap_idx = np.argmax(gaps)
        threshold = (sorted_a[max_gap_idx] + sorted_a[max_gap_idx+1]) / 2

        results[label] = {
            'sigma': sigma_val,
            'adoptions': final_adoptions.tolist(),
            'hhis': final_hhi.tolist(),
            'top4s': final_top4.tolist(),
            'adoption_mean': float(final_adoptions.mean()),
            'adoption_std': float(final_adoptions.std()),
            'adoption_cv': float(final_adoptions.std() / final_adoptions.mean())
                if final_adoptions.mean() > 0 else 0.0,
            'trajectories': trajectories.tolist(),
            'n_low': int(n_low), 'n_mid': int(n_mid), 'n_high': int(n_high),
            'gap_threshold': float(threshold),
            'max_gap': float(gaps[max_gap_idx]),
        }

        print(f'  {label} (sigma={sigma_val}):')
        print(f'    Adoption: mean={final_adoptions.mean()*100:.1f}% std={final_adoptions.std()*100:.1f}%')
        print(f'    Bins: low(<30%)={n_low} mid(30-70%)={n_mid} high(>70%)={n_high}')
        if n_low > 5 and n_high > 5:
            print(f'    ** BIMODAL: two basins of attraction detected')
            print(f'    Gap threshold: {threshold*100:.1f}%')
        elif n_mid > NUM_SEEDS * 0.5:
            print(f'    UNIMODAL with wide spread (continuous distribution)')
        else:
            print(f'    UNIMODAL: concentrated around {final_adoptions.mean()*100:.0f}%')
        print(f'    HHI: mean={final_hhi.mean():.4f} std={final_hhi.std():.4f}')

    with open(os.path.join(SAVE_DIR, 'analysis2_bistability.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'  Saved: analysis2_bistability.json')

    generate_fig2(results)
    return results


def generate_fig2(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'font.family': 'serif', 'mathtext.fontset': 'cm',
    })

    labels = list(results.keys())
    n_groups = len(labels)
    fig, axes = plt.subplots(2, n_groups, figsize=(7 * n_groups, 10))
    if n_groups == 1:
        axes = axes[:, None]

    for idx, label in enumerate(labels):
        d = results[label]
        adoptions = np.array(d['adoptions'])

        # Panel A/C/E: Adoption histogram
        ax = axes[0, idx]
        ax.hist(adoptions * 100, bins=20, color=['#2196F3', '#4CAF50', '#F44336'][idx % 3],
                alpha=0.7, edgecolor='white')
        if d['gap_threshold'] > 0:
            ax.axvline(d['gap_threshold'] * 100, color='red', linestyle='--',
                       linewidth=1, label='Gap threshold')
        ax.set_xlabel('Final AI adoption rate (%)')
        ax.set_ylabel('Frequency')
        title_label = f"$\\sigma$={d['sigma']:.1f} ({label.split('(')[-1].rstrip(')')})"
        ax.set_title(f'({chr(ord("a") + 2 * idx)}) Adoption Distribution ({title_label})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Panel B/D/F: Trajectory spaghetti
        ax = axes[1, idx]
        traj = np.array(d['trajectories'])
        n_seeds = traj.shape[0]
        # sample i is the state after tick 6*(i+1) (engine samples at every
        # 6th step since the 2026-10-07 phase fix)
        years = (np.arange(traj.shape[1]) + 1) * 6 / 12
        for s in range(min(n_seeds, 30)):
            ax.plot(years, traj[s] * 100, '-', alpha=0.15, linewidth=0.5,
                    color=['#2196F3', '#4CAF50', '#F44336'][idx % 3])
        # Mean trajectory
        ax.plot(years, traj.mean(axis=0) * 100, 'k-', linewidth=2, label='Mean')
        ax.plot(years, np.median(traj, axis=0) * 100, 'k--', linewidth=1.5, label='Median')
        ax.set_xlabel('Years after AI introduction')
        ax.set_ylabel('AI adoption rate (%)')
        ax.set_title(f'({chr(ord("b") + 2 * idx)}) Adoption Trajectories ({title_label})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Supplementary Analysis 2: Bistability in High-$\\sigma$ Sectors',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'figures', 'S2_bistability.png'))
    print(f'  Saved: S2_bistability.png')
    plt.close(fig)


# ============================================================
# Analysis 3: Markup inequality -> measured-TFP transmission
# ============================================================

def analysis3_markup_tfp():
    print('\n' + '='*60)
    print('ANALYSIS 3: Markup inequality -> TFP transmission')
    print('='*60)

    # Vary sigma_firm_spread to get different Gini levels
    # Then vary AI gain to see how TFP transmission changes with Gini
    SPREADS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    GAINS = [0.02, 0.04, 0.06, 0.08]

    NUM_SEEDS = 20
    PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
    POST_TICKS = 120  # v3.1 (2026-10-09): AI window unified to 120 ticks (10 yr)

    results = {}

    for spread in SPREADS:
        results[f'spread={spread}'] = {}
        for gain in GAINS:
            config = merge_configs({
                'sigma_firm_spread': spread,
                'sigma_firm_A_sensitivity': 0.5,
                'ai_efficiency_gain': gain,
            })
            sc = _make_static_config(config)

            tfps = []
            ginis = []

            for seed_idx in range(NUM_SEEDS):
                seed_key = seed_idx * 1000 + 42
                state = init_simulation(config, seed=seed_key)
                pre = run_n_ticks(state, sc, PRE_TICKS)
                final = run_n_ticks(pre, sc, POST_TICKS)

                alpha = float(sc.alpha_CD)
                alive = final.firms.alive
                alive_pre = pre.firms.alive

                K = jnp.maximum(final.firms.K, 0.01)
                L = jnp.maximum(final.firms.L, 0.01)
                eff = jnp.power(K, alpha) * jnp.power(L, 1.0 - alpha)
                post_tfp = float(jnp.sum(final.firms.Q[alive]) / max(float(jnp.sum(eff[alive])), 0.01))

                K0 = jnp.maximum(pre.firms.K, 0.01)
                L0 = jnp.maximum(pre.firms.L, 0.01)
                eff0 = jnp.power(K0, alpha) * jnp.power(L0, 1.0 - alpha)
                pre_tfp = float(jnp.sum(pre.firms.Q[alive_pre]) / max(float(jnp.sum(eff0[alive_pre])), 0.01))

                d_tfp = (post_tfp - pre_tfp) / pre_tfp
                tfps.append(d_tfp)

                # Markup Gini over SURVIVING firms only.
                # Audit 2026-10-07: the old version sorted all 200 slots
                # (dead firms imputed at sigma=3 -> mu=1.5) into the pairwise
                # numerator while dividing by 2*n_alive^2*mean — inflating the
                # Gini 3-7x depending on mortality. Numerator and denominator
                # now use the same alive-only set.
                sigma_alive = final.firms.sigma_i[alive]
                mu_alive = sigma_alive / (sigma_alive - jnp.float32(1.0))
                mu_vals = jnp.sort(mu_alive)
                n = int(jnp.sum(alive))
                if n > 2:
                    diff = jnp.abs(mu_vals[:, None] - mu_vals[None, :])
                    g = float(jnp.sum(diff) / (2.0 * n * n * jnp.mean(mu_vals)))
                else:
                    g = 0.0
                ginis.append(g)

            agg = {
                'delta_tfp_mean': float(np.mean(tfps)),
                'delta_tfp_std': float(np.std(tfps)),
                'markup_gini_mean': float(np.mean(ginis)),
                'markup_gini_std': float(np.std(ginis)),
                'annualized_tfp': ((1 + np.mean(tfps)) ** (12/POST_TICKS) - 1),
            }
            results[f'spread={spread}'][f'gain={gain}'] = agg

    # Print summary
    print(f'  {"Spread":>8s} {"Gain":>5s} {"Gini":>6s} {"dTFP%":>7s} {"Ann%/yr":>8s}')
    print(f'  {"-"*38}')
    for spread in SPREADS:
        for gain in GAINS:
            d = results[f'spread={spread}'][f'gain={gain}']
            print(f'  {spread:8.1f} {gain:5.2f} {d["markup_gini_mean"]:6.4f} '
                  f'{d["delta_tfp_mean"]*100:+7.1f} {d["annualized_tfp"]*100:+7.1f}')
        print()

    # TFP transmission efficiency: delta_TFP / gain
    print('  TFP transmission efficiency (delta_TFP / AI_gain):')
    print(f'  {"Spread":>8s} {"Gini":>6s} ', end='')
    for gain in GAINS:
        print(f'{f"g={gain}":>8s} ', end='')
    print()
    print(f'  {"-"*(8 + 6 + 8*len(GAINS) + len(GAINS))}')
    for spread in SPREADS:
        gini = results[f'spread={spread}'][f'gain={GAINS[0]}']['markup_gini_mean']
        print(f'  {spread:8.1f} {gini:6.4f} ', end='')
        for gain in GAINS:
            d = results[f'spread={spread}'][f'gain={gain}']
            efficiency = d['delta_tfp_mean'] / gain if gain > 0 else 0
            print(f'{efficiency:8.2f} ', end='')
        print()

    with open(os.path.join(SAVE_DIR, 'analysis3_markup_tfp.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'  Saved: analysis3_markup_tfp.json')

    generate_fig3(results, SPREADS, GAINS)
    return results


def generate_fig3(results, SPREADS, GAINS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'font.family': 'serif', 'mathtext.fontset': 'cm',
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#FFC107', '#2196F3', '#4CAF50', '#F44336']

    # Panel A: dTFP vs Gini (for each gain level)
    ax = axes[0]
    for gi, gain in enumerate(GAINS):
        ginis = [results[f'spread={s}'][f'gain={gain}']['markup_gini_mean'] for s in SPREADS]
        dtfps = [results[f'spread={s}'][f'gain={gain}']['delta_tfp_mean'] * 100 for s in SPREADS]
        ax.plot(ginis, dtfps, 'o-', color=colors[gi], linewidth=2, markersize=6,
                label=f'gain={gain}')
    ax.set_xlabel('Markup Gini')
    ax.set_ylabel(r'$\Delta$ macro TFP (%)')
    ax.set_title('(a) TFP vs Markup Inequality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: TFP transmission efficiency vs Gini
    ax = axes[1]
    for gi, gain in enumerate(GAINS):
        ginis = [results[f'spread={s}'][f'gain={gain}']['markup_gini_mean'] for s in SPREADS]
        efficiencies = [results[f'spread={s}'][f'gain={gain}']['delta_tfp_mean'] / gain
                       for s in SPREADS]
        ax.plot(ginis, efficiencies, 's-', color=colors[gi], linewidth=2, markersize=6,
                label=f'gain={gain}')
    ax.set_xlabel('Markup Gini')
    ax.set_ylabel(r'TFP transmission: $\Delta$TFP / AI gain')
    ax.set_title('(b) Transmission Efficiency vs Inequality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Heatmap of annualized TFP
    ax = axes[2]
    ann_matrix = np.array([[results[f'spread={s}'][f'gain={g}']['annualized_tfp'] * 100
                             for g in GAINS] for s in SPREADS])
    im = ax.imshow(ann_matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(GAINS)))
    ax.set_xticklabels([str(g) for g in GAINS])
    ax.set_yticks(range(len(SPREADS)))
    ax.set_yticklabels([f'{s:.1f}' for s in SPREADS])
    ax.set_xlabel('AI efficiency gain')
    ax.set_ylabel(r'$\sigma$ firm spread')
    ax.set_title(r'(c) Annualized TFP (%) by Gini $\times$ gain')
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(SPREADS)):
        for j in range(len(GAINS)):
            ax.text(j, i, f'{ann_matrix[i,j]:.1f}', ha='center', va='center', fontsize=8)

    fig.suptitle('Supplementary Analysis 3: Markup Inequality and TFP Transmission',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'figures', 'S3_markup_tfp.png'))
    print(f'  Saved: S3_markup_tfp.png')
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['a1', 'a2', 'a3'], default=None,
                    help='run a single analysis (default: all three)')
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(SAVE_DIR, 'figures'), exist_ok=True)

    which = [args.only] if args.only else ['a1', 'a2', 'a3']
    if 'a1' in which:
        r1 = analysis1_time_crossover()
    if 'a2' in which:
        r2 = analysis2_bistability()
    if 'a3' in which:
        r3 = analysis3_markup_tfp()

    elapsed = time.time() - t0
    print(f'\nSupplementary analyses complete ({",".join(which)}). Total: {elapsed:.0f}s ({elapsed/60:.1f}min)')


if __name__ == '__main__':
    main()
