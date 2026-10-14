"""Industry-level BTOS calibration: map sector markup -> sigma -> model AI adoption.

Uses BTOS AI adoption rates (yes2wk_new) as empirical target.
Maps each sector to a markup estimate from DEU (2020) and related literature,
then converts to sigma = mu/(mu-1) and runs the ABM per-sector.

Tests whether the model's predicted AI adoption ranking correlates
with BTOS empirical adoption rates (target: Spearman rho > 0.5).
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

# BTOS sector AI adoption rates (2-week usage, latest wave)
# Source: btos_ai_adoption_by_sector.csv, yes2wk_new column
BTOS_DATA = {
    'Information': 36.55,
    'Professional Services': 33.35,
    'Finance and Insurance': 29.65,
    'Educational Services': 27.75,
    'Multi-sector Companies': 23.0,
    'Real Estate and Rental': 22.65,
    'Health Care': 19.75,
    'Arts and Recreation': 17.85,
    'Wholesale Trade': 13.0,
    'Retail Trade': 12.8,
    'Manufacturing': 11.6,
    'Other Services': 9.95,
    'Construction': 8.35,
    'Accommodation and Food': 8.35,
    'Transportation': 7.15,
}

# Industry markup estimates from DEU (2020) "Market Concentration"
# and related literature (Autor et al. 2020, Baslandze 2024).
# sigma = mu / (mu - 1)
INDUSTRY_MARKUP = {
    'Information': 1.55,
    'Professional Services': 1.50,
    'Finance and Insurance': 1.60,
    'Educational Services': 1.20,
    'Multi-sector Companies': 1.35,
    'Real Estate and Rental': 1.35,
    'Health Care': 1.40,
    'Arts and Recreation': 1.15,
    'Wholesale Trade': 1.18,
    'Retail Trade': 1.15,
    'Manufacturing': 1.25,
    'Other Services': 1.20,
    'Construction': 1.10,
    'Accommodation and Food': 1.10,
    'Transportation': 1.12,
}

NUM_SEEDS = int(os.environ.get('NUM_SEEDS', '15'))
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120  # 10 years post-AI
TRAJECTORY_INTERVALS = [12, 24, 36, 60, 120]  # tick snapshots for time analysis
SAVE_DIR = os.path.join(_abm_dir, 'output', 'industry_calibration')
os.makedirs(SAVE_DIR, exist_ok=True)


def markup_to_sigma(mu):
    return mu / (mu - 1.0)


def compute_sector_metrics(pre_firms, post_firms, sc):
    alive = post_firms.alive
    alive_pre = pre_firms.alive

    # AI adoption rate (fraction of alive firms with meaningful experience)
    adoption = float(jnp.sum((post_firms.ai_experience > 0.1) & alive) /
                     max(float(jnp.sum(alive)), 1))

    # AI adoption at meaningful experience thresholds. Audit 2026-10-07:
    # thresholds 1.0/3.0 dropped — the engine caps ai_experience at 1.0, so
    # "> 1.0" and "> 3.0" were identically zero.
    early_adoption = []
    for threshold in [0.05, 0.1, 0.3, 0.5, 0.9]:
        a = float(jnp.sum((post_firms.ai_experience > threshold) & alive) /
                 max(float(jnp.sum(alive)), 1))
        early_adoption.append(a)

    # Mean AI investment over SURVIVING firms (was diluted by dead slots)
    mean_invest = float(jnp.mean(post_firms.ai_investment[alive])) \
        if bool(jnp.sum(alive)) else 0.0

    # Mean markup (alive firms only; sigma/(sigma-1) bound-implied markup)
    sigma_i = post_firms.sigma_i
    mu_i = sigma_i / (sigma_i - 1.0)
    mean_markup = (float(jnp.mean(mu_i[alive])) if bool(jnp.sum(alive)) else 1.0)

    # HHI
    shares = jnp.where(alive, post_firms.market_share, 0)
    hhi = float(jnp.sum(shares ** 2))

    return {
        'adoption_rate': adoption,
        'adoption_by_threshold': early_adoption,
        'mean_ai_investment': mean_invest,
        'mean_markup': mean_markup,
        'hhi': hhi,
    }


def run_sector(sector_name, sigma_val, config_base, seed_idx):
    overrides = {
        'sigma_base': sigma_val,
        'sigma_firm_spread': 0.15,  # small within-sector spread
        'sigma_firm_A_sensitivity': 0.3,
    }
    config = merge_configs(config_base, overrides)
    sc = _make_static_config(config)

    seed_key = seed_idx * 1000 + 42
    state = init_simulation(config, seed=seed_key)
    pre = run_n_ticks(state, sc, PRE_TICKS)

    # Use run_n_ticks_collect to get states at every tick
    final, sampled = run_n_ticks_collect(pre, sc, POST_TICKS, sample_every=12)

    # Compute trajectory adoption at sampled intervals (every 12 ticks = 1 year)
    n_samples = sampled.firms.alive.shape[0]
    traj = {}
    for i in range(n_samples):
        tick = (i + 1) * 12
        alive = sampled.firms.alive[i]
        a = float(jnp.sum((sampled.firms.ai_experience[i] > 0.1) & alive) /
                 max(float(jnp.sum(alive)), 1))
        traj[tick] = a

    metrics = compute_sector_metrics(pre.firms, final.firms, sc)
    metrics['trajectory'] = traj
    return metrics


def main():
    t0 = time.time()

    sectors = list(BTOS_DATA.keys())
    n_sectors = len(sectors)

    # Compute sigma values
    sigma_map = {}
    for s, mu in INDUSTRY_MARKUP.items():
        sigma_map[s] = markup_to_sigma(mu)

    print(f'Industry-level BTOS calibration')
    print(f'  {n_sectors} sectors x {NUM_SEEDS} seeds x {PRE_TICKS}+{POST_TICKS} ticks')
    print()
    print(f'  {"Sector":<28s} {"Markup":>6s} {"sigma":>6s} {"BTOS%":>7s} '
          f'{"Model%":>8s} {"Model_std":>9s}')
    print(f'  {"-"*70}')

    all_results = {}
    model_adoption_means = []
    btos_rates = []

    # Resume fully-completed sectors from the per-sector checkpoint
    _ckpt_path = os.path.join(SAVE_DIR, 'industry_calibration_partial.json')
    if os.path.exists(_ckpt_path):
        with open(_ckpt_path) as f:
            cached = json.load(f)
        for sector, agg in cached.items():
            if len(agg.get('_per_seed', [])) >= NUM_SEEDS:
                for row in agg['_per_seed']:
                    if 'trajectory' in row:
                        row['trajectory'] = {int(k): v
                                             for k, v in row['trajectory'].items()}
                all_results[sector] = agg
        if all_results:
            print(f'  Resumed {len(all_results)} sectors from checkpoint')

    for sector in sectors:
        if sector in all_results:
            agg = all_results[sector]
            model_adoption_means.append(agg['adoption_rate_mean'])
            btos_rates.append(agg['btos_rate'])
            print(f'  {sector:<28s} (resumed: '
                  f'{agg["adoption_rate_mean"]*100:6.1f}%)')
            continue

        btos_rate = BTOS_DATA[sector]
        mu = INDUSTRY_MARKUP[sector]
        sigma = sigma_map[sector]

        per_seed = []
        for seed_idx in range(NUM_SEEDS):
            m = run_sector(sector, sigma, {}, seed_idx)
            per_seed.append(m)

        agg = {
            'sector': sector,
            'btos_rate': btos_rate,
            'markup': mu,
            'sigma': sigma,
        }
        for k in ['adoption_rate', 'mean_ai_investment', 'mean_markup', 'hhi']:
            vals = [s[k] for s in per_seed]
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals))

        # Adoption by experience threshold (mean across seeds)
        thresh_means = np.mean([s['adoption_by_threshold'] for s in per_seed], axis=0)
        agg['adoption_by_threshold_mean'] = thresh_means.tolist()

        model_adoption_means.append(agg['adoption_rate_mean'])
        btos_rates.append(btos_rate)
        all_results[sector] = agg
        all_results[sector]['_per_seed'] = per_seed

        # Checkpoint after each sector so long runs survive interruption
        _ckpt = os.path.join(SAVE_DIR, 'industry_calibration_partial.json')
        with open(_ckpt, 'w') as f:
            json.dump(all_results, f, default=str)

        # Each sector has a distinct sigma_base -> fresh XLA compilations;
        # clear the compile cache or memory grows ~0.8GB per sector.
        import jax
        jax.clear_caches()

        print(f'  {sector:<28s} {mu:6.2f} {sigma:6.2f} {btos_rate:6.1f}% '
              f'{agg["adoption_rate_mean"]*100:6.1f}% {agg["adoption_rate_std"]*100:7.1f}%',
              flush=True)

    # Correlation analysis
    model_arr = np.array(model_adoption_means)
    btos_arr = np.array(btos_rates)

    # Spearman rank correlation
    from scipy.stats import spearmanr, pearsonr
    spearman_rho, spearman_p = spearmanr(model_arr, btos_arr)
    pearson_r, pearson_p = pearsonr(model_arr, btos_arr)

    # Bootstrap CI over sectors (resample the n sector pairs; fixed rng).
    # Audit 2026-10-07: sectors sharing the same markup map to the SAME sigma
    # and (with the shared seed rule) produce bit-identical model predictions,
    # so the 15 sector pairs carry fewer independent model values. We now
    # (a) record n_eff / the duplicate groups, and (b) bootstrap by resampling
    # DISTINCT-sigma clusters so the CI reflects the actual amount of
    # independent model information.
    sigma_arr = np.array([sigma_map[s] for s in sectors])
    uniq_sigmas = np.unique(np.round(sigma_arr, 6))
    cluster_of = {s: int(np.where(uniq_sigmas == round(sigma_map[s], 6))[0][0])
                  for s in sectors}
    dup_groups = {}
    for s in sectors:
        dup_groups.setdefault(cluster_of[s], []).append(s)
    n_eff = len(uniq_sigmas)

    rng = np.random.default_rng(2026)
    n_sec = len(model_arr)
    boot_rho = []
    for _ in range(10000):
        idx = rng.integers(0, n_sec, n_sec)
        if np.std(model_arr[idx]) < 1e-9 or np.std(btos_arr[idx]) < 1e-9:
            continue
        boot_rho.append(spearmanr(model_arr[idx], btos_arr[idx])[0])
    boot_rho = np.array(boot_rho)
    rho_ci = [float(np.percentile(boot_rho, 2.5)), float(np.percentile(boot_rho, 97.5))]
    print(f'  Bootstrap 95% CI for rho (10k resamples): [{rho_ci[0]:.3f}, {rho_ci[1]:.3f}]')

    # cluster bootstrap: resample distinct-sigma clusters (kept sectors move
    # together with their duplicates)
    cluster_ids = [cluster_of[s] for s in sectors]
    boot_rho_c = []
    for _ in range(10000):
        pick = rng.integers(0, n_eff, n_eff)
        keep = [i for i in range(n_sec) if cluster_ids[i] in pick]
        if len(keep) < 4:
            continue
        if np.std(model_arr[keep]) < 1e-9 or np.std(btos_arr[keep]) < 1e-9:
            continue
        boot_rho_c.append(spearmanr(model_arr[keep], btos_arr[keep])[0])
    boot_rho_c = np.array(boot_rho_c)
    rho_ci_cluster = ([float(np.percentile(boot_rho_c, 2.5)),
                       float(np.percentile(boot_rho_c, 97.5))]
                      if boot_rho_c.size else [None, None])
    print(f'  Cluster-bootstrap 95% CI (n_eff={n_eff} distinct sigma): '
          f'[{rho_ci_cluster[0]}, {rho_ci_cluster[1]}]')

    print(f'\n  Correlation: Spearman rho={spearman_rho:.3f} (p={spearman_p:.4f})')
    print(f'               Pearson  r ={pearson_r:.3f} (p={pearson_p:.4f})')
    print(f'  Target: rho > 0.5')

    corr_summary = {
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'spearman_ci95': rho_ci,
        'spearman_ci95_cluster': rho_ci_cluster,
        'n_sectors': int(n_sec),
        'n_effective': int(n_eff),
        'duplicate_sigma_groups': [g for g in dup_groups.values() if len(g) > 1],
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
    }

    # Time-horizon correlation analysis
    print(f'\n  Time-horizon correlation:')
    time_corrs = []
    for t_tick in TRAJECTORY_INTERVALS:
        model_at_t = []
        for sector in sectors:
            d = all_results[sector]
            vals = [s['trajectory'].get(t_tick, 0) for s in d.get('_per_seed', [])]
            model_at_t.append(np.mean(vals) if vals else 0)
        model_at_t = np.array(model_at_t)
        if np.std(model_at_t) < 1e-8:
            print(f'    tick={t_tick:3d} ({t_tick/12:4.1f}yr): rho=n/a (zero variance)')
            continue
        rho_t, p_t = spearmanr(model_at_t, btos_arr)
        time_corrs.append({'tick': t_tick, 'years': t_tick / 12,
                           'spearman_rho': float(rho_t), 'p': float(p_t)})
        print(f'    tick={t_tick:3d} ({t_tick/12:4.1f}yr): rho={rho_t:.3f} (p={p_t:.4f})')

    corr_summary['time_horizon'] = time_corrs

    elapsed = time.time() - t0
    print(f'\nTotal: {elapsed:.0f}s')

    raw_path = os.path.join(SAVE_DIR, 'industry_calibration_results.json')
    save_data = {
        'correlation': corr_summary,
        'sectors': all_results,
        'sigma_map': {k: float(v) for k, v in sigma_map.items()},
    }
    with open(raw_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f'Saved: {raw_path}')

    generate_figures(all_results, corr_summary, sigma_map)
    return all_results, corr_summary


def generate_figures(all_results, corr_summary, sigma_map):
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

    sectors = list(all_results.keys())
    btos_rates = np.array([all_results[s]['btos_rate'] for s in sectors])
    model_rates = np.array([all_results[s]['adoption_rate_mean'] for s in sectors])
    model_stds = np.array([all_results[s]['adoption_rate_std'] for s in sectors])
    markups = np.array([all_results[s]['markup'] for s in sectors])
    sigmas = np.array([all_results[s]['sigma'] for s in sectors])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Model vs BTOS scatter
    ax = axes[0, 0]
    for i, s in enumerate(sectors):
        short = s[:12]
        ax.errorbar(btos_rates[i], model_rates[i] * 100, yerr=model_stds[i] * 100,
                    fmt='o', color='#2196F3', markersize=6, alpha=0.7)
        ax.annotate(short, (btos_rates[i], model_rates[i] * 100),
                   fontsize=5, ha='left', va='bottom', xytext=(2, 2),
                   textcoords='offset points')
    lim_lo = min(btos_rates.min(), (model_rates * 100).min()) - 5
    lim_hi = max(btos_rates.max(), (model_rates * 100).max()) + 5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], '--', color='gray', linewidth=0.5)
    rho = corr_summary['spearman_rho']
    ax.set_xlabel('BTOS empirical AI adoption (%)')
    ax.set_ylabel('Model AI adoption rate (%)')
    ax.set_title(f'(a) Model vs BTOS (Spearman $\\rho$={rho:.2f})')
    ax.grid(True, alpha=0.3)

    # Panel B: Ranked comparison
    ax = axes[0, 1]
    btos_rank = np.argsort(btos_rates)[::-1]
    x = np.arange(len(sectors))
    width = 0.35
    ax.bar(x - width/2, btos_rates[btos_rank], width, label='BTOS empirical',
           color='#9E9E9E', alpha=0.8)
    ax.bar(x + width/2, model_rates[btos_rank] * 100, width, label='Model',
           color='#2196F3', alpha=0.8)
    short_labels = [sectors[j][:10] for j in btos_rank]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('AI adoption (%)')
    ax.set_title('(b) Ranked Comparison (BTOS order)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: AI adoption vs sigma
    ax = axes[1, 0]
    for i, s in enumerate(sectors):
        ax.scatter(sigmas[i], model_rates[i] * 100, color='#2196F3', s=40, alpha=0.7)
    ax.set_xlabel(r'Sector $\sigma$ (substitution elasticity)')
    ax.set_ylabel('Model AI adoption rate (%)')
    ax.set_title(r'(c) AI Adoption vs $\sigma$')
    ax.grid(True, alpha=0.3)

    # Panel D: AI adoption vs markup
    ax = axes[1, 1]
    for i, s in enumerate(sectors):
        ax.scatter(markups[i], model_rates[i] * 100, color='#F44336', s=40, alpha=0.7)
    ax.set_xlabel(r'Sector markup $\mu$')
    ax.set_ylabel('Model AI adoption rate (%)')
    ax.set_title(r'(d) AI Adoption vs Markup $\mu$')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Industry-Level BTOS Calibration', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'F1_industry_calibration.png'))
    print(f'Saved: F1_industry_calibration.png')
    plt.close(fig)

    print(f'\nAll figures saved to: {fig_dir}/')

    # Figure 2: Time-horizon correlation
    if 'time_horizon' in corr_summary:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
        years = [tc['years'] for tc in corr_summary['time_horizon']]
        rhos = [tc['spearman_rho'] for tc in corr_summary['time_horizon']]
        ps = [tc['p'] for tc in corr_summary['time_horizon']]

        ax2.plot(years, rhos, 'o-', color='#2196F3', linewidth=2, markersize=8)
        ax2.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Target rho=0.5')
        ax2.fill_between(years, 0, rhos, alpha=0.1, color='#2196F3')
        ax2.set_xlabel('Years after AI introduction')
        ax2.set_ylabel('Spearman $\\rho$ (model vs BTOS ranking)')
        ax2.set_title('Calibration Quality Over Time')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        for i, (y, r, p) in enumerate(zip(years, rhos, ps)):
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax2.annotate(f'{r:.2f}{sig}', (y, r), fontsize=7,
                        ha='center', va='bottom', xytext=(0, 5),
                        textcoords='offset points')

        fig2.tight_layout()
        fig2.savefig(os.path.join(fig_dir, 'F2_time_horizon_correlation.png'))
        print(f'Saved: F2_time_horizon_correlation.png')
        plt.close(fig2)


if __name__ == '__main__':
    all_results, corr = main()
    print('\nIndustry calibration complete.')
