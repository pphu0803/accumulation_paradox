"""Verify AI adoption S-curve reduces short-run TFP to empirical range.

Compares adoption dynamics and short-run TFP with vs without S-curve gating.
Also sweeps adoption_readiness_speed to find optimal calibration.
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

PARAM_GRID = [
    {'adoption_readiness_speed': 0.0, 'adoption_threshold': 0.0, 'label': 'No S-curve (baseline)'},
    {'adoption_readiness_speed': 0.01, 'adoption_threshold': 0.8, 'label': 'speed=0.01 threshold=0.8'},
    {'adoption_readiness_speed': 0.02, 'adoption_threshold': 0.8, 'label': 'speed=0.02 threshold=0.8'},
    {'adoption_readiness_speed': 0.03, 'adoption_threshold': 0.7, 'label': 'speed=0.03 threshold=0.7'},
    {'adoption_readiness_speed': 0.02, 'adoption_threshold': 0.6, 'label': 'speed=0.02 threshold=0.6'},
    {'adoption_readiness_speed': 0.04, 'adoption_threshold': 0.8, 'label': 'speed=0.04 threshold=0.8'},
]

NUM_SEEDS = 12
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
AI_TICKS = 120  # v3.1 (2026-10-09): AI window unified to 120 ticks (10 yr) across all experiments
SAVE_DIR = os.path.join(_abm_dir, 'output', 's_curve_verification')
os.makedirs(SAVE_DIR, exist_ok=True)


def compute_metrics(firms_pre, firms_post, sc, trajectory_data):
    alpha = float(sc.alpha_CD)

    # Pre-AI TFP
    alive_pre = np.array(firms_pre.alive)
    K = np.maximum(np.array(firms_pre.K), 0.01)
    L = np.maximum(np.array(firms_pre.L), 0.01)
    eff = K**alpha * L**(1-alpha)
    pre_tfp = np.sum(np.array(firms_pre.Q)[alive_pre]) / max(np.sum(eff[alive_pre]), 0.01)

    # Post-AI TFP
    alive = np.array(firms_post.alive)
    K = np.maximum(np.array(firms_post.K), 0.01)
    L = np.maximum(np.array(firms_post.L), 0.01)
    eff = K**alpha * L**(1-alpha)
    post_tfp = np.sum(np.array(firms_post.Q)[alive]) / max(np.sum(eff[alive]), 0.01)

    delta_tfp = (post_tfp - pre_tfp) / pre_tfp
    annualized = (1 + delta_tfp) ** (12 / AI_TICKS) - 1

    # AI adoption
    adoption = np.sum((np.array(firms_post.ai_experience) > 0.1) & alive) / max(np.sum(alive), 1)

    # Readiness
    readiness = np.mean(np.array(firms_post.adoption_readiness))

    return {
        'delta_tfp': delta_tfp,
        'annualized_tfp': annualized,
        'adoption_rate': adoption,
        'readiness_mean': readiness,
        'readiness_trajectory': trajectory_data.get('readiness', []),
        'adoption_trajectory': trajectory_data.get('adoption', []),
    }


def main():
    t0 = time.time()
    all_results = {}

    print(f'S-curve verification: {len(PARAM_GRID)} param sets × {NUM_SEEDS} seeds')
    print(f'  {PRE_TICKS} pre-AI + {AI_TICKS} post-AI ticks')
    print()

    for idx, params in enumerate(PARAM_GRID):
        label = params.pop('label')
        overrides = dict(params)
        config = merge_configs(overrides)
        sc = _make_static_config(config)

        per_seed = []
        traj_readiness = []
        traj_adoption = []

        for seed_idx in range(NUM_SEEDS):
            seed_key = seed_idx * 1000 + 42
            state = init_simulation(config, seed=seed_key)
            pre = run_n_ticks(state, sc, PRE_TICKS)

            # Collect trajectory
            readiness_traj = []
            adoption_traj = []
            current = pre
            for t in range(AI_TICKS):
                current = run_n_ticks(current, sc, 1)
                alive_np = np.array(current.firms.alive)
                readiness_traj.append(float(jnp.mean(current.firms.adoption_readiness)))
                adoption_traj.append(float(jnp.sum(
                    (current.firms.ai_experience > 0.1) & current.firms.alive)) /
                    max(float(jnp.sum(current.firms.alive)), 1))

            traj_data = {'readiness': readiness_traj, 'adoption': adoption_traj}
            metrics = compute_metrics(pre.firms, current.firms, sc, traj_data)
            per_seed.append(metrics)
            traj_readiness.append(readiness_traj)
            traj_adoption.append(adoption_traj)

        # Aggregate
        agg = {'label': label}
        for k in ['delta_tfp', 'annualized_tfp', 'adoption_rate', 'readiness_mean']:
            vals = [s[k] for s in per_seed]
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals))

        # Mean trajectories
        r_arr = np.array(traj_readiness)
        a_arr = np.array(traj_adoption)
        agg['readiness_traj_mean'] = np.mean(r_arr, axis=0).tolist()
        agg['readiness_traj_std'] = np.std(r_arr, axis=0).tolist()
        agg['adoption_traj_mean'] = np.mean(a_arr, axis=0).tolist()
        agg['adoption_traj_std'] = np.std(a_arr, axis=0).tolist()

        all_results[label] = agg
        params['label'] = label  # restore

        print(f'  [{idx+1}/{len(PARAM_GRID)}] {label}: '
              f'TFP_ann={agg["annualized_tfp_mean"]*100:+.1f}%±{agg["annualized_tfp_std"]*100:.1f}% '
              f'ΔTFP={agg["delta_tfp_mean"]*100:+.1f}% '
              f'adopt={agg["adoption_rate_mean"]:.1%}',
              flush=True)

    elapsed = time.time() - t0
    print(f'\nTotal: {elapsed:.0f}s')

    raw_path = os.path.join(SAVE_DIR, 's_curve_results.json')
    save_data = {}
    for label, val in all_results.items():
        save_data[label] = {k: v for k, v in val.items()
                            if k not in ('readiness_traj_mean', 'readiness_traj_std',
                                         'adoption_traj_mean', 'adoption_traj_std')}
    with open(raw_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    traj_path = os.path.join(SAVE_DIR, 'trajectories.json')
    traj_data = {}
    for label, val in all_results.items():
        traj_data[label] = {
            'readiness': {'mean': val['readiness_traj_mean'], 'std': val['readiness_traj_std']},
            'adoption': {'mean': val['adoption_traj_mean'], 'std': val['adoption_traj_std']},
        }
    with open(traj_path, 'w') as f:
        json.dump(traj_data, f, indent=2)
    print(f'Saved: {raw_path}, {traj_path}')

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

    labels = list(all_results.keys())
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Adoption S-curves
    ax = axes[0, 0]
    for label in labels:
        a_mean = np.array(all_results[label]['adoption_traj_mean'])
        a_std = np.array(all_results[label]['adoption_traj_std'])
        ticks = np.arange(len(a_mean))
        short_label = label.split('=')[-1].split(',')[0].strip() if '=' in label else label[:10]
        ax.plot(ticks, a_mean * 100, '-', label=short_label, linewidth=1.5)
    ax.axhline(50, color='gray', linestyle=':', linewidth=0.5)
    ax.set_xlabel('Post-AI ticks')
    ax.set_ylabel('AI adoption rate (%)')
    ax.set_title('(a) Adoption S-Curve Dynamics')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel B: Annualized TFP
    ax = axes[0, 1]
    means = [all_results[l]['annualized_tfp_mean'] * 100 for l in labels]
    stds = [all_results[l]['annualized_tfp_std'] * 100 for l in labels]
    ax.bar(x, means, yerr=stds, capsize=3, color='#2196F3', alpha=0.8)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='US empirical ≈ 1%/yr')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l[:15] for l in labels], rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Annualized TFP growth (%)')
    ax.set_title('(b) Short-Run TFP (target: < 2%/yr)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Final adoption rate
    ax = axes[1, 0]
    adopt = [all_results[l]['adoption_rate_mean'] * 100 for l in labels]
    adopt_std = [all_results[l]['adoption_rate_std'] * 100 for l in labels]
    ax.bar(x, adopt, yerr=adopt_std, capsize=3, color='#4CAF50', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l[:15] for l in labels], rotation=30, ha='right', fontsize=7)
    ax.set_ylabel(f'AI adoption rate at tick {AI_TICKS} (%)')
    ax.set_title('(c) Final Adoption Rate')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: Readiness trajectories
    ax = axes[1, 1]
    for label in labels:
        r_mean = np.array(all_results[label]['readiness_traj_mean'])
        ticks = np.arange(len(r_mean))
        short_label = label.split('=')[-1].split(',')[0].strip() if '=' in label else label[:10]
        ax.plot(ticks, r_mean, '-', label=short_label, linewidth=1.5)
    ax.axhline(0.8, color='red', linestyle='--', linewidth=1, label='Threshold (0.8)')
    ax.set_xlabel('Post-AI ticks')
    ax.set_ylabel('Mean adoption readiness')
    ax.set_title('(d) Readiness Accumulation')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.suptitle('AI Adoption S-Curve: Short-Run TFP Calibration', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'F1_s_curve_verification.png'))
    print(f'Saved: F1_s_curve_verification.png')
    plt.close(fig)

    print(f'\nAll figures saved to: {fig_dir}/')


if __name__ == '__main__':
    all_results = main()
    print('\nS-curve verification complete.')
