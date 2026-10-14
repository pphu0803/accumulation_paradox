"""Markup × Gate Decomposition Experiment

Factorial design: gate (ON/OFF) × σ sweep to decompose TFP suppression
into the adoption gate contribution and the markup absorption contribution,
and test for interaction.

Gate ON:  default S-curve params (speed=0.02, peer=0.05, threshold=0.8)
Gate OFF: no gating (speed=0.0, peer=0.0, threshold=0.0)
σ sweep:  [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
gain:     0.04 (fixed)
seeds:    40 per condition
"""

import sys
import os
import json
import time
import gc

import numpy as np

_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_abm_dir, '..'))
for _p in (_project_root, _abm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from abm.src.state import init_simulation
from abm.src.step import _make_static_config, run_n_ticks
from abm.src.config import merge_configs

import jax
import jax.numpy as jnp

SAVE_DIR = os.path.join(_abm_dir, 'output', 'robustness')
os.makedirs(SAVE_DIR, exist_ok=True)

GATE_CONDITIONS = {
    'gate_on': {
        'adoption_readiness_speed': 0.02,
        'adoption_peer_influence': 0.05,
        'adoption_threshold': 0.8,
    },
    'gate_off': {
        'adoption_readiness_speed': 0.0,
        'adoption_peer_influence': 0.0,
        'adoption_threshold': 0.0,
    },
}

SIGMA_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
GAIN = 0.04
NUM_SEEDS = int(os.environ.get('NUM_SEEDS', '40'))
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
POST_TICKS = 120
SIGMA_FIRM_SPREAD = 0.4
SIGMA_FIRM_A_SENS = 0.5


def compute_metrics(pre_firms, post_firms, post_market, alpha,
                    theta=0.15, gain=0.04, j_curve_depth=0.15):
    """Compute TFP, adoption, HHI, markup from pre/post states.

    Audit 2026-10-07: TFP recomputes Q = A_eff * q^theta * K^alpha L^(1-alpha)
    from the state itself instead of reading firms.Q (which is a within-tick
    S3 quantity, mismatched with the end-of-tick K/L and zero for newborn
    entrants — 3.5-6% level bias).
    """
    def tfp_of(firms):
        alive = np.array(firms.alive)
        K = np.maximum(np.array(firms.K), 0.01)
        L = np.maximum(np.array(firms.L), 0.01)
        eff = K ** alpha * L ** (1 - alpha)
        mult = np.where(np.array(firms.ai_experience) > 0,
                        1.0 + gain * ((1.0 + j_curve_depth)
                                      * np.array(firms.ai_experience) - j_curve_depth),
                        1.0)
        a_eff = np.array(firms.A_base) * mult
        q = np.maximum(np.array(firms.product_quality), 0.1)
        Qhat = a_eff * q ** theta * eff
        return (np.sum(Qhat[alive]) / max(np.sum(eff[alive]), 0.01))

    # Pre TFP
    alive_pre = np.array(pre_firms.alive)
    pre_tfp = tfp_of(pre_firms)

    # Post TFP
    alive = np.array(post_firms.alive)
    post_tfp = tfp_of(post_firms)

    delta_tfp = (post_tfp - pre_tfp) / pre_tfp
    annualized = (1 + delta_tfp) ** (12.0 / POST_TICKS) - 1

    # Adoption
    adoption = float(jnp.sum((post_firms.ai_experience > 0.1) & post_firms.alive) /
                     max(float(jnp.sum(post_firms.alive)), 1))

    # HHI
    shares = np.where(alive, np.array(post_firms.market_share), 0.0)
    hhi = float(np.sum(shares ** 2))

    # Mean markup
    markups = np.where(alive, np.array(post_firms.effective_markup), 0.0)
    mean_markup = float(np.mean(markups[alive])) if np.any(alive) else 1.0

    return {
        'delta_tfp': delta_tfp,
        'annualized_tfp': annualized,
        'adoption_rate': adoption,
        'hhi': hhi,
        'mean_markup': mean_markup,
        'pre_tfp': pre_tfp,
        'post_tfp': post_tfp,
    }


def run_condition(gate_label, sigma):
    """Run one (gate, σ) condition across NUM_SEEDS."""
    gate_params = GATE_CONDITIONS[gate_label]
    config = merge_configs({
        'sigma_base': sigma,
        'sigma_firm_spread': SIGMA_FIRM_SPREAD,
        'sigma_firm_A_sensitivity': SIGMA_FIRM_A_SENS,
        'ai_efficiency_gain': GAIN,
        **gate_params,
    })
    sc = _make_static_config(config)
    alpha = float(sc.alpha_CD)

    results = []
    for seed_idx in range(NUM_SEEDS):
        seed_key = seed_idx * 1000 + 42
        state = init_simulation(config, seed=seed_key)
        pre = run_n_ticks(state, sc, PRE_TICKS)
        post = run_n_ticks(pre, sc, POST_TICKS)

        metrics = compute_metrics(
            pre.firms, post.firms, post.market, alpha,
            theta=float(sc.quality_weight), gain=float(sc.ai_efficiency_gain),
            j_curve_depth=float(sc.j_curve_depth))
        metrics['seed'] = seed_idx
        results.append(metrics)

    return results


def compute_decomposition(data):
    """Compute gate effect, markup effect, and interaction."""
    decomposition = {}

    for sigma in SIGMA_VALUES:
        on_seeds = [d for d in data if d['sigma'] == sigma and d['gate'] == 'gate_on']
        off_seeds = [d for d in data if d['sigma'] == sigma and d['gate'] == 'gate_off']

        if not on_seeds or not off_seeds:
            continue

        on_tfp = np.mean([s['delta_tfp'] for s in on_seeds])
        off_tfp = np.mean([s['delta_tfp'] for s in off_seeds])
        on_ann = np.mean([s['annualized_tfp'] for s in on_seeds])
        off_ann = np.mean([s['annualized_tfp'] for s in off_seeds])
        on_adopt = np.mean([s['adoption_rate'] for s in on_seeds])
        off_adopt = np.mean([s['adoption_rate'] for s in off_seeds])
        on_hhi = np.mean([s['hhi'] for s in on_seeds])
        off_hhi = np.mean([s['hhi'] for s in off_seeds])
        on_markup = np.mean([s['mean_markup'] for s in on_seeds])
        off_markup = np.mean([s['mean_markup'] for s in off_seeds])

        # Audit 2026-10-07: paired SE for the gate effect. Arms share the seed
        # rule (seed_idx*1000+42), so per-seed differences are the right unit;
        # the old *_std fields are LEVEL stds and say nothing about the effect.
        on_by_seed = {s['seed']: s['delta_tfp'] for s in on_seeds}
        off_by_seed = {s['seed']: s['delta_tfp'] for s in off_seeds}
        common = sorted(set(on_by_seed) & set(off_by_seed))
        if len(common) >= 2:
            diffs = np.array([off_by_seed[s] - on_by_seed[s] for s in common])
            gate_se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
            gate_ci = [float(diffs.mean() - 1.96 * gate_se),
                       float(diffs.mean() + 1.96 * gate_se)]
        else:
            gate_se, gate_ci = None, None

        decomposition[f'sigma={sigma}'] = {
            'sigma': sigma,
            'tfp_gate_on': on_tfp,
            'tfp_gate_off': off_tfp,
            'tfp_gate_on_std': float(np.std([s['delta_tfp'] for s in on_seeds])),
            'tfp_gate_off_std': float(np.std([s['delta_tfp'] for s in off_seeds])),
            'ann_gate_on': on_ann,
            'ann_gate_off': off_ann,
            'adopt_gate_on': on_adopt,
            'adopt_gate_off': off_adopt,
            'hhi_gate_on': on_hhi,
            'hhi_gate_off': off_hhi,
            'markup_gate_on': on_markup,
            'markup_gate_off': off_markup,
            'gate_effect_tfp': off_tfp - on_tfp,      # TFP gain from removing gate
            'gate_effect_ann': off_ann - on_ann,       # annualized gain
            'gate_effect_adopt': off_adopt - on_adopt,  # adoption gain
            'gate_effect_tfp_paired_se': gate_se,
            'gate_effect_tfp_paired_ci95': gate_ci,
        }

    # Interaction: does gate effect vary with sigma?
    sigmas_list = sorted(SIGMA_VALUES)
    if len(sigmas_list) >= 2:
        sigma_low = sigmas_list[0]  # 2.0
        sigma_high = sigmas_list[-1]  # 10.0
        d_low = decomposition.get(f'sigma={sigma_low}', {})
        d_high = decomposition.get(f'sigma={sigma_high}', {})
        if d_low and d_high:
            gate_effect_low = d_low['gate_effect_tfp']
            gate_effect_high = d_high['gate_effect_tfp']
            markup_effect_on = d_high['tfp_gate_on'] - d_low['tfp_gate_on']
            markup_effect_off = d_high['tfp_gate_off'] - d_low['tfp_gate_off']
            interaction = gate_effect_low - gate_effect_high

            summary = {
                'gate_effect_range': [d['gate_effect_tfp'] for d in decomposition.values()],
                'gate_effect_mean': float(np.mean([d['gate_effect_tfp'] for d in decomposition.values()])),
                'gate_effect_std': float(np.std([d['gate_effect_tfp'] for d in decomposition.values()])),
                'markup_effect_gate_on': markup_effect_on,
                'markup_effect_gate_off': markup_effect_off,
                'interaction': interaction,
                'sigma_low': sigma_low,
                'sigma_high': sigma_high,
                'gate_effect_at_sigma_low': gate_effect_low,
                'gate_effect_at_sigma_high': gate_effect_high,
            }
            decomposition['summary'] = summary

    return decomposition


def generate_figure(decomposition):
    """Generate 3-panel decomposition figure."""
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

    sigmas = sorted([decomposition[k]['sigma'] for k in decomposition if k.startswith('sigma=')])
    tfp_on = [decomposition[f'sigma={s}']['tfp_gate_on'] * 100 for s in sigmas]
    tfp_off = [decomposition[f'sigma={s}']['tfp_gate_off'] * 100 for s in sigmas]
    tfp_on_std = [decomposition[f'sigma={s}']['tfp_gate_on_std'] * 100 for s in sigmas]
    tfp_off_std = [decomposition[f'sigma={s}']['tfp_gate_off_std'] * 100 for s in sigmas]
    gate_effects = [decomposition[f'sigma={s}']['gate_effect_tfp'] * 100 for s in sigmas]
    adopt_on = [decomposition[f'sigma={s}']['adopt_gate_on'] * 100 for s in sigmas]
    adopt_off = [decomposition[f'sigma={s}']['adopt_gate_off'] * 100 for s in sigmas]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): ΔTFP vs σ
    ax = axes[0]
    ax.fill_between(sigmas, tfp_on, tfp_off, alpha=0.15, color='#F44336', label='Gate effect')
    ax.plot(sigmas, tfp_off, 'o-', color='#2196F3', linewidth=2, markersize=6, label='Gate OFF')
    ax.plot(sigmas, tfp_on, 's-', color='#F44336', linewidth=2, markersize=6, label='Gate ON (default)')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
    ax.set_xlabel('Substitution elasticity $\\sigma$')
    ax.set_ylabel('$\\Delta$TFP (\%)')
    ax.set_title('(a) TFP change by competition and gate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (b): Gate effect vs σ
    ax = axes[1]
    ax.bar(sigmas, gate_effects, width=0.6, color='#FF9800', alpha=0.8, edgecolor='#E65100')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
    ax.set_xlabel('Substitution elasticity $\\sigma$')
    ax.set_ylabel('Gate effect on $\\Delta$TFP (pp)')
    ax.set_title('(b) Gate suppression by competition level')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel (c): Adoption rate vs σ
    ax = axes[2]
    ax.plot(sigmas, adopt_off, 'o-', color='#2196F3', linewidth=2, markersize=6, label='Gate OFF')
    ax.plot(sigmas, adopt_on, 's-', color='#F44336', linewidth=2, markersize=6, label='Gate ON (default)')
    ax.set_xlabel('Substitution elasticity $\\sigma$')
    ax.set_ylabel('AI adoption rate (\%)')
    ax.set_title('(c) Adoption by competition and gate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'fig_decomposition.png')
    fig.savefig(fig_path)
    plt.close(fig)
    print(f'Saved: {fig_path}')


def main():
    t0 = time.time()

    n_conditions = len(GATE_CONDITIONS) * len(SIGMA_VALUES)
    total_runs = n_conditions * NUM_SEEDS

    print('=' * 60)
    print('MARKUP × GATE DECOMPOSITION')
    print(f'  {len(GATE_CONDITIONS)} gate × {len(SIGMA_VALUES)} σ = {n_conditions} conditions')
    print(f'  {NUM_SEEDS} seeds × ({PRE_TICKS} + {POST_TICKS} ticks)')
    print(f'  Total: {total_runs} runs')
    print('=' * 60)

    checkpoint_path = os.path.join(SAVE_DIR, 'decomposition_checkpoint.json')
    all_data = []

    # Config stamp (audit 2026-10-07): a checkpoint from a run with different
    # gain/PRE/POST/NUM_SEEDS used to be silently mixed into the new batch.
    config_stamp = {'gain': GAIN, 'pre_ticks': PRE_TICKS,
                    'post_ticks': POST_TICKS, 'num_seeds': NUM_SEEDS,
                    'sigmas': list(SIGMA_VALUES)}
    stamp_path = checkpoint_path + '.stamp'

    # Restore checkpoint
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            all_data = json.load(f)
        old_stamp = None
        if os.path.exists(stamp_path):
            with open(stamp_path) as f:
                old_stamp = json.load(f)
        if old_stamp is not None and old_stamp != config_stamp:
            print(f'WARNING: checkpoint stamp {old_stamp} != current config '
                  f'{config_stamp}; discarding checkpoint to avoid mixing batches')
            all_data = []
        else:
            # Archived entries lack 'seed'; reconstruct it as the position within
            # each (gate, sigma) group (runs were appended in seed order).
            from collections import Counter
            counts = Counter()
            for d in all_data:
                if 'seed' not in d:
                    d['seed'] = counts[(d['gate'], d['sigma'])]
                    counts[(d['gate'], d['sigma'])] += 1
            print(f'Restored {len(all_data)} completed runs from checkpoint')

    completed = set((d['gate'], d['sigma'], d['seed']) for d in all_data)

    for gate_label in GATE_CONDITIONS:
        for sigma in SIGMA_VALUES:
            remaining = [s for s in range(NUM_SEEDS)
                          if (gate_label, sigma, s) not in completed]

            if not remaining:
                print(f'  [{gate_label}, σ={sigma}] already complete')
                continue

            print(f'\n  {gate_label}, σ={sigma}: {len(remaining)} seeds remaining')
            t_cond = time.time()

            for seed_idx in remaining:
                gate_params = GATE_CONDITIONS[gate_label]
                config = merge_configs({
                    'sigma_base': sigma,
                    'sigma_firm_spread': SIGMA_FIRM_SPREAD,
                    'sigma_firm_A_sensitivity': SIGMA_FIRM_A_SENS,
                    'ai_efficiency_gain': GAIN,
                    **gate_params,
                })
                sc = _make_static_config(config)
                alpha = float(sc.alpha_CD)

                seed_key = seed_idx * 1000 + 42
                state = init_simulation(config, seed=seed_key)
                pre = run_n_ticks(state, sc, PRE_TICKS)
                post = run_n_ticks(pre, sc, POST_TICKS)

                metrics = compute_metrics(pre.firms, post.firms, post.market, alpha)
                metrics['gate'] = gate_label
                metrics['sigma'] = sigma
                metrics['seed'] = seed_idx
                all_data.append(metrics)

            # Checkpoint after each (gate, σ) pair
            with open(checkpoint_path, 'w') as f:
                json.dump(all_data, f, indent=2, default=float)
            with open(stamp_path, 'w') as f:
                json.dump(config_stamp, f)

            jax.clear_caches()
            gc.collect()

            elapsed = time.time() - t_cond
            print(f'    Done in {elapsed:.0f}s')

            # Print running summary for this condition
            cond_data = [d for d in all_data if d['gate'] == gate_label and d['sigma'] == sigma]
            mean_tfp = np.mean([d['delta_tfp'] for d in cond_data]) * 100
            mean_ann = np.mean([d['annualized_tfp'] for d in cond_data]) * 100
            mean_adopt = np.mean([d['adoption_rate'] for d in cond_data]) * 100
            print(f'    ΔTFP={mean_tfp:.1f}%, annual={mean_ann:.1f}%, adopt={mean_adopt:.1f}%')

    # Save raw data
    raw_path = os.path.join(SAVE_DIR, 'decomposition.json')
    with open(raw_path, 'w') as f:
        json.dump(all_data, f, indent=2, default=float)
    print(f'\nSaved: {raw_path}')

    # Compute decomposition
    print('\n' + '=' * 60)
    print('DECOMPOSITION RESULTS')
    decomposition = compute_decomposition(all_data)

    for key in sorted(decomposition.keys()):
        if key == 'summary':
            continue
        d = decomposition[key]
        print(f'  σ={d["sigma"]:5.1f}: '
              f'Gate ON ΔTFP={d["tfp_gate_on"]*100:+6.1f}% (ann={d["ann_gate_on"]*100:+.1f}%), '
              f'Gate OFF ΔTFP={d["tfp_gate_off"]*100:+6.1f}% (ann={d["ann_gate_off"]*100:+.1f}%), '
              f'gate effect={d["gate_effect_tfp"]*100:+.1f}pp, '
              f'adopt ON={d["adopt_gate_on"]*100:.0f}% OFF={d["adopt_gate_off"]*100:.0f}%')

    if 'summary' in decomposition:
        s = decomposition['summary']
        print(f'\n  SUMMARY:')
        print(f'    Mean gate effect on ΔTFP: {s["gate_effect_mean"]*100:.1f}pp ± {s["gate_effect_std"]*100:.1f}pp')
        print(f'    Markup effect (gate ON):  {s["markup_effect_gate_on"]*100:.1f}pp (σ={s["sigma_low"]}→{s["sigma_high"]})')
        print(f'    Markup effect (gate OFF): {s["markup_effect_gate_off"]*100:.1f}pp (σ={s["sigma_low"]}→{s["sigma_high"]})')
        print(f'    Interaction:             {s["interaction"]*100:.1f}pp')

    # Save decomposition
    summary_path = os.path.join(SAVE_DIR, 'decomposition_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(decomposition, f, indent=2, default=float)
    print(f'\nSaved: {summary_path}')

    # Generate figure
    generate_figure(decomposition)

    total_time = time.time() - t0
    print(f'\nTotal elapsed: {total_time:.0f}s ({total_time/60:.1f}min)')


if __name__ == '__main__':
    main()
