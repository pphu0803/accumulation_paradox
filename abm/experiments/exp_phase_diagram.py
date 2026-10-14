"""Phase diagram experiment for ABM v2.0 — dynamical analysis.

Sweeps σ × ai_gain parameter grid, runs multiple seeds, extracts
time series of key indicators (TFP, HHI, labor share, demand shock),
applies nonlinear dynamics tools to classify each (σ, gain) point
into dynamical phases, and generates a phase diagram.

Dynamics tools from dynamical-observability (local copy):
  - Lyapunov exponent (chaos vs convergence)
  - 0-1 chaos test (Gottwald-Melbourne)
  - DFA exponent (long-range correlation)
  - Sample entropy (complexity)
  - Phase transition detection (abrupt shifts)
  - RQA metrics (determinism)

Phase classification:
  Phase I   — Equilibrium convergence (LLE < 0, DFA α < 0.7, K ≈ 0)
  Phase II  — Periodic/quasi-periodic oscillation (LLE ≈ 0, DFA α ≈ 1, high RQA DET)
  Phase III — Chaotic/complex (LLE > 0, DFA α > 1, K ≈ 1)
  Phase IV  — Paradox phase (micro TFP↑ but macro TFP ≈ 0, within any dynamical regime)
"""

import sys
import os
import json
import time
import math

import numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _abm_dir not in sys.path:
    sys.path.insert(0, _abm_dir)

from abm.src.state import init_simulation
from abm.src.step import _make_static_config, step_core, run_n_ticks, run_n_ticks_collect
from abm.src.config import merge_configs
from abm.src.metrics import compute_metrics
from abm.src.dynamics import (
    largest_lyapunov_exponent, chaos_01_test, dfa_exponent,
    sample_entropy, detect_phase_transitions, rqa_metrics,
    mutual_information, lag1_autocorr,
)

import jax
import jax.numpy as jnp

# ============================================================
# Configuration
# ============================================================

SIGMA_VALS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
GAIN_VALS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
NUM_SEEDS = int(os.environ.get('NUM_SEEDS', '8'))
NUM_TICKS = 370  # v3.1 (2026-10-09): AI window unified to 120 ticks (10 yr; was 50) across all experiments
PRE_TICKS = 250  # steady-state pre-period (audit 2026-10-07 S2)
SAMPLE_EVERY = 2  # sample every 2 ticks -> 60 points over the 120-tick AI window (v3.1)

SAVE_DIR = os.path.join(_abm_dir, 'output', 'phase_diagram')
os.makedirs(SAVE_DIR, exist_ok=True)

KEY_INDICATORS = [
    'macro_tfp', 'hhi', 'labor_share', 'demand_shock',
    'ai_adoption_rate',
    # 'avg_markup' removed 2026-10-07: the batch metric never computed it, so
    # analyze_trajectory silently skipped it.
]


# ============================================================
# Time-series collection
# ============================================================

def _lightweight_metrics_batch(sampled_firms, sampled_market, sc, pre_ai_snapshot=None):
    """Extract key indicators from BATCHED states (vectorized over time).

    Args:
        sampled_firms: FirmState with leaves of shape (T, N_FIRMS)
        sampled_market: MarketState with leaves of shape (T,)
        sc: static config
        pre_ai_snapshot: dict with baseline values

    Returns:
        dict of indicator_name -> np.ndarray of shape (T,)
    """
    T = sampled_firms.alive.shape[0]
    alpha = float(sc.alpha_CD)

    alive = sampled_firms.alive  # (T, N)
    n_alive = jnp.maximum(jnp.sum(alive.astype(float), axis=-1), 1.0)  # (T,)

    K_safe = jnp.maximum(sampled_firms.K, 0.01)
    L_safe = jnp.maximum(sampled_firms.L, 0.01)
    eff = jnp.power(K_safe, alpha) * jnp.power(L_safe, 1.0 - alpha)  # (T, N)

    # Audit 2026-10-07: recompute Q from the same (sampled-state) K/L instead
    # of reading firms.Q (within-tick S3 vintage, entrants carry Q=0).
    gain = float(sc.ai_efficiency_gain)
    depth = float(sc.j_curve_depth)
    theta = float(sc.quality_weight)
    x = sampled_firms.ai_experience
    mult = jnp.where(
        x > 0, 1.0 + gain * ((1.0 + depth) * x - depth), 1.0)
    a_eff = sampled_firms.A_base * mult
    q_safe = jnp.maximum(sampled_firms.product_quality, 0.1)
    Qhat = a_eff * jnp.power(q_safe, theta) * eff

    alive_float = alive.astype(float)
    total_eff = jnp.sum(eff * alive_float, axis=-1)  # (T,)
    total_Q = jnp.sum(Qhat * alive_float, axis=-1)  # (T,)
    macro_tfp = total_Q / jnp.maximum(total_eff, 0.01)  # (T,)

    # HHI
    shares = jnp.where(alive, sampled_firms.market_share, 0.0)
    hhi = jnp.sum(shares ** 2, axis=-1)  # (T,)

    # Labor share
    total_revenue = jnp.sum(sampled_firms.revenue * alive_float, axis=-1)
    total_L = jnp.sum(sampled_firms.L * alive_float, axis=-1)
    labor_share = sampled_market.wage_rate * total_L / jnp.maximum(total_revenue, 0.01)

    demand_shock = sampled_market.demand_externalit_shock  # (T,)
    ai_adoption = sampled_market.ai_adoption_rate  # (T,)

    # Micro TFP (AI vs no-AI)
    ai_mask = alive & (sampled_firms.ai_experience > 0.1)
    no_ai_mask = alive & (sampled_firms.ai_experience <= 0.1)
    n_ai = jnp.maximum(jnp.sum(ai_mask.astype(float), axis=-1), 1.0)
    n_no_ai = jnp.maximum(jnp.sum(no_ai_mask.astype(float), axis=-1), 1.0)

    ai_Q = jnp.sum(Qhat * ai_mask.astype(float), axis=-1)
    ai_eff = jnp.sum(eff * ai_mask.astype(float), axis=-1)
    micro_tfp_ai = ai_Q / jnp.maximum(ai_eff, 0.01)

    no_ai_Q = jnp.sum(Qhat * no_ai_mask.astype(float), axis=-1)
    no_ai_eff = jnp.sum(eff * no_ai_mask.astype(float), axis=-1)
    micro_tfp_no_ai = no_ai_Q / jnp.maximum(no_ai_eff, 0.01)

    result = {
        'macro_tfp': np.array(macro_tfp),
        'micro_tfp_ai': np.array(micro_tfp_ai),
        'micro_tfp_no_ai': np.array(micro_tfp_no_ai),
        'hhi': np.array(hhi),
        'labor_share': np.array(labor_share),
        'demand_shock': np.array(demand_shock),
        'ai_adoption_rate': np.array(ai_adoption),
    }

    if pre_ai_snapshot is not None:
        pre_tfp = pre_ai_snapshot.get('macro_tfp', float(macro_tfp[0]))
        delta_tfp = (macro_tfp - pre_tfp) / max(pre_tfp, 0.01)
        result['delta_macro_tfp'] = np.array(delta_tfp)
        micro_gain = micro_tfp_ai - micro_tfp_no_ai
        result['paradox_score'] = (micro_gain > 0.01) & (jnp.abs(delta_tfp) < 0.05)
    else:
        result['delta_macro_tfp'] = np.zeros(T)
        result['paradox_score'] = np.zeros(T, dtype=bool)

    return result


def _lightweight_metrics(firms, market, sc, pre_ai_snapshot=None):
    """Extract only the 6 key indicators — single state (non-batched)."""
    sampled_f = jax.tree_util.tree_map(lambda a: a[jnp.newaxis], firms)
    sampled_m = jax.tree_util.tree_map(lambda a: a[jnp.newaxis], market)
    batch = _lightweight_metrics_batch(sampled_f, sampled_m, sc, pre_ai_snapshot)
    result = {k: float(v[0]) if not isinstance(v, bool) else bool(v[0]) for k, v in batch.items()}
    return result


def run_with_trajectory(config, seed, num_ticks, sample_every=2):
    """Run simulation in blocks, collect metrics at each block boundary.

    Uses run_n_ticks for each block (JIT-compiled, reuses cache).
    Blocks of `sample_every` ticks keeps memory low while giving enough
    time series length for DFA (~60 points with sample_every=2, v3.1 window).
    """
    sc = _make_static_config(config)
    state = init_simulation(config, seed=seed)

    # Phase 1: Pre-AI baseline (one JIT call)
    pre_state = run_n_ticks(state, sc, PRE_TICKS)
    pre_ai_snapshot = _lightweight_metrics(pre_state.firms, pre_state.market, sc)

    # Phase 2: AI period — run in blocks of sample_every
    n_ai = num_ticks - PRE_TICKS
    n_blocks = n_ai // sample_every
    firms_list = []
    market_list = []

    current = pre_state
    for s in range(n_blocks):
        current = run_n_ticks(current, sc, sample_every)
        firms_list.append(current.firms)
        market_list.append(current.market)

    # Batch extract metrics from all collected states
    # Stack into batched arrays
    n = len(firms_list)
    def _stack(field_name, source='firms'):
        if source == 'firms':
            return jnp.stack([getattr(f, field_name) for f in firms_list], axis=0)
        else:
            return jnp.stack([getattr(m, field_name) for m in market_list], axis=0)

    class _BatchFirms:
        pass
    class _BatchMarket:
        pass

    bf = _BatchFirms()
    bm = _BatchMarket()
    # A_base / product_quality needed by the vintage-consistent TFP recompute
    # in _lightweight_metrics_batch (fix for the 2026-10-08 rc=1 crash).
    for attr in ['alive', 'K', 'L', 'Q', 'revenue', 'market_share',
                 'ai_experience', 'effective_markup', 'A_base',
                 'product_quality']:
        setattr(bf, attr, _stack(attr, 'firms'))

    for attr in ['wage_rate', 'demand_externalit_shock', 'ai_adoption_rate']:
        setattr(bm, attr, _stack(attr, 'market'))

    batch_metrics = _lightweight_metrics_batch(bf, bm, sc, pre_ai_snapshot)

    # Convert to list of dicts
    T = batch_metrics['macro_tfp'].shape[0]
    trajectory = []
    for idx in range(T):
        m = {k: float(v[idx]) if not isinstance(v, (bool, np.bool_)) else bool(v[idx])
             for k, v in batch_metrics.items()}
        m['tick'] = PRE_TICKS + (idx + 1) * sample_every
        trajectory.append(m)

    return trajectory, pre_ai_snapshot


# ============================================================
# Dynamics analysis on a single trajectory
# ============================================================

def analyze_dynamics(series, label='unknown'):
    """Apply all dynamics tools to a 1D time series."""
    series = np.array(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 30:
        return None

    result = {}

    lle = largest_lyapunov_exponent(series)
    result['lyapunov'] = lle if lle is not None else float('nan')

    k01 = chaos_01_test(series)
    result['chaos_K'] = k01 if k01 is not None else float('nan')

    dfa = dfa_exponent(series)
    result['dfa_alpha'] = dfa if dfa is not None else float('nan')

    se = sample_entropy(series)
    result['sampen'] = se if se is not None else float('nan')

    ac = lag1_autocorr(series)
    result['autocorr'] = ac if ac is not None else float('nan')

    transitions = detect_phase_transitions(series, window=15, threshold_sigma=2.0)
    result['n_transitions'] = len(transitions)

    result['std'] = float(np.std(series))
    result['mean'] = float(np.mean(series))

    return result


def analyze_trajectory(trajectory, indicators=KEY_INDICATORS):
    """Analyze dynamics for all key indicators in a trajectory."""
    result = {}
    for ind in indicators:
        series = [m.get(ind, 0) for m in trajectory if ind in m]
        if series:
            dyn = analyze_dynamics(series, label=ind)
            if dyn:
                result[ind] = dyn
    return result


# ============================================================
# Phase classification
# ============================================================

def classify_phase(all_dyn, macro_mean, micro_gain):
    """Classify dynamical phase based on TFP dynamics + paradox check.

    Args:
        all_dyn: full output from analyze_trajectory() (dict of indicators -> dynamics)
        macro_mean: delta_macro_tfp
        micro_gain: micro_tfp_ai - micro_tfp_no_ai

    Returns: 'equilibrium', 'oscillation', 'chaotic', 'paradox_equilibrium',
             'paradox_oscillation', 'paradox_chaotic'
    """
    d = all_dyn.get('macro_tfp', {})
    lle = d.get('lyapunov', float('nan'))
    dfa = d.get('dfa_alpha', float('nan'))
    k01 = d.get('chaos_K', float('nan'))
    std = d.get('std', 0)
    mean_abs = abs(d.get('mean', 0))

    # Relative variability
    cv = std / max(mean_abs, 1e-6)

    is_paradox = (micro_gain > 0.01) and (abs(macro_mean) < 0.05)

    # Dynamics classification
    if math.isnan(lle) or math.isnan(dfa):
        dyn_phase = 'unknown'
    elif lle > 0.005:
        dyn_phase = 'chaotic'
    elif lle < -0.005 and cv < 0.05:
        dyn_phase = 'equilibrium'
    elif dfa > 0.8 and cv > 0.03:
        dyn_phase = 'oscillation'
    elif cv > 0.1:
        dyn_phase = 'oscillation'
    else:
        dyn_phase = 'equilibrium'

    if is_paradox:
        return f'paradox_{dyn_phase}'
    return dyn_phase


# ============================================================
# Main sweep
# ============================================================

def main():
    t0 = time.time()
    all_results = {}
    total = len(SIGMA_VALS) * len(GAIN_VALS)
    count = 0

    print(f'Phase diagram sweep: {len(SIGMA_VALS)} σ × {len(GAIN_VALS)} gain = {total} points')
    print(f'  {NUM_SEEDS} seeds × {NUM_TICKS} ticks each')
    print(f'  Save dir: {SAVE_DIR}')
    print()

    for sigma in SIGMA_VALS:
        for gain in GAIN_VALS:
            count += 1
            key = f'sigma={sigma}_gain={gain}'
            # 'sigma' (consumer-level) is INERT in the dynamics: per-firm
            # sigma_i is initialized from 'sigma_base' (state.py:141) and
            # sc.sigma is never read. Sweeping 'sigma' made every cell
            # bit-identical (verified 2026-09-01); sweep sigma_base instead.
            config = merge_configs({'sigma_base': sigma, 'ai_efficiency_gain': gain})

            per_seed_dyn = []
            per_seed_phases = []
            per_seed_macro = []

            for seed in range(NUM_SEEDS):
                seed_key = seed * 1000 + 42

                # Run with trajectory (post-AI period only)
                traj, pre_snap = run_with_trajectory(
                    config, seed_key, NUM_TICKS, SAMPLE_EVERY)

                # Final metrics
                final_m = traj[-1] if traj else {}
                micro_ai = final_m.get('micro_tfp_ai', 0)
                micro_no = final_m.get('micro_tfp_no_ai', 0)
                micro_gain = micro_ai - micro_no
                macro_delta = final_m.get('delta_macro_tfp', 0)
                hhi = final_m.get('hhi', 0)
                ls = final_m.get('labor_share', 0)
                paradox = final_m.get('paradox_score', False)
                shock = final_m.get('demand_shock', 0)

                per_seed_macro.append({
                    'macro_tfp_delta': macro_delta,
                    'micro_gain': micro_gain,
                    'hhi': hhi,
                    'labor_share': ls,
                    'demand_shock': shock,
                    # 'markup_gradient' removed 2026-10-07: the referenced keys
                    # were never computed, so the column was identically 0.
                    'paradox': bool(paradox),
                })

                # Dynamics analysis
                dyn = analyze_trajectory(traj)
                per_seed_dyn.append(dyn)

                # Phase classification
                phase = classify_phase(dyn, macro_delta, micro_gain)
                per_seed_phases.append(phase)

            # Aggregate across seeds
            agg = aggregate_seeds(per_seed_macro, per_seed_dyn, per_seed_phases)

            # Phase voting (most common phase)
            from collections import Counter
            phase_counts = Counter(per_seed_phases)
            agg['dominant_phase'] = phase_counts.most_common(1)[0][0]
            agg['phase_distribution'] = dict(phase_counts)

            all_results[key] = agg

            # Compact print
            phase_short = agg['dominant_phase'][:5]
            lle_s = agg.get('lyapunov_tfp_mean', float('nan'))
            dfa_s = agg.get('dfa_alpha_tfp_mean', float('nan'))
            print(f'  [{count}/{total}] σ={sigma:4.1f} gain={gain:.2f} | '
                  f'TFP={agg["macro_tfp_delta_mean"]:+.3f} '
                  f'HHI={agg["hhi_mean"]:.4f}±{agg["hhi_std"]:.4f} '
                  f'paradox={agg["paradox_rate"]:.0%} '
                  f'LLE={lle_s:+.4f} '
                  f'DFA={dfa_s:.3f} '
                  f'phase={phase_short}',
                  flush=True)

    elapsed = time.time() - t0
    print(f'\nTotal: {total} points in {elapsed:.0f}s ({elapsed/total:.1f}s/point)')

    # Save raw results
    raw_path = os.path.join(SAVE_DIR, 'phase_diagram_raw.json')
    with open(raw_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    print(f'Saved: {raw_path}')

    # Save per-seed trajectories for visualization
    traj_path = os.path.join(SAVE_DIR, 'trajectories.json')
    # Only save a subset (first 3 seeds per point) to keep file manageable
    traj_subset = {}
    for key, agg in all_results.items():
        traj_subset[key] = {
            'dominant_phase': agg['dominant_phase'],
            'phase_distribution': agg['phase_distribution'],
        }
    with open(traj_path, 'w') as f:
        json.dump(traj_subset, f, indent=2)
    print(f'Saved: {traj_path}')

    return all_results


def aggregate_seeds(per_seed_macro, per_seed_dyn, per_seed_phases):
    """Aggregate per-seed results into mean ± std."""
    n = len(per_seed_macro)
    if n == 0:
        return {}

    agg = {}

    # Macro metrics
    for k in ['macro_tfp_delta', 'micro_gain', 'hhi', 'labor_share',
              'demand_shock']:
        vals = [s[k] for s in per_seed_macro]
        agg[f'{k}_mean'] = float(np.mean(vals))
        agg[f'{k}_std'] = float(np.std(vals))

    # Paradox rate
    agg['paradox_rate'] = sum(s['paradox'] for s in per_seed_macro) / n

    # Dynamics metrics (from TFP analysis)
    dyn_keys = ['lyapunov', 'chaos_K', 'dfa_alpha', 'sampen', 'autocorr',
                'n_transitions', 'std', 'mean']
    for k in dyn_keys:
        vals = [d.get('macro_tfp', {}).get(k, float('nan')) for d in per_seed_dyn]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            agg[f'{k}_tfp_mean'] = float(np.mean(vals))
            agg[f'{k}_tfp_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0
        else:
            agg[f'{k}_tfp_mean'] = 0.0
            agg[f'{k}_tfp_std'] = 0.0

    # HHI dynamics
    for k in ['lyapunov', 'std']:
        vals = [d.get('hhi', {}).get(k, float('nan')) for d in per_seed_dyn]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            agg[f'{k}_hhi_mean'] = float(np.mean(vals))
            agg[f'{k}_hhi_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0

    # Labor share dynamics
    for k in ['lyapunov', 'dfa_alpha']:
        vals = [d.get('labor_share', {}).get(k, float('nan')) for d in per_seed_dyn]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            agg[f'{k}_ls_mean'] = float(np.mean(vals))
            agg[f'{k}_ls_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0

    return agg


# ============================================================
# Visualization
# ============================================================

def generate_figures(all_results):
    """Generate phase diagram and supporting figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    fig_dir = os.path.join(SAVE_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
    })

    n_sigma = len(SIGMA_VALS)
    n_gain = len(GAIN_VALS)

    # Build matrices
    def build_matrix(key, default=0):
        M = np.full((n_sigma, n_gain), default, dtype=float)
        for i, sigma in enumerate(SIGMA_VALS):
            for j, gain in enumerate(GAIN_VALS):
                k = f'sigma={sigma}_gain={gain}'
                if k in all_results:
                    v = all_results[k].get(key, default)
                    if isinstance(v, (int, float)):
                        M[i, j] = v
        return M

    # Phase matrix
    phase_map = {'equilibrium': 0, 'oscillation': 1, 'chaotic': 2,
                 'unknown': 3, 'paradox_equilibrium': 4, 'paradox_oscillation': 5,
                 'paradox_chaotic': 6}
    phase_M = np.full((n_sigma, n_gain), -1, dtype=float)
    for i, sigma in enumerate(SIGMA_VALS):
        for j, gain in enumerate(GAIN_VALS):
            k = f'sigma={sigma}_gain={gain}'
            if k in all_results:
                phase_M[i, j] = phase_map.get(all_results[k]['dominant_phase'], -1)

    # ---- Figure 1: Phase Diagram (main figure) ----
    fig, ax = plt.subplots(figsize=(8, 6))

    # Custom colormap for phases
    phase_colors = ['#2196F3', '#4CAF50', '#F44336', '#9E9E9E',
                    '#1565C0', '#2E7D32', '#B71C1C']
    phase_names = ['Equilibrium', 'Oscillation', 'Chaotic', 'Unknown',
                   'Paradox\n(equil.)', 'Paradox\n(oscil.)', 'Paradox\n(chaotic)']
    cmap = ListedColormap(phase_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.pcolormesh(GAIN_VALS, SIGMA_VALS, phase_M, cmap=cmap, norm=norm,
                       shading='auto')

    # Mark cells with no data
    for i in range(n_sigma):
        for j in range(n_gain):
            if phase_M[i, j] == -1:
                ax.text(GAIN_VALS[j], SIGMA_VALS[i], '×', ha='center', va='center',
                        color='gray', fontsize=10)

    # Contour: macro TFP delta
    tfp_M = build_matrix('macro_tfp_delta_mean', default=0)
    try:
        cs = ax.contour(GAIN_VALS, SIGMA_VALS, tfp_M,
                        levels=[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
                        colors='k', linewidths=0.6, linestyles='--', alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f'ΔTFP={x:.1f}')
    except Exception:
        pass

    # Contour: paradox rate
    paradox_M = build_matrix('paradox_rate', default=0)
    try:
        cs2 = ax.contour(GAIN_VALS, SIGMA_VALS, paradox_M,
                         levels=[0.1, 0.3, 0.5],
                         colors='white', linewidths=1.0, linestyles='-')
        ax.clabel(cs2, inline=True, fontsize=8,
                   fmt=lambda x: f'paradox={x:.0%}')
    except Exception:
        pass

    cbar = fig.colorbar(im, ax=ax, ticks=range(7), shrink=0.8)
    cbar.ax.set_yticklabels(phase_names, fontsize=9)

    ax.set_xlabel(r'AI efficiency gain $\Delta A$')
    ax.set_ylabel(r'Substitution elasticity $\sigma$')
    ax.set_title('Dynamical Phase Diagram: AI Productivity Paradox ABM v2.0')

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'F1_phase_diagram.png'))
    print(f'Saved: F1_phase_diagram.png')
    plt.close(fig)

    # ---- Figure 2: Lyapunov exponent heatmap ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    lle_M = build_matrix('lyapunov_tfp_mean', default=0.0)
    cmap_lle = plt.cm.RdBu_r
    im = ax.pcolormesh(GAIN_VALS, SIGMA_VALS, lle_M, cmap=cmap_lle,
                       vmin=-0.05, vmax=0.05, shading='gouraud')
    cs = ax.contour(GAIN_VALS, SIGMA_VALS, lle_M,
                    levels=[0], colors='k', linewidths=1.5)
    ax.clabel(cs, fmt={0: r'$\lambda=0$'})
    fig.colorbar(im, ax=ax, label=r'Max Lyapunov exponent $\lambda_1$')
    ax.set_xlabel(r'AI efficiency gain $\Delta A$')
    ax.set_ylabel(r'$\sigma$')
    ax.set_title(r'Largest Lyapunov Exponent (macro TFP trajectory)')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'F2_lyapunov_heatmap.png'))
    print(f'Saved: F2_lyapunov_heatmap.png')
    plt.close(fig)

    # ---- Figure 3: DFA exponent heatmap ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    dfa_M = build_matrix('dfa_alpha_tfp_mean', default=float('nan'))
    im = ax.pcolormesh(GAIN_VALS, SIGMA_VALS, dfa_M, cmap='viridis',
                       vmin=0, vmax=2, shading='gouraud')
    cs = ax.contour(GAIN_VALS, SIGMA_VALS, dfa_M,
                    levels=[0.5, 1.0, 1.5],
                    colors='white', linewidths=0.8)
    ax.clabel(cs, fmt=lambda x: f'α={x:.1f}')
    fig.colorbar(im, ax=ax, label='DFA exponent α')
    ax.set_xlabel(r'AI efficiency gain $\Delta A$')
    ax.set_ylabel(r'$\sigma$')
    ax.set_title('Detrended Fluctuation Analysis (macro TFP)')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'F3_dfa_heatmap.png'))
    print(f'Saved: F3_dfa_heatmap.png')
    plt.close(fig)

    # ---- Figure 4: Paradox rate heatmap ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.pcolormesh(GAIN_VALS, SIGMA_VALS, paradox_M,
                       cmap='YlOrRd', vmin=0, vmax=1, shading='gouraud')
    cs = ax.contour(GAIN_VALS, SIGMA_VALS, paradox_M,
                    levels=[0.1, 0.3, 0.5],
                    colors='k', linewidths=0.8)
    ax.clabel(cs, fmt=lambda x: f'{x:.0%}')
    fig.colorbar(im, ax=ax, label='Paradox rate')
    ax.set_xlabel(r'AI efficiency gain $\Delta A$')
    ax.set_ylabel(r'$\sigma$')
    ax.set_title('AI Productivity Paradox Rate')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'F4_paradox_rate.png'))
    print(f'Saved: F4_paradox_rate.png')
    plt.close(fig)

    # ---- Figure 5: HHI path dependence (CV across seeds) ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    hhi_cv_M = np.zeros((n_sigma, n_gain))
    for i, sigma in enumerate(SIGMA_VALS):
        for j, gain in enumerate(GAIN_VALS):
            k = f'sigma={sigma}_gain={gain}'
            if k in all_results:
                m = all_results[k]['hhi_mean']
                s = all_results[k]['hhi_std']
                hhi_cv_M[i, j] = s / max(m, 1e-6)
    im = ax.pcolormesh(GAIN_VALS, SIGMA_VALS, hhi_cv_M,
                       cmap='magma', shading='gouraud')
    fig.colorbar(im, ax=ax, label='HHI coefficient of variation')
    ax.set_xlabel(r'AI efficiency gain $\Delta A$')
    ax.set_ylabel(r'$\sigma$')
    ax.set_title('Path Dependence Strength (HHI CV across seeds)')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'F5_hhi_path_dependence.png'))
    print(f'Saved: F5_hhi_path_dependence.png')
    plt.close(fig)

    # ---- Figure 6: 2x2 panel — σ slices ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sigma_slices = [2.0, 3.0, 5.0, 8.0]
    panel_idx = 0

    for ax in axes.flat:
        if panel_idx >= len(sigma_slices):
            ax.set_visible(False)
            continue
        sigma = sigma_slices[panel_idx]
        gains = []
        tfps = []
        tfp_stds = []
        lles = []
        paradoxes = []

        for gain in GAIN_VALS:
            k = f'sigma={sigma}_gain={gain}'
            if k in all_results:
                gains.append(gain)
                tfps.append(all_results[k]['macro_tfp_delta_mean'])
                tfp_stds.append(all_results[k]['macro_tfp_delta_std'])
                lles.append(all_results[k]['lyapunov_tfp_mean'])
                paradoxes.append(all_results[k]['paradox_rate'])

        if not gains:
            continue

        gains = np.array(gains)
        tfps = np.array(tfps)
        tfp_stds = np.array(tfp_stds)
        lles = np.array(lles)

        color = 'tab:blue'
        ax.fill_between(gains, tfps - tfp_stds, tfps + tfp_stds,
                        alpha=0.15, color=color)
        ax.plot(gains, tfps, 'o-', color=color, markersize=4, linewidth=1.5,
                label=r'$\Delta$TFP')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

        ax2 = ax.twinx()
        ax2.plot(gains, lles, 's--', color='tab:red', markersize=3, linewidth=1,
                 label=r'$\lambda_1$')
        ax2.axhline(0, color='tab:red', linestyle=':', linewidth=0.5, alpha=0.5)
        ax2.set_ylabel(r'$\lambda_1$', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Paradox rate as background shading
        paradoxes = np.array(paradoxes)
        for i in range(len(gains) - 1):
            if paradoxes[i] > 0.1:
                ax.axvspan(gains[i], gains[i+1], alpha=0.08, color='orange')

        ax.set_xlabel(r'AI gain $\Delta A$')
        ax.set_ylabel(r'$\Delta$ macro TFP')
        ax.set_title(f'σ = {sigma}')
        ax.legend(loc='upper left', fontsize=9)

        panel_idx += 1

    fig.suptitle('TFP Response & Lyapunov Exponent vs AI Gain', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'F6_sigma_slices.png'))
    print(f'Saved: F6_sigma_slices.png')
    plt.close(fig)

    # ---- Figure 7: Phase boundary summary ----
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all points colored by phase, with paradox contour overlay
    for i, sigma in enumerate(SIGMA_VALS):
        for j, gain in enumerate(GAIN_VALS):
            k = f'sigma={sigma}_gain={gain}'
            if k not in all_results:
                continue
            phase = all_results[k]['dominant_phase']
            tfp = all_results[k]['macro_tfp_delta_mean']
            lle = all_results[k]['lyapunov_tfp_mean']
            is_paradox = all_results[k]['paradox_rate'] > 0.1

            color_map = {'equilibrium': '#2196F3', 'oscillation': '#4CAF50',
                        'chaotic': '#F44336', 'unknown': '#9E9E9E',
                        'paradox_equilibrium': '#1565C0', 'paradox_oscillation': '#2E7D32',
                        'paradox_chaotic': '#B71C1C'}
            marker = '*' if is_paradox else 'o'
            size = 80 if is_paradox else 40
            ax.scatter(tfp, lle, c=color_map.get(phase, 'gray'),
                       marker=marker, s=size, edgecolors='k', linewidths=0.3,
                       alpha=0.7)

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel(r'$\Delta$ macro TFP')
    ax.set_ylabel(r'Largest Lyapunov exponent $\lambda_1$')

    legend_elements = [
        Patch(facecolor='#2196F3', edgecolor='k', label='Equilibrium'),
        Patch(facecolor='#4CAF50', edgecolor='k', label='Oscillation'),
        Patch(facecolor='#F44336', edgecolor='k', label='Chaotic'),
    ]
    from matplotlib.lines import Line2D
    legend_elements.append(Line2D([0], [0], marker='*', color='w',
                                  markerfacecolor='orange', markersize=10,
                                  label='Paradox (>10%)'))
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_title('Dynamical Phase Space: TFP change vs Lyapunov exponent')

    # Annotate quadrants
    ax.text(0.02, 0.98, 'Converging\n(positive)', transform=ax.transAxes,
            fontsize=8, va='top', ha='left', color='gray', alpha=0.5)
    ax.text(0.98, 0.98, 'Diverging\n(negative TFP)', transform=ax.transAxes,
            fontsize=8, va='top', ha='right', color='gray', alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'F7_phase_space_scatter.png'))
    print(f'Saved: F7_phase_space_scatter.png')
    plt.close(fig)

    print(f'\nAll figures saved to: {fig_dir}/')


if __name__ == '__main__':
    all_results = main()
    print('\nGenerating figures...')
    generate_figures(all_results)
    print('\nPhase diagram experiment complete.')
