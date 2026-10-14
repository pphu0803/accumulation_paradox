"""Visualizations for AI Productivity Paradox ABM.

Five figure types:
  Fig 1: Productivity paradox — micro TFP↑ vs macro TFP
  Fig 2: J-curve dynamics — firm productivity post-adoption
  Fig 3: Redistribution — labor share and profit share over time
  Fig 4: Phase diagram — demand_elasticity × ai_gain → macro TFP
  Fig 5: Adoption S-curve — AI diffusion over time
"""

import numpy as np
from typing import Dict, Any, List, Optional


def _setup_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': 150,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.figsize': (8, 5),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def fig1_paradox(metrics_history: List[Dict],
                 ai_intro_tick: int = 51,
                 save_path: Optional[str] = None):
    """Micro TFP (AI firms) vs macro TFP over time.

    If paradox holds: micro line rises, macro line stays flat.
    """
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(9, 5))

    ticks = [m['tick'] for m in metrics_history if m['tick'] >= 1]
    micro_ai = [m.get('micro_tfp_ai', 0) for m in metrics_history if m['tick'] >= 1]
    micro_noai = [m.get('micro_tfp_no_ai', 0) for m in metrics_history if m['tick'] >= 1]
    macro = [m.get('macro_tfp', 0) for m in metrics_history if m['tick'] >= 1]

    # Normalize to pre-AI baseline
    pre_vals = [(m['micro_tfp_ai'], m['micro_tfp_no_ai'], m['macro_tfp'])
               for m in metrics_history if m['tick'] < ai_intro_tick]
    if pre_vals:
        base_ai, base_noai, base_macro = np.mean(pre_vals, axis=0)
        micro_ai = [(v / base_ai - 1) * 100 for v in micro_ai]
        micro_noai = [(v / base_noai - 1) * 100 for v in micro_noai]
        macro = [(v / base_macro - 1) * 100 for v in macro]
    else:
        micro_ai = [0] * len(ticks)
        micro_noai = [0] * len(ticks)
        macro = [0] * len(ticks)

    ax.axvline(x=ai_intro_tick, color='red', linestyle='--', alpha=0.5, label='AI intro')
    ax.plot(ticks, micro_ai, label='Micro TFP (AI firms)', linewidth=2)
    ax.plot(ticks, macro, label='Macro TFP', linewidth=2)
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xlabel('Tick')
    ax.set_ylabel('TFP change from baseline (%)')
    ax.set_title('AI Productivity Paradox: Micro vs Macro TFP')
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig


def fig2_jcurve(metrics_history: List[Dict],
                ai_intro_tick: int = 51,
                save_path: Optional[str] = None):
    """J-curve: micro TFP of AI firms over time showing initial dip then recovery."""
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(9, 5))

    post = [m for m in metrics_history if m['tick'] >= ai_intro_tick]
    if not post:
        print('No post-AI data')
        return None

    ticks = [m['tick'] - ai_intro_tick for m in post]
    micro_ai = [m.get('micro_tfp_ai', 0) for m in post]

    ax.plot(ticks, micro_ai, linewidth=2, color='#2196F3')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(ticks, 1.0, micro_ai,
                    where=[v > 1.0 for v in micro_ai],
                    alpha=0.2, color='green', label='Net gain')
    ax.fill_between(ticks, 1.0, micro_ai,
                    where=[v < 1.0 for v in micro_ai],
                    alpha=0.2, color='red', label='J-curve dip')

    ax.set_xlabel(f'Ticks since AI introduction')
    ax.set_ylabel('Micro TFP (AI firms)')
    ax.set_title('J-Curve: AI Productivity Learning Dynamics')
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig


def fig3_redistribution(metrics_history: List[Dict],
                         ai_intro_tick: int = 51,
                         save_path: Optional[str] = None):
    """Labor share and profit share over time."""
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(9, 5))

    ticks = [m['tick'] for m in metrics_history if m['tick'] >= 1]
    labor = [m.get('labor_share', 0) for m in metrics_history if m['tick'] >= 1]
    profit = [m.get('profit_share', 0) for m in metrics_history if m['tick'] >= 1]

    ax.axvline(x=ai_intro_tick, color='red', linestyle='--', alpha=0.5, label='AI intro')
    ax.plot(ticks, labor, label='Labor share', linewidth=2)
    ax.plot(ticks, profit, label='Profit share', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlabel('Tick')
    ax.set_ylabel('Income share')
    ax.set_title('Redistribution: Labor vs Profit Share')
    ax.legend()
    ax.set_ylim(0, 1)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig


def fig4_phase_diagram(scan_results: Dict,
                      fixed_depth=0.0, fixed_cost=5.0,
                      metric='delta_macro_tfp',
                      save_path: Optional[str] = None):
    """Heatmap: demand_elasticity × ai_efficiency_gain → metric.

    Shows phase boundaries where the paradox emerges.
    """
    import matplotlib.pyplot as plt
    _setup_style()

    # Extract unique eps and gain values
    combos = {}
    for tag, data in scan_results.items():
        if tag == 'seeds' or 'summary' not in data:
            continue
        s = data['summary']
        if abs(s.get('depth', -1) - fixed_depth) < 0.01 and abs(s.get('cost', -1) - fixed_cost) < 0.1:
            combos[(s['eps'], s['gain'])] = s.get(metric, 0)

    if not combos:
        print(f'No data for depth={fixed_depth}, cost={fixed_cost}')
        return None

    epsilons = sorted(set(k[0] for k in combos.keys()))
    gains = sorted(set(k[1] for k in combos.keys()))

    Z = np.full((len(epsilons), len(gains)), np.nan)
    for (eps, gain), val in combos.items():
        i = epsilons.index(eps)
        j = gains.index(gain)
        Z[i, j] = val

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(Z, aspect='auto', cmap='RdBu_r',
                   extent=[min(gains)-0.01, max(gains)+0.01,
                           min(epsilons)-0.5, max(epsilons)+0.5],
                   origin='lower')

    ax.set_xlabel('AI Efficiency Gain')
    ax.set_ylabel('Demand Elasticity (ε)')
    metric_labels = {
        'delta_macro_tfp': 'Macro TFP change from baseline',
        'paradox_frac': 'Paradox fraction (micro↑ + macro≈0)',
        'delta_labor_share': 'Labor share change',
        'ai_adoption_rate': 'AI adoption rate',
    }
    ax.set_title(f'Phase Diagram: {metric_labels.get(metric, metric)}')

    # Add value annotations
    for i, eps in enumerate(epsilons):
        for j, gain in enumerate(gains):
            if not np.isnan(Z[i, j]):
                fmt = '.0%' if metric == 'paradox_frac' else '+.2f'
                ax.text(gain, eps, format(Z[i, j], fmt),
                        ha='center', va='center', fontsize=8,
                        color='white' if abs(Z[i, j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig


def fig5_adoption_scurve(metrics_history: List[Dict],
                         ai_intro_tick: int = 51,
                         save_path: Optional[str] = None):
    """AI adoption rate over time (S-curve)."""
    import matplotlib.pyplot as plt
    _setup_style()

    fig, ax = plt.subplots(figsize=(9, 5))

    ticks = [m['tick'] for m in metrics_history if m['tick'] >= 1]
    adoption = [m.get('ai_adoption_rate', 0) for m in metrics_history if m['tick'] >= 1]

    ax.axvline(x=ai_intro_tick, color='red', linestyle='--', alpha=0.5, label='AI intro')
    ax.plot(ticks, adoption, linewidth=2, color='purple')
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel('Tick')
    ax.set_ylabel('AI adoption rate')
    ax.set_title('AI Adoption Diffusion (S-curve)')
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig
