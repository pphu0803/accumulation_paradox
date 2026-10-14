"""Generate all paper figures from JSON data with consistent journal styling.

Figures are designed at their true printed size (the sn-jnl text block is
372pt = 5.17in wide, so canvases are ~5.2in and fonts render at face value:
nominal sizes span 6-9pt and nothing drops below the ~6pt journal floor after
\\includegraphics scaling. Do not upscale figsize without scaling every font
size with it.
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Layout-robust data path: in the release package this script sits at the
# package root (abm/ is a sibling directory); in the source repository it
# lives one directory down (abm/ sits one level up).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_CANDIDATES = [os.path.join(_HERE, 'abm', 'output'),
                    os.path.join(_HERE, '..', 'abm', 'output')]
DATA_DIR = next((p for p in _DATA_CANDIDATES if os.path.isdir(p)), _DATA_CANDIDATES[-1])

RC = {
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    # supxlabel/supylabel: 'large' would give 9.6pt, oversized relative to the
    # per-panel 8.5pt axis labels -- pin it to the same 8.5pt
    'figure.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    # 300: the imshow'd heatmaps rasterize at this resolution; 200 sat below
    # the Elsevier 300-dpi floor for halftone content
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    # pad so end-of-string mathtext glyphs (e.g. a trailing sigma) are not
    # clipped by the tight bbox, which under-measures their advance width
    'savefig.pad_inches': 0.04,
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    'pdf.fonttype': 42, 'axes.axisbelow': True,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'lines.linewidth': 1.2,
    'legend.fontsize': 7, 'legend.framealpha': 0.9,
}

# Paper-wide color convention (keep every figure inside this set):
#   C_BLUE   -- the default arm / level quantities (imitation ON, gate ON,
#               single-series bars)
#   C_RED    -- the counterfactual arm (no imitation, gate OFF); also the
#               lone loss component (exit) in the figE2 stack
#   C_ORANGE -- derived quantities and event markers (gate effect, crossover,
#               pass-through ratio, the within-firm efficiency channel)
#   C_GRAY / C_LGRAY -- auxiliary references and residual components
#   viridis  -- every heatmap / sequential field
# Ordered sets deepen through one hue instead of switching hues: the blue ramp
# below (RdBu's blue half, light -> dark) carries ordinal gains (fig4), sector
# types (fig6) and gate speeds (figA2, with the ungated baseline in gray).
C_BLUE = '#2166AC'
C_RED = '#B2182B'
C_ORANGE = '#E08214'
C_GRAY = '#888888'
C_SKY = '#56B4E9'
C_LGRAY = '#BFBFBF'

def _blues(fracs):
    return [plt.get_cmap('RdBu')(f) for f in fracs]


def _load(name):
    p = os.path.join(DATA_DIR, name)
    with open(p) as f:
        return json.load(f)


def _sigma_gain_grid(data, field):
    sigmas = sorted(set(float(k.split('sigma=')[1].split('_')[0]) for k in data))
    gains = sorted(set(float(k.split('gain=')[1].split('_')[0]) for k in data))
    mat = np.full((len(sigmas), len(gains)), np.nan)
    for key, v in data.items():
        i = sigmas.index(float(key.split('sigma=')[1].split('_')[0]))
        j = gains.index(float(key.split('gain=')[1].split('_')[0]))
        mat[i, j] = v[field]
    return sigmas, gains, mat


def _cell_edges(vals):
    """True cell boundaries for a (possibly non-uniform) sampled grid, so
    pcolormesh places each cell between midpoints of neighbouring samples
    instead of stretching it (the imshow-default distortion)."""
    v = np.asarray(sorted(vals), dtype=float)
    edges = np.empty(len(v) + 1)
    edges[1:-1] = (v[:-1] + v[1:]) / 2.0
    edges[0] = v[0] - (v[1] - v[0]) / 2.0
    edges[-1] = v[-1] + (v[-1] - v[-2]) / 2.0
    return edges


def _annot_color(cmap, frac):
    """Black or white cell-annotation text, whichever contrasts with the cell.
    The 0.6 switch point keeps white type off viridis' mid-bright band, where
    its contrast falls under the 4.5:1 small-text floor."""
    rgba = plt.get_cmap(cmap)(min(max(frac, 0.0), 1.0))
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return 'white' if lum < 0.6 else 'black'


def fig1_phase_diagram():
    plt.rcParams.update(RC)
    data = _load('phase_diagram/phase_diagram_raw.json')

    fig, axes = plt.subplots(1, 2, figsize=(5.15, 2.45))

    # Both heatmaps on index axes with uniform cells; the tick labels carry
    # the sampled values, so the categorical reading is unambiguous.
    sigmas, gains, dtfp = _sigma_gain_grid(data, 'macro_tfp_delta_mean')

    # Panel A: delta TFP heatmap (continuous measure)
    ax = axes[0]
    im = ax.imshow(dtfp * 100, cmap='viridis', aspect='auto', origin='lower')
    ax.set_title('(a) Measured TFP change (%)')
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels([f'{g:.2f}' for g in gains], rotation=45, ha='right')
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels([f'{s:g}' for s in sigmas])
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    # no colorbar caption: it repeated the panel title verbatim
    cb.ax.tick_params(labelsize=7)

    # Panel B: Low-transmission rate. The rate is a count over 20 seeds, so
    # it only takes the values 0/5/10/15/20% -- a discrete 5-band colormap
    # makes that quantization visible instead of implying a continuous field.
    _, _, paradox = _sigma_gain_grid(data, 'paradox_rate')
    ax = axes[1]
    cmap_d = ListedColormap(plt.get_cmap('viridis')(np.linspace(0.05, 0.9, 5)))
    # first boundary at -2.5 (not 0) so every tick, including 0, sits at a
    # band centre instead of 0 hugging the bottom edge of the colorbar
    norm_d = BoundaryNorm([-2.5, 2.5, 7.5, 12.5, 17.5, 22.5], cmap_d.N)
    im2 = ax.imshow(paradox * 100, cmap=cmap_d, norm=norm_d,
                    aspect='auto', origin='lower')
    ax.set_title('(b) Low-transmission rate (%)')
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels([f'{g:.2f}' for g in gains], rotation=45, ha='right')
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels([f'{s:g}' for s in sigmas])
    # no colorbar caption: it repeated the panel title verbatim
    cb2 = plt.colorbar(im2, ax=ax, shrink=0.85, ticks=[0, 5, 10, 15, 20])
    cb2.ax.tick_params(labelsize=7)

    # paper-wide convention: panels sharing an axis carry one shared label
    fig.supxlabel('AI efficiency gain $g$')
    fig.supylabel(r'Substitution elasticity $\sigma$')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_phase_diagram.pdf'), metadata={'CreationDate': None})
    plt.close(fig)
    print('Saved fig1_phase_diagram.pdf')


def fig2_industry_calibration():
    """Rank correlation over time. The calibration scatter is Table 4's
    content; the table carries it, so the figure keeps only the time series."""
    plt.rcParams.update(RC)
    crossover = _load('supplementary/analysis1_crossover.json')

    fig, ax = plt.subplots(figsize=(5.2, 2.0))

    ticks = crossover['tick_labels']
    rhos = crossover['rho']
    years = [t / 12.0 for t in ticks]

    for i in range(1, len(years)):
        color = C_RED if rhos[i] < 0 else C_BLUE
        style = '--' if rhos[i - 1] < 0 else '-'
        ax.plot(years[i - 1:i + 1], rhos[i - 1:i + 1], linestyle=style,
                color=color, linewidth=1.4)

    ax.axhline(y=0, color='black', linewidth=0.6)
    ax.fill_between(years, 0, rhos, alpha=0.15,
                    where=[r < 0 for r in rhos], color=C_RED)
    ax.fill_between(years, 0, rhos, alpha=0.15,
                    where=[r > 0 for r in rhos], color=C_BLUE)
    # Crossover marker: read the sign-change tick from the archived data
    # (do not hardcode -- the pre-v3 data crossed at year 4, the v3 rerun at year 3).
    sign_changes = crossover.get('sign_changes') or []
    if sign_changes:
        cross_year = sign_changes[0][0] / 12.0
        label_year = f'{cross_year:.0f}' if cross_year >= 1 else f'{cross_year:.1f}'
        ax.axvline(x=cross_year, color=C_ORANGE, linestyle='--', linewidth=1.0, alpha=0.8)
        ax.text(cross_year + 0.15, -0.55,
                f'Crossover\n$\\approx${label_year} years', fontsize=7, color=C_ORANGE,
                va='center')

    ax.text(1.35, 0.95, 'Competitive\nsectors\nadopt faster', fontsize=6.5, color=C_RED,
            ha='center', va='top', style='italic')
    ax.text(8.0, 0.28, 'Concentrated sectors\nreach higher adoption', fontsize=6.5,
            color=C_BLUE, ha='center', style='italic')

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Spearman $\\rho$')
    ax.set_xlim(0, max(years))
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2_industry_calibration.pdf'), metadata={'CreationDate': None})
    plt.close(fig)
    print('Saved fig2_industry_calibration.pdf')


def fig3_bistability():
    plt.rcParams.update(RC)
    data = _load('supplementary/analysis2_bistability.json')

    fig, axes = plt.subplots(1, 3, figsize=(5.2, 1.95))
    # Histograms deepen through the blue ramp with the sector's sigma: the
    # panels are ordinal (information -> manufacturing -> construction), and
    # one hue deepening reads as an ordering, not as unrelated categories.
    # Explicit low-to-high order so the panels match the caption's
    # "(a) information ... (c) construction" reading; fall back to file order.
    sector_shades = _blues([0.70, 0.85, 1.00])
    order = ['Low-sigma (Information)', 'Mid-sigma (Manufacturing)',
             'High-sigma (Construction)']
    labels = [k for k in order if k in data] or list(data.keys())

    for idx, label in enumerate(labels):
        d = data[label]
        ax = axes[idx]
        adoptions = np.array(d['adoptions']) * 100
        # Derived stats may be absent in freshly generated files; compute from
        # the raw seed list so the figure works against any writer version.
        a_mean = d.get('adoption_mean', float(np.mean(adoptions)) / 100) * 100
        a_std = d.get('adoption_std', float(np.std(adoptions)) / 100)
        a_cv = d.get('adoption_cv', (a_std / a_mean) if a_mean else 0.0)
        ax.hist(adoptions, bins=15, color=sector_shades[idx],
                alpha=0.8, edgecolor='white', linewidth=0.4)
        # Classification cuts that define the Low/Mid/High counts (dotted);
        # fixed x-limits keep the out-of-range cuts from stretching the axis.
        # Line identities are given in the caption.
        xmin, xmax = adoptions.min(), adoptions.max()
        pad = max((xmax - xmin) * 0.07, 0.5)
        ax.axvline(30, color=C_GRAY, linestyle=':', linewidth=0.9)
        ax.axvline(70, color=C_GRAY, linestyle=':', linewidth=0.9)
        ax.axvline(a_mean, color='black', linestyle='--', linewidth=0.9)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        n_low, n_mid, n_high = d['n_low'], d['n_mid'], d['n_high']
        # Stats live in a second title line, not in an in-axes box: the box
        # occluded the modal bar (panel c) and the 70% cut line (panel b),
        # and its translucent backing made the occlusion look like data.
        sigma_label = label.split('(')[1].rstrip(')')
        ax.set_title(f'({chr(97+idx)}) $\\sigma$={d["sigma"]:.1f} ({sigma_label})\n'
                     f'CV = {a_cv:.2f}, L{n_low}/M{n_mid}/H{n_high}',
                     fontsize=7.5)

    fig.supxlabel('AI adoption rate (%)')
    fig.supylabel('Frequency (seeds)')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_bistability.pdf'), metadata={'CreationDate': None})
    plt.close(fig)
    print('Saved fig3_bistability.pdf')


def fig4_markup_tfp():
    plt.rcParams.update(RC)
    data = _load('supplementary/analysis3_markup_tfp.json')

    spreads = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    gains = [0.02, 0.04, 0.06, 0.08]
    # Gain is ordinal, so the four series share one hue that deepens with it
    # (line style + marker keep them separable when color fails).
    colors = _blues([0.70, 0.80, 0.90, 1.00])
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']

    # taller than the minimum: panel (a) stacks four series over a ~7-unit
    # range, and at the previous height its plot area was squeezed to ~65pt
    # (the legend below takes the rest); height is free under \textwidth, so
    # only the layout breathes, never the fonts
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))

    # Panel A: TFP transmission vs Gini (line style + marker redundancy so the
    # four gain levels survive grayscale print)
    ax = axes[0]
    for gi, gain in enumerate(gains):
        ginis = [data[f'spread={s}'][f'gain={gain}']['markup_gini_mean'] for s in spreads]
        effs = [data[f'spread={s}'][f'gain={gain}']['delta_tfp_mean'] / gain for s in spreads]
        ax.plot(ginis, effs, marker=markers[gi], linestyle=styles[gi],
                color=colors[gi], linewidth=1.2, markersize=3.5,
                label=f'gain = {gain}')
    ax.set_xlabel('Markup Gini coefficient')
    # two lines: the single-line label is longer than the panel is tall and
    # the tight bbox truncates it at the canvas edge
    ax.set_ylabel('Transmission\n($\\Delta$TFP / gain)')
    ax.set_title('(a) Transmission')
    ax.legend(fontsize=6.5, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.44),
              frameon=False, handlelength=1.8, columnspacing=0.8)
    ax.grid(True, alpha=0.3)

    # Annotation
    gini_low = data['spread=0.0']['gain=0.04']['markup_gini_mean']
    eff_low = data['spread=0.0']['gain=0.04']['delta_tfp_mean'] / 0.04
    gini_high = data['spread=1.0']['gain=0.04']['markup_gini_mean']
    eff_high = data['spread=1.0']['gain=0.04']['delta_tfp_mean'] / 0.04
    # endpoint annotations, color-matched to the gain=0.04 series they belong
    # to (both endpoints sit on that series -- the text quotes "5.0 to 9.7 at
    # g = 0.04"); white backing keeps them legible where another series' line
    # passes underneath. The high endpoint floats in clear space above its
    # marker; the low one sits in the crowded left-edge cluster, so it is
    # placed in axes fraction just above the cluster and below the gain=0.02
    # line -- an offset-points placement lands outside the axes there.
    ann_bbox = dict(boxstyle='round,pad=0.2', facecolor='white',
                    edgecolor='none', alpha=0.85)
    ax.annotate(f'{eff_high:.1f}', (gini_high, eff_high), fontsize=7,
                color=colors[1], textcoords='offset points', xytext=(-2, 8),
                ha='right', bbox=ann_bbox)
    ax.annotate(f'{eff_low:.1f}', (gini_low, eff_low), fontsize=7,
                color=colors[1], textcoords='axes fraction',
                xytext=(0.035, 0.24), ha='left', va='bottom', bbox=ann_bbox)

    # Panel B: Annualized TFP heatmap
    ax = axes[1]
    ann = np.array([[data[f'spread={s}'][f'gain={g}']['annualized_tfp'] * 100
                      for g in gains] for s in spreads])
    lo, hi = float(np.nanmin(ann)), float(np.nanmax(ann))
    im = ax.imshow(ann, cmap='viridis', aspect='auto', vmin=lo, vmax=hi)
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels([str(g) for g in gains])
    ax.set_yticks(range(len(spreads)))
    ax.set_yticklabels([f'{s:.1f}' for s in spreads])
    ax.set_xlabel('AI gain')
    ax.set_ylabel(r'$\sigma_{\mathrm{firm}}$ spread')
    ax.set_title(r'(b) Annualized TFP (%)')
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    # two lines: the single-line label is taller than the colorbar and the
    # tight bbox truncates it at the canvas edge
    cb.set_label('Annualized TFP\ngrowth (%)', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    for i in range(len(spreads)):
        for j in range(len(gains)):
            frac = (ann[i, j] - lo) / (hi - lo)
            ax.text(j, i, f'{ann[i,j]:.1f}', ha='center', va='center',
                    fontsize=6, color=_annot_color('viridis', frac))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig4_markup_tfp.pdf'), metadata={'CreationDate': None})
    plt.close(fig)
    print('Saved fig4_markup_tfp.pdf')


def fig5_counterfactual():
    plt.rcParams.update(RC)
    data = _load('counterfactual_strategy/counterfactual_strategy_results.json')

    demands = [0.0, 0.04, 0.08, 0.12, 0.20]
    lr_keys = {0.0: [f'lr=0.0_demand={d}' for d in demands],
               0.05: [f'lr=0.05_demand={d}' for d in demands]}

    def get_vals(keys, field):
        return [data[k][field] for k in keys]

    # One style per arm everywhere: solid blue squares (imitation) vs dashed
    # red circles (no imitation), so the arms survive grayscale print. The
    # counterfactual arm is drawn last so where the arms converge its caps
    # stay visible instead of being overplotted. A single figure-level legend
    # below the grid serves all four panels.
    def arm(ax, ys0, ys1, e0=None, e1=None):
        kw = dict(markersize=3.5, capsize=2, elinewidth=0.7, linewidth=1.0)
        if e1 is None:
            ax.plot(demands, ys1, 's-', color=C_BLUE, label='With imitation ($\\lambda$=0.05)', **kw)
        else:
            ax.errorbar(demands, ys1, yerr=e1, fmt='s-', color=C_BLUE,
                        label='With imitation ($\\lambda$=0.05)', **kw)
        if e0 is None:
            ax.plot(demands, ys0, 'o--', color=C_RED, label='No imitation ($\\lambda$=0)', **kw)
        else:
            ax.errorbar(demands, ys0, yerr=e0, fmt='o--', color=C_RED,
                        label='No imitation ($\\lambda$=0)', **kw)
        ax.set_xticks(demands)
        ax.set_xticklabels(['0.00', '0.04', '0.08', '0.12', '0.20'])
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.1))

    # Panel A: TFP (with cross-seed s.d., which the text's significance
    # qualifier refers to)
    ax = axes[0, 0]
    v0 = [x * 100 for x in get_vals(lr_keys[0.0], 'delta_tfp_mean')]
    v1 = [x * 100 for x in get_vals(lr_keys[0.05], 'delta_tfp_mean')]
    e0 = [x * 100 for x in get_vals(lr_keys[0.0], 'delta_tfp_std')]
    e1 = [x * 100 for x in get_vals(lr_keys[0.05], 'delta_tfp_std')]
    arm(ax, v0, v1, e0, e1)
    ax.set_ylabel(r'$\Delta$TFP (%)')
    ax.set_title('(a) TFP change')

    # Panel B: HHI
    ax = axes[0, 1]
    h0 = get_vals(lr_keys[0.0], 'hhi_mean')
    h1 = get_vals(lr_keys[0.05], 'hhi_mean')
    h0e = get_vals(lr_keys[0.0], 'hhi_std')
    h1e = get_vals(lr_keys[0.05], 'hhi_std')
    arm(ax, h0, h1, h0e, h1e)
    ax.set_ylabel('HHI (market concentration)')
    ax.set_title('(b) Market concentration')
    pct = (np.mean(h0) - np.mean(h1)) / np.mean(h1) * 100
    # upper-left, with the direction spelled out: a bare "Delta HHI: +31%"
    # reads as a within-arm change, but it is the across-arm mean difference
    ax.text(0.05, 0.95, f'Without imitation: HHI {pct:+.0f}%',
            transform=ax.transAxes, fontsize=7, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9))

    # Panel C: Strategy diversity
    ax = axes[1, 0]
    s0 = get_vals(lr_keys[0.0], 'strategy_std_mean')
    s1 = get_vals(lr_keys[0.05], 'strategy_std_mean')
    s0e = get_vals(lr_keys[0.0], 'strategy_std_std')
    s1e = get_vals(lr_keys[0.05], 'strategy_std_std')
    arm(ax, s0, s1, s0e, s1e)
    ax.set_ylabel('Strategy diversity (std)')
    ax.set_title('(c) Strategy diversity')

    # Panel D: AI investment
    ax = axes[1, 1]
    inv0 = get_vals(lr_keys[0.0], 'mean_ai_investment_mean')
    inv1 = get_vals(lr_keys[0.05], 'mean_ai_investment_mean')
    inv0e = get_vals(lr_keys[0.0], 'mean_ai_investment_std')
    inv1e = get_vals(lr_keys[0.05], 'mean_ai_investment_std')
    arm(ax, inv0, inv1, inv0e, inv1e)
    ax.set_ylabel('Mean AI investment rate')
    ax.set_title('(d) AI investment intensity')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01),
               ncol=2, frameon=False, fontsize=7, handlelength=1.8,
               columnspacing=1.2)
    fig.supxlabel('Demand shock coefficient')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig5_counterfactual.pdf'), metadata={'CreationDate': None})
    plt.close(fig)
    print('Saved fig5_counterfactual.pdf')


def fig_decomposition():
    plt.rcParams.update(RC)
    data = _load('robustness/decomposition_summary.json')

    cells = {k: v for k, v in data.items() if k.startswith('sigma=')}
    sigmas = sorted(float(k.split('=')[1]) for k in cells)
    # Categorical positions: the design runs {2,...,8,10}, and a numeric axis
    # would read the unrun sigma=9 slot as a zero gate effect.
    xpos = np.arange(len(sigmas))
    xlabels = [f'{s:g}' for s in sigmas]
    tfp_on = [cells[f'sigma={s}']['tfp_gate_on'] * 100 for s in sigmas]
    tfp_off = [cells[f'sigma={s}']['tfp_gate_off'] * 100 for s in sigmas]
    tfp_on_std = [cells[f'sigma={s}']['tfp_gate_on_std'] * 100 for s in sigmas]
    tfp_off_std = [cells[f'sigma={s}']['tfp_gate_off_std'] * 100 for s in sigmas]
    gate_effects = [cells[f'sigma={s}']['gate_effect_tfp'] * 100 for s in sigmas]
    ci_lo = [cells[f'sigma={s}']['gate_effect_tfp_paired_ci95'][0] * 100 for s in sigmas]
    ci_hi = [cells[f'sigma={s}']['gate_effect_tfp_paired_ci95'][1] * 100 for s in sigmas]
    adopt_on = [cells[f'sigma={s}']['adopt_gate_on'] * 100 for s in sigmas]
    adopt_off = [cells[f'sigma={s}']['adopt_gate_off'] * 100 for s in sigmas]

    def cat_axis(ax):
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels)

    fig, axes = plt.subplots(1, 3, figsize=(5.2, 2.0))

    # Panel (a): dTFP vs sigma, gate ON/OFF with seed-std bands. Solid blue
    # squares (ON, the paper-wide default-arm color) vs open dashed red
    # circles (OFF) keep the arms separable in grayscale; ON is drawn first
    # and OFF last so coincident points at the converging end stay visible
    # (same convention as panel c). The marginal bands are seed dispersion,
    # not paired intervals.
    ax = axes[0]
    ax.fill_between(xpos, np.array(tfp_on) - np.array(tfp_on_std),
                    np.array(tfp_on) + np.array(tfp_on_std),
                    color=C_BLUE, alpha=0.15, linewidth=0)
    ax.fill_between(xpos, np.array(tfp_off) - np.array(tfp_off_std),
                    np.array(tfp_off) + np.array(tfp_off_std),
                    color=C_RED, alpha=0.10, linewidth=0)
    ax.plot(xpos, tfp_on, 's-', color=C_BLUE, markersize=3.5,
            label='Gate ON (default)')
    ax.plot(xpos, tfp_off, 'o--', color=C_RED, markerfacecolor='none',
            markersize=3.5, label='Gate OFF')
    cat_axis(ax)
    ax.set_ylabel(r'$\Delta$TFP (%)')
    ax.set_title('(a) $\\Delta$TFP by gate', fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # Panel (b): gate suppression by sigma, with the paired 95% CI over
    # seed-level differences (the interval the text quotes).
    ax = axes[1]
    ax.bar(xpos, gate_effects, width=0.6, color=C_ORANGE, alpha=0.85,
           edgecolor='white', linewidth=0.4)
    yerr = np.array([np.array(gate_effects) - np.array(ci_lo),
                     np.array(ci_hi) - np.array(gate_effects)])
    ax.errorbar(xpos, gate_effects, yerr=yerr, fmt='none', ecolor='black',
                elinewidth=0.7, capsize=1.8)
    ax.axhline(0, color='black', linewidth=0.5)
    cat_axis(ax)
    ax.set_ylabel('Gate effect (pp)')
    ax.set_title('(b) Gate suppression', fontsize=7.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel (c): adoption rate by gate condition (open markers drawn last so
    # coincident points at the extremes stay visible)
    ax = axes[2]
    ax.plot(xpos, adopt_on, 's-', color=C_BLUE, markersize=3.5,
            label='Gate ON (default)')
    ax.plot(xpos, adopt_off, 'o--', color=C_RED, markerfacecolor='none',
            markersize=3.5, label='Gate OFF')
    cat_axis(ax)
    ax.set_ylabel('AI adoption rate (%)')
    ax.set_title('(c) Adoption by gate', fontsize=7.5)
    ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01),
               ncol=2, frameon=False, fontsize=7, handlelength=1.8,
               columnspacing=1.2)
    # one shared xlabel -- paper-wide convention for panels sharing an axis
    # (here it also dodges the tight-bbox under-measure of a trailing mathtext
    # sigma, which silently dropped the rightmost panel's own label)
    fig.supxlabel(r'Substitution elasticity $\sigma$')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_decomposition.pdf'), metadata={'CreationDate': None})
    plt.close(fig)
    print('Saved fig_decomposition.pdf')


def figE2_mechanism():
    """E2: shift-share mechanism + pass-through across sigma."""
    plt.rcParams.update(RC)
    data = _load('revision/e2/e2_sigma_mechanism.json')
    sigmas = sorted(float(k.split('=')[1]) for k in data)

    def mean_over(k, fn):
        return float(np.mean([fn(r) for r in data[k]]))

    within = [mean_over(f'sigma={s}', lambda r: r['shift_share']['within'] * 100) for s in sigmas]
    realloc = [mean_over(f'sigma={s}', lambda r: r['shift_share']['reallocation'] * 100) for s in sigmas]
    exit_ = [mean_over(f'sigma={s}', lambda r: r['shift_share']['exit'] * 100) for s in sigmas]
    entry = [mean_over(f'sigma={s}', lambda r: r['shift_share']['entry'] * 100) for s in sigmas]
    inter = [mean_over(f'sigma={s}', lambda r: r['shift_share']['interaction'] * 100) for s in sigmas]
    # Net change in the bars' own units (index points): the five components
    # sum exactly to TFP_post - TFP_pre per run, so the overlaid line closes
    # on the stack instead of mixing a relative measure with level points.
    total_level = [mean_over(f'sigma={s}',
                             lambda r: sum(r['shift_share'].values()) * 100)
                   for s in sigmas]
    pt = [mean_over(f'sigma={s}', lambda r: r['passthrough_median']) for s in sigmas]
    adopt = [mean_over(f'sigma={s}', lambda r: r['adoption_rate'] * 100) for s in sigmas]

    fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.3))

    # Panel A: diverging stack -- positive components pile up from zero,
    # negative components downward, so no overlay paints over another.
    # Solid fills only: hatching read as texture noise at print size. The
    # stack order interleaves light and dark so adjacent segments keep their
    # luminance separation in grayscale (orange 146 | blue 89 | sky 158
    # luma), with the residual interaction a light-gray sliver on top and
    # exit -- the one loss term, alone below zero -- in red.
    ax = axes[0]
    n = len(sigmas)
    pos_bottom = np.zeros(n)
    neg_bottom = np.zeros(n)
    for vals, label, color in [
            (within, 'Within-firm AI gains', C_ORANGE),
            (realloc, 'Reallocation', C_BLUE),
            (entry, 'Entry', C_SKY),
            (inter, 'Interaction', C_LGRAY),
            (exit_, 'Exit', C_RED)]:
        v = np.array(vals)
        bottom = np.where(v >= 0, pos_bottom, neg_bottom)
        ax.bar(sigmas, v, bottom=bottom, width=0.55, label=label,
               color=color, alpha=0.9, edgecolor='white', linewidth=0.4)
        pos_bottom += np.clip(v, 0, None)
        neg_bottom += np.clip(v, None, 0)
    ax.plot(sigmas, total_level, 'ko-', label=r'$\Delta$TFP$_{\mathrm{AI}}$',
            markersize=3.5, linewidth=1.0, zorder=5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylim(min(neg_bottom) * 1.2, max(pos_bottom) * 1.12)
    ax.set_xticks(sigmas)
    ax.set_ylabel('Contribution to $\\Delta$TFP$_{\\mathrm{AI}}$\n(index points)')
    ax.set_title('(a) Shift-share decomposition', fontsize=7.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: dual axis, each side color-matched to its series (orange for
    # the derived ratio, blue for the level bars -- paper-wide convention)
    ax = axes[1]
    ax.plot(sigmas, pt, 'o-', color=C_ORANGE, markersize=3.5,
            label='Pass-through (median firm)')
    ax.axhline(1.0, color=C_GRAY, linestyle='--', linewidth=0.8,
               label='Complete pass-through')
    ax.set_xticks(sigmas)
    ax.set_ylabel(r'$\Delta \log P\, / \, \Delta \log MC$', color=C_ORANGE)
    ax.tick_params(axis='y', colors=C_ORANGE)
    ax.set_ylim(0.9, 1.1)
    ax2 = ax.twinx()
    ax2.bar(sigmas, adopt, width=0.4, color=C_BLUE, alpha=0.5, label='AI adoption (%)')
    ax2.set_ylabel('AI adoption rate (%)', color=C_BLUE)
    ax2.tick_params(axis='y', colors=C_BLUE)
    ax2.set_ylim(0, 100)
    # draw the pass-through line above the adoption bars (twinx puts ax2 last)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.set_title('(b) Pass-through vs adoption', fontsize=7.5)
    # no grid on ax: raising ax's zorder lifts its grid above ax2's bars too
    # (axisbelow only holds within one axes), scratching grey lines through
    # every bar; the grid belongs to the bar axis instead
    ax.grid(False)
    ax2.grid(True, alpha=0.3)

    # one figure-level legend below the shared xlabel (axes-level legends
    # anchored under the panels hang outside the canvas and strand the
    # supxlabel above a band of dead space)
    h_a, l_a = axes[0].get_legend_handles_labels()
    fig.legend(h_a + lines1 + lines2, l_a + labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3,
               frameon=False, fontsize=6.5, handlelength=1.3, columnspacing=0.8)
    fig.supxlabel(r'Substitution elasticity $\sigma$')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figE2_mechanism.pdf'), metadata={'CreationDate': None})
    plt.close(fig)
    print('Saved figE2_mechanism.pdf')


def figA1_comprehensive():
    plt.rcParams.update(RC)
    data = _load('comprehensive/comprehensive_results.json')

    gains = sorted(set(data[k]['ai_gain'] for k in data))
    demands = sorted(set(data[k]['demand_coeff'] for k in data))

    def heatmap(ax, mat, title, fmt, unit):
        lo, hi = float(np.nanmin(mat)), float(np.nanmax(mat))
        # index axis: uniform cells (the demand grid is unevenly spaced, and
        # true-coordinate cells read as a layout bug rather than as honesty)
        im = ax.imshow(mat, cmap='viridis', aspect='auto', vmin=lo, vmax=hi,
                       origin='lower')
        ax.set_xticks(range(len(demands)))
        ax.set_xticklabels([f'{d:g}' for d in demands], fontsize=6.5,
                           rotation=45, ha='right')
        ax.set_yticks(range(len(gains)))
        ax.set_yticklabels([f'{g:g}' for g in gains], fontsize=6.5)
        ax.set_title(title, fontsize=7.5)
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(unit, fontsize=6.5)
        cb.ax.tick_params(labelsize=6.5)
        # anchor ticks at both data ends: the auto locator leaves the
        # brightest cells unticked, so the extremes carry no numeric anchor
        # (and panel b, with no cell annotations, had none at all)
        mid = [t for t in MaxNLocator(4).tick_values(lo, hi) if lo < t < hi]
        span = hi - lo
        ticks = sorted(set(mid + [lo, hi]))
        ticks = [t for t in ticks
                 if t in (lo, hi) or all(abs(t - e) > 0.08 * span for e in (lo, hi))]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{t:.1f}' if span >= 2 else f'{t:.2f}' for t in ticks])
        if fmt is None:
            return
        for i in range(len(gains)):
            for j in range(len(demands)):
                frac = (mat[i, j] - lo) / (hi - lo)
                ax.text(j, i, fmt.format(mat[i, j]),
                        ha='center', va='center', fontsize=6,
                        color=_annot_color('viridis', frac))

    fig, axes = plt.subplots(1, 3, figsize=(5.26, 2.15))

    # Panel A: delta TFP heatmap
    mat = np.array([[data[f'gain={g}_demand={d}']['delta_tfp_mean'] * 100
                     for d in demands] for g in gains])
    heatmap(axes[0], mat, '(a) $\\Delta$TFP (%)', '{:.0f}', '$\\Delta$TFP (%)')

    # Panel B: Markup Gini heatmap (scaled to x10^-3 so the 4-digit cell
    # annotations fit and the colorbar ticks stay readable). The colorbar
    # label stays bare: a mathtext superscript at 6.5pt base renders ~4.5pt,
    # under the journal floor -- the x10^-3 scaling is declared in the title.
    mat2 = np.array([[data[f'gain={g}_demand={d}']['markup_gini_mean'] * 1000
                      for d in demands] for g in gains])
    heatmap(axes[1], mat2, '(b) Markup Gini ($\\times 10^{-3}$)', None, 'Gini')

    # Panel C: Adoption rate heatmap (auto range -- the 50-100 floor hid the
    # 87-91% spread the appendix text quotes)
    mat3 = np.array([[data[f'gain={g}_demand={d}']['adoption_rate_mean'] * 100
                      for d in demands] for g in gains])
    heatmap(axes[2], mat3, '(c) Adoption rate (%)', '{:.0f}', 'Adoption rate (%)')

    fig.supxlabel('Demand coefficient')
    fig.supylabel('AI efficiency gain')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figA1_comprehensive.pdf'), metadata={'CreationDate': None})
    plt.close(fig)
    print('Saved figA1_comprehensive.pdf')


def figA2_scurve():
    plt.rcParams.update(RC)
    data = _load('s_curve_verification/s_curve_results.json')

    # sharey: both panels rank the same six configurations, so repeating the
    # labels on panel b only steals ~40% of its width
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.1), sharey=True)

    configs = sorted(data.keys())

    def short(k):
        if 'baseline' in k:
            return 'No gate (baseline)'
        return (k.replace('speed=', r'$\nu$=')
                 .replace(' threshold=', ', $R^{\\star}$='))

    labels_short = [short(k) for k in configs]
    tfp_vals = [data[k].get('annualized_tfp_mean', 0) * 100 for k in configs]
    adopt_vals = [data[k].get('adoption_rate_mean', 0) * 100 for k in configs]

    # Bars deepen with the diffusion speed nu (the ordinal parameter here);
    # the ungated baseline takes gray, the paper's reference-slot color.
    # Identity stays in the y labels -- color only makes the speed ordering
    # visible across the two panels.
    nu_shades = {0.01: 0.70, 0.02: 0.80, 0.03: 0.90, 0.04: 1.00}

    def bar_color(key):
        if 'baseline' in key:
            return C_GRAY
        nu = float(key.split('speed=')[1].split(' ')[0])
        return _blues([nu_shades[nu]])[0]

    bar_colors = [bar_color(k) for k in configs]

    ax = axes[0]
    bars = ax.barh(range(len(configs)), tfp_vals, color=bar_colors,
                   alpha=0.85, edgecolor='white', linewidth=0.4)
    # end labels anchor the numbers the caption quotes (+2.1 without gating,
    # +1.9 under the default gate): without them the caption's headline
    # comparison has no visible locus in the panel
    ax.bar_label(bars, fmt='%+.2f', fontsize=6.5, padding=2)
    ax.set_xlim(0, 2.6)  # headroom so the end labels clear the right spine
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(labels_short, fontsize=6.5)
    ax.set_xlabel('Annualized TFP growth (%)')
    # short titles: the x labels already carry the full quantity names, and
    # with sharey the panels sit close enough that long titles run together
    ax.set_title('(a) Annualized TFP', fontsize=7.5)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.0,
               label='BLS ~1%/yr')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.36),
              frameon=False, handlelength=1.5)
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1]
    ax.barh(range(len(configs)), adopt_vals, color=bar_colors,
            alpha=0.85, edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(labels_short, fontsize=6.5)
    ax.set_xlabel('AI adoption rate (%)')
    ax.set_title('(b) AI adoption', fontsize=7.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figA2_scurve.pdf'), metadata={'CreationDate': None})
    plt.close(fig)
    print('Saved figA2_scurve.pdf')


if __name__ == '__main__':
    fig1_phase_diagram()
    fig2_industry_calibration()
    fig3_bistability()
    fig4_markup_tfp()
    fig5_counterfactual()
    fig_decomposition()
    figE2_mechanism()
    figA1_comprehensive()
    figA2_scurve()
    print('\nAll figures generated.')
