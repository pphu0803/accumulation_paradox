"""v3.1 smoke test: verify AI-window unification (120 ticks) end-to-end.

Checks (2026-10-09, v3.1 engineering):
  1. The five affected experiment scripts carry the unified constants
     (parse check on source text).
  2. Each affected design runs end-to-end through its real code path on a
     trimmed grid (2 seeds / 1 cell), with the annualization formula using
     12/120 wherever an annualized rate is computed.
  3. analysis3 (locals not patchable from outside) is verified by replicating
     its core loop inline with the new constant.

All trimmed runs write to a throwaway scratch directory (created under the
system temp dir, or $SMOKE_SCRATCH if set, and printed at start), so the
archived results in abm/output are never overwritten by the smoke test.

Usage: python abm/experiments/exp_smoke_v31.py
"""
import json
import os
import re
import sys
import tempfile

_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_abm_dir, '..'))
for _p in (_project_root, _abm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SCRATCH = os.environ.get('SMOKE_SCRATCH') or tempfile.mkdtemp(prefix='smoke_v31_')


def _redirect(m):
    """Quarantine a trimmed run's outputs in the scratch dir so the archived
    results in abm/output stay untouched."""
    m.SAVE_DIR = os.path.join(_SCRATCH, os.path.basename(m.__file__)[:-3])
    os.makedirs(m.SAVE_DIR, exist_ok=True)


CHECKS = []


def check(name, ok, detail=''):
    CHECKS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))


def source_checks():
    print("== 1. source-constant checks ==")
    specs = [
        ('exp_comprehensive.py', r'POST_TICKS = 120'),
        ('exp_counterfactual_strategy_lock.py',
         r"POST_TICKS = int\(os\.environ\.get\('CF_POST_TICKS', 120\)"),
        ('exp_s_curve_verification.py', r'AI_TICKS = 120'),
        ('exp_phase_diagram.py', r'NUM_TICKS = 370'),
        ('exp_supplementary.py', r'POST_TICKS = 120  # v3\.1'),
    ]
    for fn, pat in specs:
        src = open(os.path.join(_abm_dir, 'experiments', fn),
                  encoding='utf-8').read()
        check(f"{fn} unified constant", bool(re.search(pat, src)), pat)
    # no stray 250/60 windows left in the five scripts
    for fn, pats in [
        ('exp_comprehensive.py', [r'POST_TICKS = 250']),
        ('exp_counterfactual_strategy_lock.py', [r'POST_TICKS = 250']),
        ('exp_s_curve_verification.py', [r'AI_TICKS = 60']),
        ('exp_phase_diagram.py', [r'NUM_TICKS = 300']),
    ]:
        src = open(os.path.join(_abm_dir, 'experiments', fn),
                  encoding='utf-8').read()
        left = [p for p in pats if re.search(p, src)]
        check(f"{fn} no stale window", not left, str(left))


def _annualized(delta, ticks=120):
    return (1.0 + delta) ** (12.0 / ticks) - 1.0


def run_comprehensive():
    print("== 2. comprehensive (trimmed: 1 gain x 1 demand x 2 seeds) ==")
    import importlib
    import exp_comprehensive as m
    importlib.reload(m)
    _redirect(m)
    m.AI_GAINS = [0.04]
    m.DEMAND_COEFFS = [0.08]
    m.NUM_SEEDS = 2
    m.main()
    out = json.load(open(os.path.join(m.SAVE_DIR, 'comprehensive_results.json'),
                        encoding='utf-8'))
    (key, cell), = out.items()
    dt = cell['delta_tfp_mean']
    an = cell['annualized_tfp_mean']
    check("comprehensive runs", True, f"{key} ΔTFP={dt*100:+.1f}% ann={an*100:+.2f}%/yr")
    check("annualized uses 12/120",
          abs(an - _annualized(dt)) < 1e-6,
          f"json {an:.6f} vs formula {_annualized(dt):.6f}")


def run_counterfactual():
    print("== 3. counterfactual (trimmed: 1 cell x 2 seeds) ==")
    import importlib
    import exp_counterfactual_strategy_lock as m
    importlib.reload(m)
    _redirect(m)
    m.STRATEGY_RATES = [0.0, 0.05]  # summary section hardcodes both arms
    m.DEMAND_COEFFS = [0.0]
    m.NUM_SEEDS = 2
    m.main()
    out = json.load(open(os.path.join(m.SAVE_DIR, 'counterfactual_strategy_results.json'),
                        encoding='utf-8'))
    key, cell = next(iter(out.items()))
    check("counterfactual runs", True,
          f"{key} ΔTFP={cell['delta_tfp_mean']*100:+.1f}% HHI={cell['hhi_mean']:.3f} adopt={cell['adoption_rate_mean']*100:.0f}%")


def run_s_curve():
    print("== 4. s-curve (trimmed: baseline only, 2 seeds) ==")
    import importlib
    import exp_s_curve_verification as m
    importlib.reload(m)
    _redirect(m)
    m.NUM_SEEDS = 2
    # keep only the ungated baseline + default gate configs
    m.PARAM_GRID = [c for c in m.PARAM_GRID
                    if c['label'] == 'No S-curve (baseline)' or
                       (c['adoption_readiness_speed'] == 0.02 and
                        c['adoption_threshold'] == 0.8)]
    m.main()
    out = json.load(open(os.path.join(m.SAVE_DIR, 's_curve_results.json'),
                        encoding='utf-8'))
    base = next(v for k, v in out.items() if 'baseline' in k)
    gated = next(v for k, v in out.items() if 'speed=0.02' in k)
    check("s-curve runs", True,
          f"ungated ann={base['annualized_tfp_mean']*100:+.2f}%/yr, gated ann={gated['annualized_tfp_mean']*100:+.2f}%/yr")
    # script annualizes per-seed then averages; mean-of-power != power-of-mean
    # (Jensen), so allow a small gap that shrinks with seed count
    check("s-curve annualized consistent with 12/120 (within per-seed Jensen gap)",
          abs(base['annualized_tfp_mean'] - _annualized(base['delta_tfp_mean'])) < 1e-4,
          f"{base['annualized_tfp_mean']:.6f} vs {_annualized(base['delta_tfp_mean']):.6f}")


def run_phase():
    print("== 5. phase diagram (trimmed: 1 sigma x 1 gain x 2 seeds) ==")
    import importlib
    import exp_phase_diagram as m
    importlib.reload(m)
    _redirect(m)
    m.SIGMA_VALS = [2.0]
    m.GAIN_VALS = [0.04]
    m.NUM_SEEDS = 2
    m.NUM_TICKS = 370
    m.main()  # writes raw json; figure step skipped (not called)
    raw = json.load(open(os.path.join(m.SAVE_DIR, 'phase_diagram_raw.json'),
                         encoding='utf-8'))
    (key, agg), = raw.items()
    check("phase runs at 370 ticks", True,
          f"{key} ΔTFP={agg['macro_tfp_delta_mean']*100:+.1f}% paradox={agg['paradox_rate']:.0%}")
    # trajectory length: run_with_trajectory gives n_blocks = (370-250)/2 = 60
    traj, pre = m.run_with_trajectory(
        m.merge_configs({'sigma_base': 2.0, 'ai_efficiency_gain': 0.04}),
        42, m.NUM_TICKS, m.SAMPLE_EVERY)
    check("phase trajectory covers 120 AI ticks (60 blocks, last tick 370)",
          len(traj) == 60 and traj[-1]['tick'] == 370,
          f"n={len(traj)} last_tick={traj[-1]['tick']}")


def run_analysis3_inline():
    print("== 6. analysis3 core loop replicated inline (spread=0.4, g=0.04, 2 seeds) ==")
    import numpy as np
    import jax.numpy as jnp
    from abm.src.state import init_simulation
    from abm.src.step import run_n_ticks, _make_static_config
    from abm.src.config import merge_configs

    PRE, POST = 250, 120
    tfps = []
    for seed_idx in range(2):
        config = merge_configs({'sigma_firm_spread': 0.4, 'ai_efficiency_gain': 0.04})
        sc = _make_static_config(config)
        state = init_simulation(config, seed=seed_idx * 1000 + 42)
        pre = run_n_ticks(state, sc, PRE)
        final = run_n_ticks(pre, sc, POST)
        alpha = float(sc.alpha_CD)
        alive, alive_pre = final.firms.alive, pre.firms.alive
        K, L = jnp.maximum(final.firms.K, 0.01), jnp.maximum(final.firms.L, 0.01)
        eff = jnp.power(K, alpha) * jnp.power(L, 1.0 - alpha)
        post_tfp = float(jnp.sum(final.firms.Q[alive]) / max(float(jnp.sum(eff[alive])), 0.01))
        K0, L0 = jnp.maximum(pre.firms.K, 0.01), jnp.maximum(pre.firms.L, 0.01)
        eff0 = jnp.power(K0, alpha) * jnp.power(L0, 1.0 - alpha)
        pre_tfp = float(jnp.sum(pre.firms.Q[alive_pre]) / max(float(jnp.sum(eff0[alive_pre])), 0.01))
        tfps.append((post_tfp - pre_tfp) / pre_tfp)
    dt = float(np.mean(tfps))
    ann = (1 + dt) ** (12 / POST) - 1
    check("analysis3 core runs (same aggregation as exp_supplementary)",
          np.isfinite(dt), f"ΔTFP={dt*100:+.1f}% ann(12/120)={ann*100:+.2f}%/yr")


def main():
    print(f'Scratch dir (all trimmed-run outputs): {_SCRATCH}')
    source_checks()
    run_comprehensive()
    run_counterfactual()
    run_s_curve()
    run_phase()
    run_analysis3_inline()

    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print(f"\n===== SMOKE {'PASS' if n_fail == 0 else 'FAIL'}: "
          f"{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed =====")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
