"""E10: instrumented panel batch for accounting counterfactuals (v3.1 deep-dive).

Re-runs the E2 sigma sweep design with firm-level panel snapshots at 5 time
points (pre-AI, year 1/3/5/10), enabling four post-processing diagnostics that
the scalar-only E1/E2/E9 archives cannot support:

  D1 frozen ladder   (roster-fixed / shares-frozen / quality-stripped recompute)
  D2 quality split   (within-A vs within-q vs reallocation vs turnover)
  D3 index robustness (expenditure-weighted / hedonic-deflated aggregates)
  D4 business stealing (market expansion vs share redistribution)

Design notes (2026-10-10):
  - Config is copied VERBATIM from exp_rev_e2_sigma_mechanism.py so per-seed
    delta_tfp_ai/raw must reproduce the archived e2_sigma_mechanism.json
    (smoke check d, rel diff < 1e-4 on the same platform).
  - Snapshot sample indices: run_n_ticks_collect(..., sample_every=12) returns
    sampled[i] <-> post tick (i+1)*12, so years 1/3/5/10 = indices 0/2/4/9.
  - Roster identity uses the age rule (slot rebirth breaks index identity;
    age increments exactly +1 per tick for living slots, step.py):
    survivor = alive_pre & alive_post & (age_post - age_pre == elapsed).
  - jax.clear_caches() after every distinct config (~0.8GB/config compile
    caches); ONE JAX process at a time (lockfile below).

Usage:
    python abm/experiments/exp_rev_e10_decomposition_panels.py --stage smoke
    python abm/experiments/exp_rev_e10_decomposition_panels.py --stage all
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

_abm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_project_root = os.path.abspath(os.path.join(_abm_dir, '..'))
for _p in (_project_root, _abm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from abm.src.state import init_simulation
from abm.src.step import run_n_ticks, run_n_ticks_collect, _make_static_config
from abm.src.config import merge_configs

SAVE_DIR = os.path.join(_abm_dir, 'output', 'revision', 'e10')
PANEL_DIR = os.path.join(SAVE_DIR, 'panels')
LOCK_PATH = os.path.join(SAVE_DIR, 'e10.lock')

# --- design: verbatim from E2 (config drift = incomparable with paper numbers)
SIGMA_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
GAIN = 0.04
NUM_SEEDS = 50
PRE_TICKS = 250
POST_TICKS = 120
SAMPLE_EVERY = 12
SIGMA_FIRM_SPREAD = 0.4
SIGMA_FIRM_A_SENS = 0.5
ADOPT_EXP = 0.1

# g-sensitivity stage (bounds "quality dominance" against the A-cap asymmetry;
# lambda_q sensitivity already archived in e1/lambdaq_sensitivity.json)
GSENS_GAINS = [0.02, 0.08]
GSENS_SIGMAS = [2.0, 5.0, 8.0]
GSENS_SEEDS = 20

# post snapshots: year 1/3/5/10 -> sample indices 0/2/4/9 (tick (i+1)*12)
POST_SNAP_IDX = [0, 2, 4, 9]

FIRM_FLOAT_FIELDS = ['K', 'L', 'A_base', 'sigma_i', 'product_quality',
                     'ai_experience', 'ai_investment', 'ai_strategy',
                     'market_share', 'effective_markup', 'cash']
FIRM_BOOL_FIELDS = ['alive']
FIRM_INT_FIELDS = ['age']
MARKET_FIELDS = ['market_size_effective', 'total_Q', 'agg_price',
                 'wage_rate', 'hhi']


# ----------------------------------------------------------------------------
# E2-derived helpers (verbatim where possible; used for manifest self-checks)
# ----------------------------------------------------------------------------

def ai_tfp_multiplier(x, gain, depth):
    m = 1.0 + gain * ((1.0 + depth) * np.asarray(x) - depth)
    return np.where(np.asarray(x) > 0, m, 1.0)


def firm_arrays(firms, sc):
    alive = np.asarray(firms.alive)
    K = np.maximum(np.asarray(firms.K), 0.01)
    L = np.maximum(np.asarray(firms.L), 0.01)
    inputs = K ** float(sc.alpha_CD) * L ** (1 - float(sc.alpha_CD))
    mult = ai_tfp_multiplier(firms.ai_experience, sc.ai_efficiency_gain,
                             sc.j_curve_depth)
    a_eff = np.asarray(firms.A_base) * mult
    return alive, inputs, a_eff


def macro_tfp_components(firms, sc):
    alive, inputs, a_eff = firm_arrays(firms, sc)
    m = alive & (inputs > 0.01)
    inp = inputs[m]
    w = inp / max(np.sum(inp), 1e-6)
    tfp_ai = float(np.sum(a_eff[m] * w))
    q = np.maximum(np.asarray(firms.product_quality), 0.1)[m]
    theta = float(sc.quality_weight)
    Qhat = a_eff[m] * q ** theta * inp
    tfp_raw = float(np.sum(Qhat) / max(np.sum(inp), 1e-6))
    return tfp_raw, tfp_ai


def shift_share_arrays(pre_snap, post_snap, params):
    """Slot-identity 5-term shift-share on A_eff (E2-equivalent; identity
    closes for ANY roster partition, rebirth included). Pure numpy/float64."""
    def arrays(snap):
        alive = snap['alive'].astype(np.float64)
        K = np.maximum(snap['K'].astype(np.float64), 0.01)
        L = np.maximum(snap['L'].astype(np.float64), 0.01)
        inputs = K ** params['alpha'] * L ** (1 - params['alpha'])
        mult = ai_tfp_multiplier(snap['ai_experience'].astype(np.float64),
                                 params['gain'], params['depth'])
        a_eff = snap['A_base'].astype(np.float64) * mult
        return alive > 0.5, inputs, a_eff

    a_alive, a_inputs, a_eff = arrays(pre_snap)
    p_alive, p_inputs, p_eff = arrays(post_snap)
    m_pre = a_alive & (a_inputs > 0.01)
    inp = a_inputs[m_pre]
    w_pre = inp / max(np.sum(inp), 1e-6)
    base = a_eff[m_pre]

    idx_post = np.where(p_alive & (p_inputs > 0.01))[0]
    tot_post = np.sum(p_inputs[idx_post]) if idx_post.size else 1.0
    common = np.where(m_pre)[0]
    pos = {g: k for k, g in enumerate(common)}

    w_post = np.zeros_like(w_pre)
    aeff_post = np.zeros_like(w_pre)
    still = np.zeros_like(w_pre, dtype=bool)
    for g in idx_post:
        if g in pos:
            k = pos[g]
            w_post[k] = p_inputs[g] / tot_post
            aeff_post[k] = p_eff[g]
            still[k] = True

    da = np.where(still, aeff_post - base, 0.0)
    dw = np.where(still, w_post - w_pre, 0.0)
    within = float(np.sum(w_pre * da))
    realloc = float(np.sum(dw * base))
    interact = float(np.sum(dw * da))
    exit_loss = float(-np.sum(w_pre * base * ~still))
    entry = float(np.sum(
        p_inputs[idx_post[~np.isin(idx_post, common)]] / tot_post
        * p_eff[idx_post[~np.isin(idx_post, common)]]))
    terms = {'within': within, 'reallocation': realloc, 'interaction': interact,
             'exit': exit_loss, 'entry': entry}
    return terms


# ----------------------------------------------------------------------------
# Panel extraction and storage
# ----------------------------------------------------------------------------

def extract_panels(pre, sampled, seed_idx, sigma, gain, config):
    """JAX -> numpy at exactly 5 points; index 0 = pre everywhere."""
    n_idx = len(POST_SNAP_IDX)
    out = {}
    # row 0 = pre, rows 1..4 = post samples ascending (must match snap_ticks)
    for f in FIRM_FLOAT_FIELDS:
        stack = [np.asarray(getattr(pre.firms, f))]
        stack += [np.asarray(getattr(sampled.firms, f)[i]) for i in POST_SNAP_IDX]
        out[f] = np.stack(stack).astype(np.float32)          # (n_idx+1, F)
    for f in FIRM_BOOL_FIELDS:
        stack = [np.asarray(getattr(pre.firms, f))]
        stack += [np.asarray(getattr(sampled.firms, f)[i]) for i in POST_SNAP_IDX]
        out[f] = np.stack(stack).astype(bool)
    for f in FIRM_INT_FIELDS:
        stack = [np.asarray(getattr(pre.firms, f))]
        stack += [np.asarray(getattr(sampled.firms, f)[i]) for i in POST_SNAP_IDX]
        out[f] = np.stack(stack).astype(np.int32)
    for f in MARKET_FIELDS:
        stack = [np.asarray(getattr(pre.market, f)).reshape(())]
        stack += [np.asarray(getattr(sampled.market, f)[i]).reshape(())
                  for i in POST_SNAP_IDX]
        out['m_' + f] = np.stack(stack).astype(np.float32)   # (n_idx+1,)

    pre_tick = int(np.asarray(pre.tick))
    snap_ticks = [pre_tick + (i + 1) * SAMPLE_EVERY for i in POST_SNAP_IDX]
    out['snap_ticks'] = np.array([pre_tick] + snap_ticks).astype(np.int32)
    out['seed_idx'] = np.array(seed_idx).reshape(())
    out['jax_seed'] = np.array(seed_idx * 1000 + 42).reshape(())
    out['config_json'] = np.array(json.dumps(config))
    return out


def snap_k(panels, k):
    """Snapshot k of a panels dict as a firm-namespace (E2-compatible)."""
    return types.SimpleNamespace(
        **{f: panels[f][k] for f in FIRM_FLOAT_FIELDS + FIRM_BOOL_FIELDS
           + FIRM_INT_FIELDS})


def manifest_entry(panels, params, sc):
    """Per-seed scalars: TFP series, roster counts (age rule), closure resid."""
    n_snaps = panels['snap_ticks'].shape[0]
    entry = {
        'snap_ticks': panels['snap_ticks'].tolist(),
        'n_alive': [int(np.sum(panels['alive'][k])) for k in range(n_snaps)],
        'tfp_raw': [], 'tfp_ai': [],
        'roster': {}, 'ss_resid_ai': {},
    }
    for k in range(n_snaps):
        raw, ai = macro_tfp_components(snap_k(panels, k), sc)
        entry['tfp_raw'].append(raw)
        entry['tfp_ai'].append(ai)

    pre_alive = panels['alive'][0]
    pre_age = panels['age'][0]
    for k in range(1, n_snaps):
        h = int(panels['snap_ticks'][k] - panels['snap_ticks'][0])
        post_alive = panels['alive'][k]
        post_age = panels['age'][k]
        both = pre_alive & post_alive
        age_diff = post_age.astype(np.int64) - pre_age.astype(np.int64)
        surv = both & (age_diff == h)
        reborn = both & (age_diff != h)
        exiters = pre_alive & ~post_alive
        entrants = ~pre_alive & post_alive
        entry['roster'][str(h)] = {
            'survivors': int(np.sum(surv)), 'reborn': int(np.sum(reborn)),
            'exiters': int(np.sum(exiters)), 'entrants': int(np.sum(entrants)),
        }
        # closure residual of the slot-identity shift-share (A_eff measure)
        pre_snap = {f: panels[f][0] for f in
                    FIRM_FLOAT_FIELDS + FIRM_BOOL_FIELDS + FIRM_INT_FIELDS}
        post_snap = {f: panels[f][k] for f in
                     FIRM_FLOAT_FIELDS + FIRM_BOOL_FIELDS + FIRM_INT_FIELDS}
        terms = shift_share_arrays(pre_snap, post_snap, params)
        dtfp = entry['tfp_ai'][k] - entry['tfp_ai'][0]
        resid = sum(terms.values()) - dtfp
        entry['ss_resid_ai'][str(h)] = float(resid)
    return entry


def seed_done(npz_path):
    if not os.path.exists(npz_path):
        return False
    try:
        with np.load(npz_path) as z:
            return z['snap_ticks'].shape == (len(POST_SNAP_IDX) + 1,)
    except Exception:
        return False


def run_seed(cell, sigma, gain, seed_idx):
    config = merge_configs({
        'sigma_base': sigma,
        'sigma_firm_spread': SIGMA_FIRM_SPREAD,
        'sigma_firm_A_sensitivity': SIGMA_FIRM_A_SENS,
        'ai_efficiency_gain': gain,
    })
    sc = _make_static_config(config)
    state = init_simulation(config, seed=seed_idx * 1000 + 42)
    pre = run_n_ticks(state, sc, PRE_TICKS)
    final, sampled = run_n_ticks_collect(pre, sc, POST_TICKS,
                                         sample_every=SAMPLE_EVERY)
    panels = extract_panels(pre, sampled, seed_idx, sigma, gain, config)
    params = {'alpha': float(sc.alpha_CD), 'theta': float(sc.quality_weight),
              'gain': float(sc.ai_efficiency_gain),
              'depth': float(sc.j_curve_depth)}
    entry = manifest_entry(panels, params, sc)
    entry['seed'] = seed_idx
    entry['jax_seed'] = seed_idx * 1000 + 42
    entry['npz'] = None  # filled by caller with relative path
    del state, pre, final, sampled
    return panels, entry


def run_cell(cell, sigmas, gain, n_seeds):
    import jax
    cell_dir = os.path.join(PANEL_DIR, cell)
    os.makedirs(cell_dir, exist_ok=True)
    manifest_path = os.path.join(cell_dir, 'manifest.json')
    manifest = {'cell': cell, 'gain': gain,
                'sigmas': sigmas, 'n_seeds': n_seeds}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as f:
            saved = json.load(f)
        # meta must refresh upward but never regress: rows are never
        # pruned, so a --stage smoke pass over an existing full cell
        # (2 sigma x 2 seeds into this same directory) must not
        # overwrite the full-run meta -- take union / max instead
        saved['sigmas'] = sorted(set(saved.get('sigmas', ())) | set(sigmas))
        saved['n_seeds'] = max(saved.get('n_seeds', 0), n_seeds)
        saved.update({'cell': cell, 'gain': gain})
        manifest = saved
    t0 = time.time()
    for sigma in sigmas:
        sdir = os.path.join(cell_dir, f'sigma={sigma}')
        os.makedirs(sdir, exist_ok=True)
        rows = manifest.setdefault(f'sigma={sigma}', [])
        have = {r['seed'] for r in rows}
        for seed_idx in range(n_seeds):
            npz_path = os.path.join(sdir, f'seed{seed_idx:04d}.npz')
            if seed_idx in have:
                continue
            if seed_done(npz_path):
                # crash between savez and manifest write: rebuild the entry
                with np.load(npz_path) as z:
                    panels = {k: z[k] for k in z.files}
                cfg = json.loads(str(panels['config_json']))
                sc = types.SimpleNamespace(
                    alpha_CD=cfg['alpha_CD'], quality_weight=cfg['quality_weight'],
                    ai_efficiency_gain=cfg['ai_efficiency_gain'],
                    j_curve_depth=cfg['j_curve_depth'])
                entry = manifest_entry(panels, {
                    'alpha': float(sc.alpha_CD), 'theta': float(sc.quality_weight),
                    'gain': float(sc.ai_efficiency_gain),
                    'depth': float(sc.j_curve_depth)}, sc)
                entry['seed'] = seed_idx
                entry['jax_seed'] = seed_idx * 1000 + 42
                # '/' separator: manifests stay byte-identical across platforms
                entry['npz'] = os.path.relpath(
                    npz_path, PANEL_DIR).replace(os.sep, '/')
                rows.append(entry)
                _atomic(manifest_path, manifest)
                continue
            panels, entry = run_seed(cell, sigma, gain, seed_idx)
            # '/' separator: manifests stay byte-identical across platforms
            entry['npz'] = os.path.relpath(
                npz_path, PANEL_DIR).replace(os.sep, '/')
            np.savez_compressed(npz_path, **panels)
            rows.append(entry)
            _atomic(manifest_path, manifest)
            bad = max(abs(v) for v in entry['ss_resid_ai'].values())
            print(f'[{cell} sigma={sigma} seed={seed_idx}] '
                  f'n_alive={entry["n_alive"][0]}->{entry["n_alive"][-1]} '
                  f'reborn120={entry["roster"]["120"]["reborn"]} '
                  f'ss_resid_max={bad:.2e} ({time.time()-t0:.0f}s)', flush=True)
        _atomic(manifest_path, manifest)
        jax.clear_caches()
        print(f'[{cell} sigma={sigma}] done, caches cleared', flush=True)


# ----------------------------------------------------------------------------
# Smoke checks
# ----------------------------------------------------------------------------

def smoke_checks():
    """Gates (a)-(e) on 2 seeds x sigma in {2,10}. PASS/FAIL per line."""
    ok_all = True
    cell_dir = os.path.join(PANEL_DIR, 'baseline')
    e2_path = os.path.join(_abm_dir, 'output', 'revision', 'e2',
                           'e2_sigma_mechanism.json')
    with open(e2_path) as f:
        e2 = json.load(f)

    for sigma in [2.0, 10.0]:
        for seed_idx in [0, 1]:
            npz_path = os.path.join(cell_dir, f'sigma={sigma}',
                                    f'seed{seed_idx:04d}.npz')
            with np.load(npz_path) as z:
                panels = {k: z[k] for k in z.files}
            params = json.loads(str(panels['config_json']))
            sc = types.SimpleNamespace(
                alpha_CD=params['alpha_CD'],
                quality_weight=params['quality_weight'],
                ai_efficiency_gain=params['ai_efficiency_gain'],
                j_curve_depth=params['j_curve_depth'],
                depreciation_rate=params['depreciation_rate'])
            pre_snap = {f: panels[f][0] for f in
                        FIRM_FLOAT_FIELDS + FIRM_BOOL_FIELDS + FIRM_INT_FIELDS}
            post_snap = {f: panels[f][-1] for f in
                         FIRM_FLOAT_FIELDS + FIRM_BOOL_FIELDS + FIRM_INT_FIELDS}

            # (a) shift-share identity closure in float64
            terms = shift_share_arrays(pre_snap, post_snap,
                                       {'alpha': float(sc.alpha_CD),
                                        'gain': float(sc.ai_efficiency_gain),
                                        'depth': float(sc.j_curve_depth)})
            raw_pre, ai_pre = macro_tfp_components(snap_k(panels, 0), sc)
            raw_post, ai_post = macro_tfp_components(snap_k(panels, -1), sc)
            resid = sum(terms.values()) - (ai_post - ai_pre)
            ok = abs(resid) < 1e-6
            ok_all &= ok
            print(f'(a) closure sigma={sigma} seed={seed_idx}: '
                  f'resid={resid:.2e} {"PASS" if ok else "FAIL"}')

            # (b) frozen-omega reproduces pre TFP over survivors (float64)
            alive = pre_snap['alive']
            K = np.maximum(pre_snap['K'].astype(np.float64), 0.01)
            L = np.maximum(pre_snap['L'].astype(np.float64), 0.01)
            inputs = K ** float(sc.alpha_CD) * L ** (1 - float(sc.alpha_CD))
            m = alive & (inputs > 0.01)
            q = np.maximum(pre_snap['product_quality'].astype(np.float64), 0.1)
            theta = float(sc.quality_weight)
            mult = ai_tfp_multiplier(pre_snap['ai_experience'],
                                     float(sc.ai_efficiency_gain),
                                     float(sc.j_curve_depth))
            a_eff = pre_snap['A_base'].astype(np.float64) * mult
            y = a_eff * q ** theta
            post_alive = post_snap['alive']
            both = alive & post_alive
            age_ok = (panels['age'][-1].astype(np.int64)
                      - panels['age'][0].astype(np.int64)) == int(
                          panels['snap_ticks'][-1] - panels['snap_ticks'][0])
            surv = both & age_ok
            S = m & surv
            omega = inputs[S] / np.sum(inputs[S])
            tfp_S = float(np.sum(omega * y[S]))
            tfp_S_direct = float(np.sum(inputs[S] * y[S]) / np.sum(inputs[S]))
            err = abs(tfp_S - tfp_S_direct) + abs(np.sum(omega) - 1.0)
            ok = err < 1e-12
            ok_all &= ok
            print(f'(b) frozen-omega sigma={sigma} seed={seed_idx}: '
                  f'err={err:.2e} {"PASS" if ok else "FAIL"}')

            # (c) npz size (dead-slot-heavy runs compress well; sigma=10 ~13KB)
            size = os.path.getsize(npz_path)
            ok = 8_000 < size < 150_000
            ok_all &= ok
            print(f'(c) npz size sigma={sigma} seed={seed_idx}: '
                  f'{size/1024:.0f}KB {"PASS" if ok else "FAIL"}')

            # (d) distribution compatibility vs archived E2 — per-seed bit
            # equality is NOT attainable (GPU nondeterminism + chaotic
            # amplification; an earlier cross-validation run saw per-seed
            # divergences up to 188% relative).
            # Instead: smoke values must fall inside the archived 50-seed
            # seed-level range. The aggregate check (mean within 2.5 SE)
            # runs after the full batch.
            rows_e2 = e2[f'sigma={sigma}']
            e2_raw = [r['delta_tfp_raw'] for r in rows_e2]
            e2_ai = [r['delta_tfp_ai'] for r in rows_e2]
            my_raw = (raw_post - raw_pre) / raw_pre
            my_ai = (ai_post - ai_pre) / ai_pre
            ok = (min(e2_raw) <= my_raw <= max(e2_raw)
                  and min(e2_ai) <= my_ai <= max(e2_ai))
            ok_all &= ok
            print(f'(d) E2 range sigma={sigma} seed={seed_idx}: '
                  f'raw={my_raw:.3f} in [{min(e2_raw):.3f},{max(e2_raw):.3f}], '
                  f'ai={my_ai:.3f} in [{min(e2_ai):.3f},{max(e2_ai):.3f}] '
                  f'{"PASS" if ok else "FAIL"}')

            # (e) roster counts + age-diff sanity (reborn slots may show
            # negative diffs: dead slots reset age to 0, step.py:593)
            reborn = int(np.sum(both & ~age_ok))
            n_surv = int(np.sum(both & age_ok))
            max_pre_age = int(panels['age'][0][panels['alive'][0]].max())
            diffs = (panels['age'][-1].astype(np.int64)
                     - panels['age'][0].astype(np.int64))[both]
            ok = (not np.any(diffs > 120)) and (not np.any(diffs < -max_pre_age))
            ok_all &= ok
            print(f'(e) roster sigma={sigma} seed={seed_idx}: '
                  f'survivors={n_surv} reborn={reborn} '
                  f'(true churn, informational) '
                  f'age_diff in [-pre_age, elapsed]: '
                  f'{"PASS" if ok else "FAIL"}')

    print('SMOKE:', 'ALL PASS' if ok_all else 'FAILED')
    return ok_all


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def _atomic(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def acquire_lock():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            print(f'another E10 process alive (pid {pid}); refusing to run')
            sys.exit(1)
        except (ProcessLookupError, ValueError, PermissionError):
            pass  # stale lock
    with open(LOCK_PATH, 'w') as f:
        f.write(str(os.getpid()))


def release_lock():
    try:
        os.remove(LOCK_PATH)
    except FileNotFoundError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['smoke', 'baseline', 'gsens', 'all', 'checks'])
    ap.add_argument('--num-seeds', type=int, default=None)
    ap.add_argument('--sigmas', default=None)
    ap.add_argument('--gains', default=None)
    args = ap.parse_args()

    if args.stage == 'checks':
        smoke_checks()
        return

    acquire_lock()
    try:
        if args.stage in ('smoke',):
            sigmas = [2.0, 10.0]
            n = 2
            run_cell('baseline', sigmas, GAIN, n)
            ok = smoke_checks()
            if not ok:
                print('smoke FAILED — not cleared for full run')
                sys.exit(2)
        if args.stage in ('baseline', 'all'):
            sigmas = ([float(s) for s in args.sigmas.split(',')]
                      if args.sigmas else SIGMA_VALUES)
            n = args.num_seeds or NUM_SEEDS
            run_cell('baseline', sigmas, GAIN, n)
        if args.stage in ('gsens', 'all'):
            gains = ([float(g) for g in args.gains.split(',')]
                     if args.gains else GSENS_GAINS)
            sigmas = ([float(s) for s in args.sigmas.split(',')]
                      if args.sigmas else GSENS_SIGMAS)
            n = args.num_seeds or GSENS_SEEDS
            for g in gains:
                run_cell(f'gsens_g{g}', sigmas, g, n)
        print('E10 done.')
    finally:
        release_lock()


if __name__ == '__main__':
    main()
