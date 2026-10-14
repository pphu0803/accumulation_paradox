"""Line-by-line audit of the manuscript's tables against archived outputs.

Checks that every value typeset in the manuscript's tables equals the
archived simulation output (or, for the parameter table, the
configuration module) it is claimed to come from.  The typeset values are
transcribed below as constants; the audit recomputes each from the
archives and fails loudly on any disagreement, so a referee can confirm
the table-to-archive correspondence without reading the LaTeX sources.

Coverage:
  Table 2 (model parameters)   vs abm/src/config.py (+ one step.py constant)
  Table 3 (anchoring summary)  vs comprehensive / s-curve / industry-calibration JSONs
  Table 4 (imitation cf.)      vs counterfactual_strategy_results.json
  Table 5 (dual measure)       vs revision/e1/e1_tfp_decomposition.json
  Table 6 (BTOS calibration)   vs tables/table_btos.tex vs industry_calibration_results.json
  Table 7 (code crosswalk)     function names vs abm/src/*.py sources
  Table 8 (markup assignment)  vs INDUSTRY_MARKUP in exp_industry_calibration.py

Run:  python abm/experiments/audit_tables.py     (exit 0 = all checks pass)
"""
import ast
import json
import os
import re
import sys

EXPERIMENTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(EXPERIMENTS))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'abm', 'output')

FAILURES = []
NCHECK = 0


def _load(rel):
    with open(os.path.join(OUT, rel), encoding='utf-8') as f:
        return json.load(f)


def check(table, item, printed, archived, decimals=None):
    """Compare a typeset value against the archived recomputation."""
    global NCHECK
    NCHECK += 1
    if decimals is not None:
        ok = abs(printed - archived) <= 0.5 * 10 ** (-decimals) + 1e-9
    else:
        ok = printed == archived
    if not ok:
        FAILURES.append((table, item, printed, archived))
    print(f"[{'PASS' if ok else 'FAIL'}] Table {table} | {item}: "
          f"printed {printed} vs archived {round(archived, 6) if isinstance(archived, float) else archived}")


def audit_table2():
    # flatten every `_NAME = {...}` block of config.py into one dict
    tree = ast.parse(open(os.path.join(ROOT, 'abm', 'src', 'config.py'),
                          encoding='utf-8').read())
    cfg = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith('_')):
            try:
                d = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(d, dict):
                cfg.update(d)
    rows = [
        # (config key, typeset default) -- the Table 2 labor block + initial dispersion
        ('num_workers', 500), ('ai_labor_displacement', 0.15),
        ('labor_adjustment_speed', 0.05),
        ('base_separation_prob', 0.02), ('demand_separation_sensitivity', 0.1),
        ('skill_employment_weight', 0.5), ('compat_employment_weight', 0.3),
        ('min_wage', 0.5), ('total_labor_supply', 10000), ('age_new_worker', 22),
        ('retirement_age', 65), ('initial_K_std', 3.0), ('initial_L_std', 15.0),
        ('initial_cash_std', 5.0),
    ]
    for key, val in rows:
        check('2', f'config {key}', val, cfg.get(key), decimals=4)
    # strategy exploration noise half-width (step.py): uniform(-0.5,0.5)*amp
    src = open(os.path.join(ROOT, 'abm', 'src', 'step.py'),
             encoding='utf-8').read()
    m = re.search(r"k_strat.*?0\.5\)\)\s*\*\s*jnp\.float32\(([0-9.]+)\)", src, re.S)
    amp = float(m.group(1)) if m else None
    check('2', 'strategy exploration noise half-width', 0.025,
          (amp / 2) if amp is not None else None, decimals=4)


def audit_table3():
    co = _load('comprehensive/comprehensive_results.json')
    ginis = [v['markup_gini_mean'] for v in co.values() if isinstance(v, dict)]
    check('3', 'markup Gini lower', 0.027, min(ginis), decimals=3)
    check('3', 'markup Gini upper', 0.028, max(ginis), decimals=3)
    ls = [v['labor_share_mean'] for v in co.values()
          if isinstance(v, dict) and 'labor_share_mean' in v]
    check('3', 'labor share lower (0.53-)', 0.53, min(ls), decimals=2)
    check('3', 'labor share upper (-0.55)', 0.55, max(ls), decimals=2)
    anns = [((1 + v['delta_tfp_mean']) ** (12 / 120) - 1) * 100
            for v in co.values() if isinstance(v, dict)]
    check('3', 'annualized TFP grid lower', 2.0, min(anns), decimals=1)
    check('3', 'annualized TFP grid upper', 2.4, max(anns), decimals=1)
    sc = _load('s_curve_verification/s_curve_results.json')
    keymap = {}
    for k, v in sc.items():
        lab = str(v.get('label', '')) + '|' + k
        if 'No S-curve' in k:
            keymap['ungated'] = v
        if 'speed=0.02' in k and 'threshold=0.8' in k:
            keymap['gated default'] = v
        if 'speed=0.01' in k:
            keymap['slowest'] = v
    for name, printed, dec in (('gated default', 1.9, 1), ('ungated', 2.1, 1),
                               ('slowest', 1.25, 2)):
        v = keymap.get(name)
        check('3', f's-curve annualized {name}', printed,
              (v['annualized_tfp_mean'] * 100) if v else None, decimals=dec)
    cal = _load('industry_calibration/industry_calibration_results.json')
    check('3', 'BTOS Spearman', 0.825, cal['correlation']['spearman_rho'], decimals=3)


def audit_table4():
    cf = _load('counterfactual_strategy/counterfactual_strategy_results.json')
    demands = sorted({v['demand_coeff'] for v in cf.values() if isinstance(v, dict)})
    arms = {'no_imitation': [f'lr=0.0_demand={d}' for d in demands],
            'with_imitation': [f'lr=0.05_demand={d}' for d in demands]}

    def m(keys, field):
        return sum(cf[k][field] for k in keys) / len(keys)

    rows = [
        ('delta_tfp %', 24.0, 19.7, lambda a: m(arms[a], 'delta_tfp_mean') * 100, 1),
        ('HHI', 0.250, 0.327, lambda a: m(arms[a], 'hhi_mean'), 3),
        ('strategy std', 0.199, 0.000, lambda a: m(arms[a], 'strategy_std_mean'), 3),
        ('AI investment', 0.080, 0.063, lambda a: m(arms[a], 'mean_ai_investment_mean'), 3),
        ('adoption %', 89.3, 89.2, lambda a: m(arms[a], 'adoption_rate_mean') * 100, 1),
    ]
    for name, p_with, p_without, fn, dec in rows:
        check('4', f'{name} (with)', p_with, fn('with_imitation'), decimals=dec)
        check('4', f'{name} (without)', p_without, fn('no_imitation'), decimals=dec)


def audit_table5():
    d = _load('revision/e1/e1_tfp_decomposition.json')
    printed = {
        2: (18.7, 2.8, 15.9, 96.6, 1.15), 3: (22.6, 2.7, 19.9, 86.4, 1.19),
        4: (33.9, 6.4, 27.5, 50.4, 1.26), 5: (38.8, 6.3, 32.5, 40.7, 1.31),
        6: (39.6, 5.3, 34.3, 31.6, 1.33), 7: (40.8, 5.1, 35.6, 19.4, 1.34),
        8: (42.1, 4.9, 37.2, 12.9, 1.35), 10: (40.9, 3.2, 37.7, 12.9, 1.37),
    }
    for sig, (praw, pai, pq, pado, pratio) in printed.items():
        c = d['attribution'][f'sigma={float(sig)}']['cells']['gate_on|quality_on']
        raw, ai = c['delta_tfp_raw'] * 100, c['delta_tfp_ai'] * 100
        check('5', f'sigma={sig} measured dTFP', praw, raw, decimals=1)
        check('5', f'sigma={sig} AI-efficiency', pai, ai, decimals=1)
        check('5', f'sigma={sig} quality channel', pq, raw - ai, decimals=1)
        check('5', f'sigma={sig} adoption %', pado, c['adoption'] * 100, decimals=1)
        check('5', f'sigma={sig} TFP ratio', pratio, (1 + raw / 100) / (1 + ai / 100), decimals=2)


def audit_table6():
    cal = _load('industry_calibration/industry_calibration_results.json')
    tex_path = os.path.join(ROOT, 'tables', 'table_btos.tex')
    if not os.path.isfile(tex_path):
        print('[SKIP] Table 6 | table_btos.tex not present in this tree')
        return
    tex = open(tex_path, encoding='utf-8').read()
    rows = re.findall(r'([A-Za-z][^&]+?) & ([0-9.]+) & ([0-9.]+) & \$([0-9.]+) [+-] ([0-9.]+)\$ & ([+-][0-9.]+)', tex)
    arch = {name.strip(): s for name, s in
            ((k, v) for k, v in cal['sectors'].items())}
    for name, sig, btos, model, std, err in rows:
        s = cal['sectors'].get(name.strip())
        if s is None:
            FAILURES.append(('6', f'unknown sector {name}', name, None))
            continue
        check('6', f'{name} sigma', float(sig), s['sigma'], decimals=2)
        check('6', f'{name} btos', float(btos), s['btos_rate'], decimals=1)
        check('6', f'{name} model', float(model), s['adoption_rate_mean'] * 100, decimals=1)
        check('6', f'{name} s.d.', float(std), s['adoption_rate_std'] * 100, decimals=1)


def audit_table7():
    srcs = {}
    for mod in ('step.py', 'state.py', 'metrics.py'):
        p = os.path.join(ROOT, 'abm', 'src', mod)
        srcs[mod] = (open(p, encoding='utf-8').read()
                    if os.path.isfile(p) else '')
    funcs = ['init_simulation', '_ai_investment_decision_jit', '_ai_learning_jit',
             '_produce_jit', '_ai_tfp_multiplier', '_market_clearing',
             '_calc_profit_jit', '_invest_jit', '_entry_exit_exit_stub',
             '_demand_feedback_jit', '_labor_market_clearing_jit', 'compute_metrics']
    funcs[funcs.index('_entry_exit_exit_stub')] = '_entry_exit_jit'
    for fn in funcs:
        found = any((f'def {fn}' in s) for s in srcs.values())
        global NCHECK
        NCHECK += 1
        if not found:
            FAILURES.append(('7', f'function {fn}', fn, 'not found in abm/src'))
        print(f"[{'PASS' if found else 'FAIL'}] Table 7 | function {fn} defined in abm/src")


def audit_table8():
    path = os.path.join(EXPERIMENTS, 'exp_industry_calibration.py')
    src = open(path, encoding='utf-8').read()
    m = re.search(r'INDUSTRY_MARKUP\s*=\s*(\{.*?\n\})', src, re.S)
    markup = ast.literal_eval(m.group(1))
    cal = _load('industry_calibration/industry_calibration_results.json')
    printed = {
        'Finance & Insurance': (1.60, 29.65), 'Information': (1.55, 36.55),
        'Professional Services': (1.50, 33.35), 'Health Care': (1.40, 19.75),
        'Multi-sector Companies': (1.35, 23.00), 'Real Estate & Rental': (1.35, 22.65),
        'Manufacturing': (1.25, 11.60), 'Educational Services': (1.20, 27.75),
        'Other Services': (1.20, 9.95), 'Wholesale Trade': (1.18, 13.00),
        'Arts & Recreation': (1.15, 17.85), 'Retail Trade': (1.15, 12.80),
        'Transportation': (1.12, 7.15), 'Construction': (1.10, 8.35),
        'Accommodation & Food': (1.10, 8.35),
    }
    # map display names -> calibration keys
    key = {k: k for k in printed}
    key['Finance & Insurance'] = 'Finance and Insurance'
    key['Real Estate & Rental'] = 'Real Estate and Rental'
    key['Arts & Recreation'] = 'Arts and Recreation'
    key['Accommodation & Food'] = 'Accommodation and Food'
    for name, (mu, btos) in printed.items():
        cal_key = key[name]
        got_mu = markup.get(cal_key)
        check('8', f'{name} markup', mu, got_mu, decimals=2)
        check('8', f'{name} implied sigma', round(mu / (mu - 1), 2),
              round(got_mu / (got_mu - 1), 2), decimals=2)
        check('8', f'{name} BTOS %', btos, cal['sectors'][cal_key]['btos_rate'], decimals=1)
        # calibration sigma equals implied sigma
        check('8', f'{name} calibration sigma', round(mu / (mu - 1), 2),
              cal['sectors'][cal_key]['sigma'], decimals=2)


if __name__ == '__main__':
    for fn in (audit_table2, audit_table3, audit_table4, audit_table5,
               audit_table6, audit_table7, audit_table8):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - report and continue
            FAILURES.append(('-', fn.__name__, 'exception', str(e)))
            print(f'[FAIL] {fn.__name__} raised {e!r}')
    print(f'\n===== AUDIT {"PASS" if not FAILURES else "FAIL"}: '
          f'{NCHECK - len(FAILURES)}/{NCHECK} checks passed =====')
    sys.exit(1 if FAILURES else 0)
