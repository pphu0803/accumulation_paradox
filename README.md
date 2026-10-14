# The Accumulation Paradox — Code and Data Supplement (Anonymous)

Agent-based model of AI adoption under monopolistic competition with endogenous
market structure. This package reproduces every figure and table in the
revised manuscript from archived simulation output, and regenerates the output
itself from fixed seeds.

> **Engine audit (this revision).** Before finalising, the simulation engine
> was audited end-to-end (initial distributions, accounting identities,
> steady-state pre-period, measurement timing, per-seed clamping). The audit
> surfaced and fixed a number of engine-level defects; every experiment was
> then re-run from scratch under the corrected engine. The archived outputs in
> `abm/output/` are that final run, and all manuscript numbers come from it.
> The corrections are listed below.

## Firm-level panel diagnostics (E10/E11) and the BTOS sector panel

- `abm/experiments/exp_rev_e10_decomposition_panels.py` re-runs the Section 4.3/4.4
  design (8 elasticities x 50 seeds, plus g-sensitivity cells) storing firm-level
  snapshots at five horizons under `abm/output/revision/e10/panels/`. The four
  post-processing scripts (`post_e10_frozen|quality_split|index|stealing.py`)
  reproduce the frozen-roster/frozen-share counterfactual ladder, the exact
  within-efficiency/within-quality split, the aggregation-convention check, and
  the business-stealing decomposition of Sections 4.4 and 4.9; every identity
  closes to 1e-15 in float64 (residual_max recorded in each output JSON).
- `abm/experiments/exp_rev_e11_escape_times.py` produces the 30-year escape-time
  distribution of Section 4.7 (100 seeds at sigma=5).
- `abm/experiments/post_e10_adopters.py` produces
  `abm/output/revision/e10/adopter_composition.json` (adopter counts, pre-AI
  input shares, investment intensity, and adopters' expenditure share pre/post
  at the decade horizon).
- `abm/experiments/post_e10_bootstrap.py` produces
  `abm/output/revision/e10/post_e10_bootstrap.json`: percentile bootstrap
  (2,000 draws, seeds resampled within each sigma cell) 95% intervals for the
  frozen-counterfactual ladder ratios and the aggregation-convention slopes
  cited in Sections 4.4 and 4.9; point estimates reproduce the archived
  `post_e10_frozen.json` / `post_e10_index.json` summaries by construction.
- `abm/data/external/btos_sector_panel_two_wave.csv` and `btos_panel_stats.json`
  are produced by `abm/experiments/exp_btos_panel.py` from the public BTOS
  sector file (original AI question, consistent wording 2023-09--2025-10;
  Census Bureau via the Goldschlag BTOS download). The panel is truncated at
  2025-10-01: the 2025-11-26 biweek is the first fielding of the reworded
  question and is excluded. The rewording raises reported adoption by
  +3--16pp in that single biweek; the cross-section target -- the mean of
  the 2025-11-26 and 2025-12-10 waves -- sits 4--15pp above the
  pre-rewording series (the figure quoted in the manuscript). The
  raw third-party file is not redistributed here; the script documents its
  expected path.
- Post-audit note (2026-10-10): `post_e10_index.py` index (ii) previously
  divided post-AI output by PRE-AI inputs (cancelling to gross-output growth);
  it is corrected here and `post_e10_index.json` regenerated. The manuscript's
  aggregation-dependence numbers refer to the corrected index: the
  expenditure-weighted Tornqvist slope of 0.0045 collapses to one-seventh
  of the same-roster input-weighted slope (0.031) and one-fifth of the
  all-firms headline index (0.0225), in log points per unit sigma.

## Corrections relative to the original submission

All corrections below are already incorporated in this package; the archived
output is the post-correction run. Listed for transparency and provenance.

1. **Engine audit fixes.** (i) Initial firm heterogeneity: lognormal draws for
   capital/labor/cash were mis-specified and were corrected (the previous
   specification pushed 50–88% of draws into the elasticity clamp). (ii)
   Accounting: AI investment and physical reinvestment were each deducted from
   cash twice; both now deduct once. (iii) Pre-AI period: extended from 50 to
   250 ticks so that AI arrives in a demographically stable steady state.
   (iv) Measurement: quality depreciation used to run from tick 1, dragging the
   pre-AI baseline; it now applies only once AI is available. (v) Aggregate TFP
   is recomputed from the end-of-window state (recorded output mixed vintages
   and assigned zero output to last-tick entrants). (vi) Newborn firms now
   redraw their elasticity rather than inheriting the dead slot's value.
   (vii) Random-number keys for entrant traits and worker flows were split
   (previously perfectly correlated draws).
2. **Unified measurement horizon.** AI windows in the original submission were
   experiment-specific (five, ten, and 20.8 years, plus a four-year phase
   map). Every experiment now measures outcomes over a common ten-AI-year
   (120-tick) window; the affected experiments were re-run. The only
   intentional departures are the centennial (1200-tick) and thirty-year
   (360-tick) escape-time extensions used to classify the adoption-delay
   regime, plus the double-horizon (250-tick) check of the imitation
   counterfactual. `abm/experiments/exp_smoke_v31.py`
   verifies the unified constants and annualization end-to-end.
3. **Phase-diagram sweep mapping.** The original sweep varied a model input
   that does not enter the simulated dynamics (per-firm elasticities are read
   from `sigma_base`, not the global `sigma` key); the sweep now varies
   `sigma_base` at 20 seeds per cell.
4. **Reproducibility.** Import paths fixed so every script runs as delivered;
   fixed seed lists; the calibration table is typeset directly from the
   archived JSON by `generate_tables.py`; the no-imitation arm now sets
   `strategy_freeze=True` (a true lock — previously exploration noise kept
   random-walking strategies in the "locked" arm).

## Layout

```
abm/src/            model implementation (JAX)
abm/experiments/    all experiment scripts (exp_*.py) incl. revision experiments (exp_rev_*.py)
abm/data/external/  curated source CSVs (BTOS adoption, sector markups, elasticity estimates, FRED anchors, McKinsey cross-check)
abm/output/         archived simulation output (JSON) behind every manuscript artifact
generate_figures.py regenerates all manuscript figures into figures/
generate_tables.py  regenerates the industry-calibration table into tables/
figures/            manuscript figures (PDF)
tables/             manuscript table (LaTeX)
```

## Environment

- Python >= 3.10, JAX (CPU sufficient), NumPy, SciPy, Matplotlib, pandas, openpyxl
- `pip install -e .` (see `pyproject.toml`) or a conda environment with the same packages
- On non-UTF-8 locales (e.g. Chinese Windows), set `PYTHONUTF8=1` if any
  script reports a codec error

Each experiment script inserts the repository root on `sys.path` at import, so
all commands below run from the package root as-is. A full 370-tick simulation
(200 firms; 250-tick pre-AI period + 120 AI ticks) takes a few seconds on CPU
after JIT compilation. Experiment scripts run one configuration grid at a time
in a single process; if you re-run the full suite, run the scripts serially
(the scripts are sequential and single-process).

## Using the model directly

```python
from abm.src.config import merge_configs
from abm.src.state import init_simulation
from abm.src.step import _make_static_config, run_n_ticks

config = merge_configs({'sigma_base': 2.0, 'ai_efficiency_gain': 0.04})
sc = _make_static_config(config)          # static (hashable) config for JIT
state = init_simulation(config, seed=42)  # seed rule across experiments: seed_idx*1000+42
pre   = run_n_ticks(state, sc, 250)       # pre-AI steady state
final = run_n_ticks(pre,   sc, 120)       # ten-AI-year window
```

One tick is one month (12 ticks per year); AI is introduced at tick 251. All
parameters with defaults and roles live in `abm/src/config.py`; `merge_configs`
returns a copy with overrides applied. The per-tick sequence (S1–S10) is
documented in the manuscript's model section and mirrored by the functions in
`abm/src/step.py`.

## Reproducing manuscript artifacts

| Manuscript artifact | Command(s) | Manuscript file |
|---|---|---|
| Fig. 1 (imitation counterfactual) | `python abm/experiments/exp_counterfactual_strategy_lock.py` | `figures/fig5_counterfactual.pdf` |
| Fig. 2 (phase diagram) | `python abm/experiments/exp_phase_diagram.py` (env `NUM_SEEDS=20`) | `figures/fig1_phase_diagram.pdf` |
| Fig. 3 (shift-share mechanism decomposition) | `python abm/experiments/exp_rev_e2_sigma_mechanism.py` | `figures/figE2_mechanism.pdf` |
| Fig. 4 (gate x market-structure decomposition) | `python abm/experiments/exp_decomposition.py` (env `NUM_SEEDS=100`) | `figures/fig_decomposition.pdf` |
| Fig. 5 (industry calibration; model-BTOS rank correlation over time) | `python abm/experiments/exp_industry_calibration.py` (env `NUM_SEEDS=100`, panel a) + `python abm/experiments/exp_supplementary.py --only a1` (panel b) | `figures/fig2_industry_calibration.pdf` |
| Fig. 6 (adoption distributions at sigma = 2.8/5/11) | `python abm/experiments/exp_supplementary.py --only a2` | `figures/fig3_bistability.pdf` |
| Fig. 7 (markup inequality and transmission) | `python abm/experiments/exp_supplementary.py --only a3` | `figures/fig4_markup_tfp.pdf` |
| Figs. 8-9 (appendix: comprehensive grid; S-curve gating) | `python abm/experiments/exp_comprehensive.py`; `python abm/experiments/exp_s_curve_verification.py` | `figures/figA1_comprehensive.pdf`, `figures/figA2_scurve.pdf` |
| Table 6 (BTOS calibration) | `python generate_tables.py btos` | `tables/table_btos.tex` |
| Line-by-line table audit (Tables 2-8 vs archives/config/code) | `python abm/experiments/audit_tables.py` | exits 0 on full pass |
| Double-horizon check (Sec. 4.1) | `CF_POST_TICKS=250 python abm/experiments/exp_counterfactual_strategy_lock.py` | `abm/output/counterfactual_strategy/counterfactual_strategy_results_post250.json` |
| Sec. 4.3 quality-ablation numbers | `python abm/experiments/post_e1_quality_ablation.py` | `abm/output/revision/e1/quality_ablation_table.json` |
| Appendix D markup-table cross-checks | `python abm/experiments/exp_deu_sigma_robustness.py`; `python abm/experiments/post_calibration_dependence.py`; `python abm/experiments/exp_felten_crosscheck.py` (needs the public Felten et al. AIOE appendix; set `AIOE_XLSX` to its path) | `abm/output/industry_calibration/deu_sigma_robustness.json`, `calibration_dependence.json`, `felten_crosscheck.json` |
| All figures from archived output | `python generate_figures.py` | `figures/*.pdf` |
| Robustness: reversal sweep / threshold sweep / OOS | `exp_reversal_robustness.py` / `exp_threshold_robustness.py` (env `NUM_SEEDS=100`) / `exp_oos_validation.py` | --- |
| Robustness: investment regimes | `python abm/experiments/exp_rev_e3_investment_regimes.py --collect` | --- |
| Robustness: layer ablation | `python abm/experiments/exp_rev_e9_layer_toggles.py --collect` | --- |
| Robustness: mapping robustness | `python abm/experiments/exp_rev_e8_sigma_mapping.py` (`--collect` to evaluate) | --- |
| Robustness: quality-channel calibration | `python abm/experiments/exp_rev_lambdaq_sensitivity.py` | --- |
| Stationarity (centennial runs) | `python abm/experiments/exp_rev_e7_bistability_stationarity.py` | --- |
| Window-unification smoke test | `python abm/experiments/exp_smoke_v31.py` (17 checks: unified constants, per-design execution paths, annualization identity, trajectory coverage) | --- |

Figure PDFs under `figures/` are rendered from archived output by
`python generate_figures.py`; the experiment commands produce the archived
JSON/panel outputs those figures read, not the PDFs themselves.

Figure files carry stable names that need not match the manuscript's figure
numbers; the left column is authoritative. Tables 4 and 5 of the manuscript
report the archived numbers in `abm/output/counterfactual_strategy/` and
`abm/output/revision/e1/` respectively; the 20.8-year double-horizon check
quoted alongside Table 4 is archived as
`abm/output/counterfactual_strategy/counterfactual_double_horizon.json`
(the same experiment at the 250-tick horizon; regenerated under the audited
engine as `counterfactual_strategy_results_post250.json`, reproducing every
manuscript-quoted value at reporting precision -- investment 0.135 in both
arms, dTFP +35.2/+34.6 -- with unreported seed-dispersion fields differing
in the last significant digits); and the gate x market-structure
decomposition of Section 4.5 is reported in
`abm/output/robustness/decomposition_summary.json`.

Archived outputs in `abm/output/` correspond to the manuscript numbers; reruns
with the fixed seed lists reproduce them. Platform note: CPU/GPU float
differences can move individual high-variance cells by a few pp, shift
dispersion statistics (CV/CI/std) more visibly, occasionally flip a
near-threshold discrete classification (e.g. one seed's escape-time category,
or a bistability count at one sigma), and turn a last-digit tie in the table
audit — no quoted mean, interval, or rank correlation moves at reporting
precision (a full CPU re-run of every script reproduced all of them; the
shifts above were the only deviations). The experiment scripts also resume:
seeds whose outputs already exist are skipped, so on the shipped archive a
documented command may return within seconds having recomputed nothing
("... seeds already done"); delete the corresponding file under
`abm/output/` first if you want a from-scratch recomputation.
Runtime scale: a single simulation runs in seconds after JIT compilation; the
five core experiments regenerate in well under an hour (GPU) when run
serially; the reversal-robustness sweep is the longest single experiment
(hours). Regenerating every archived output serially is on the order of half a
day on GPU, longer on CPU — the archived outputs are the manuscript numbers,
so regeneration is a check, not a requirement.

## Data sources

Kept in `abm/data/external/` (all read directly or transcribed into the
calibration constants cited above):

| File | Source |
|---|---|
| `btos_ai_adoption_*.csv` | U.S. Census Bureau, Business Trends and Outlook Survey. The adoption-by-sector file carries both vintages of the AI question: `yes2wk_old` (pre-rewording waves) and `yes2wk_new` (November--December 2025 waves, the first fielded after the rewording --- the cross-section target; see the manuscript's Sec. 4.6 and Appendix D) |
| `deu_markups_by_sector_2020.csv` | firm-level sector averages from De Loecker, Eeckhout & Unger (2020); the synthesized sector-average table used for the headline mapping is reproduced in the manuscript's Appendix D, and `exp_deu_sigma_robustness.py` rebuilds the mapping from this file as a robustness check |
| `ahmad_riker_sigma_estimates.csv` | elasticity estimates from Ahmad & Riker (2020), read by `exp_rev_e8_sigma_mapping.py` |
| `fred_*.csv` | FRED (BLS labor productivity, TFP, real compensation, unit labor costs) |
| `mckinsey_*.csv` | McKinsey (2025) global adoption cross-check, transcribed exhibit data |

Not redistributed (obtain from the source): the BTOS micro-data extracts
behind the sector CSVs; firm-level markup estimates (De Loecker et al. 2020;
Bernard & Jensen 2019); the AI Occupational Exposure scores (Felten, Raj &
Seamans --- `exp_felten_crosscheck.py` reads them via the `AIOE_XLSX`
environment variable and archives the 14 sector aggregates it computes in
`felten_crosscheck.json`); CEPII ProdTrEe elasticity estimates; the published
papers cited above.

## Spot checks

- Industry calibration (100 seeds/sector): Spearman rho = 0.825 (p = 0.0002),
  bootstrap 95% CI [0.509, 0.963]; cluster CI (11 independent sigma
  conditions) [0.546, 0.963] —
  `abm/output/industry_calibration/industry_calibration_results.json`
- Gate x structure decomposition (100 seeds/cell): pooled mean gate effect
  6.2 pp, paired 95% CI [0.056, 0.068] —
  `abm/output/robustness/decomposition_summary.json` (+ raw per-seed records
  in `decomposition.json`)
- Dual-measure decomposition (50 seeds/cell): at sigma = 2, measured TFP
  +18.7% of which the AI-efficiency channel contributes +2.8% (CI
  [+2.1, +3.4]); ablation and gate arms in the same directory —
  `abm/output/revision/e1/`
- Centennial stationarity (sigma = 5): year-10-low seeds climb from 0.15 to
  0.85 mean adoption by year 20 and stay near 0.78–0.80 to year 100 —
  `abm/output/revision/e7/long_runs.json`
