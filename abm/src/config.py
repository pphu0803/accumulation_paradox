"""Configuration for AI Productivity Paradox ABM v2.0.

Agent-based model investigating why micro-level AI efficiency gains
(~4% firm-level TFP increase) do not translate to macro productivity growth.

v2.0 key design principles:
- Market structure EMERGES from consumer choice + firm heterogeneity (no exogenous epsilon)
- AI investment is a continuous strategic decision (not binary adopt/not)
- Demand externality creates arms-race dynamics (Falk & Tsoukalas 2026)
- One global sigma parameter replaces 10 industry-specific epsilons

Theoretical foundations:
- AI Layoff Trap (Falk & Tsoukalas 2026): demand externality -> over-automation
- Superstar Firms (Firooz, Liu & Wang 2024): variable markups + automation
- CES monopolistic competition with Kimball-style demand elasticity
"""

_FIRM = {
    'num_firms': 200,

    # Initial capital and labor
    'initial_K_mean': 10.0,
    'initial_K_std': 3.0,
    'initial_L_mean': 50.0,
    'initial_L_std': 15.0,

    # Cobb-Douglas production
    'alpha_CD': 0.3,  # capital share (1-alpha = labor share)

    # Firm-level TFP heterogeneity
    'A_base_mean': 1.0,
    'A_base_std': 0.2,

    # Financial
    'initial_cash_mean': 20.0,
    'initial_cash_std': 5.0,
    'operating_cost_rate': 0.02,  # per-tick operating cost as fraction of revenue
    'investment_rate': 0.1,  # fraction of profit reinvested
    'depreciation_rate': 0.01,  # per-tick capital depreciation

    # Entry / exit
    'firm_entry_rate': 0.02,
    'firm_exit_cash_threshold': 0.5,
}

_CONSUMER = {
    # NOTE: the old global 'sigma' key was removed (2026-10-07 audit): it was
    # loaded into the static config but never read by any dynamics — the
    # effective elasticity is per-firm sigma_i, initialized from 'sigma_base'
    # (see state.py). Keeping a dead 'sigma' key invited the phase-diagram bug.
    'market_size': 1000.0,     # aggregate demand scale
    'quality_weight': 0.15,    # theta — how much product quality matters in demand
    'worker_mpc': 0.9,         # marginal propensity to consume (workers)
    'owner_mpc': 0.5,          # marginal propensity to consume (firm owners/profit)
}

_INDUSTRY_HETEROGENEITY = {
    'sigma_base': 3.0,               # base σ (controls markup mean)
    'sigma_firm_spread': 0.4,         # std of σ_i noise (controls markup heterogeneity)
    'sigma_firm_A_sensitivity': 0.5, # how much A_base affects σ_i
    'sigma_min': 1.5,                 # minimum σ_i (keeps markup sigma/(sigma-1) finite)
    # sigma_max raised 8.0 -> 13.0 (2026-10-07 audit): the old cap silently
    # clamped every condition with sigma_base > 8 (grid rows 9-12 and the
    # Construction/Accommodation/Transportation sectors) to an identical
    # sigma_i = 8 economy. 13.0 covers the largest grid point (12.0) plus
    # A-sensitivity and spread perturbations.
    'sigma_max': 13.0,                # maximum σ_i
}

_MARKET = {
    # Labor market (simplified for v2.0 layer 1; replaced by worker agents in layer 3).
    # NOTE: matching_efficiency / matching_elasticity / labor_supply_elasticity
    # removed 2026-10-07 (external audit B3) — they were never read by any
    # dynamics (dead parameters). The worker layer records employment flows
    # but does not feed back into firm dynamics (E9 layer ablation).
    'total_labor_supply': 10000.0,
}

_AI = {
    # AI capability (maps to firm-level TFP gain at full experience)
    'ai_efficiency_gain': 0.04,

    # Strategic AI investment (v2.0: continuous, not binary)
    'ai_max_investment_rate': 0.15,  # max investment as fraction of revenue
    'ai_fixed_cost': 5.0,            # one-time adoption cost to start investing
    'ai_maintenance_cost': 0.1,      # per-tick maintenance

    # J-curve learning dynamics
    'j_curve_depth': 0.15,
    'j_curve_length': 12,

    # AI investment -> quality/cost conversion elasticities
    'ai_quality_elasticity': 0.3,    # investment -> product quality conversion
    'ai_cost_elasticity': 0.2,      # investment -> marginal cost reduction

    # Diffusion timing. Audit 2026-10-07 (external review S2): with intro at
    # tick 51 the AI treatment landed mid-shakeout (population 200->~70 over
    # t=0..75, TFP still drifting), contaminating every measured dTFP with the
    # model's own transient. Probes on the fixed engine show the no-era
    # baseline flat (TFP ~1.02 +- 0.01, quality exactly 1.0) from t~75; intro
    # moved to 251 so the 250-tick pre-period reaches that near-steady state
    # (residual: slow entry-refill population creep, ~+10%/150 ticks, to be
    # disclosed in the paper).
    'ai_intro_tick': 251,
    'ai_spread_rate': 0.03,         # information/signaling rate

    # Strategy imitation learning (audit fix 1)
    'strategy_learning_rate': 0.05,        # speed of strategy adjustment toward successful peers
    'strategy_observation_range': 0.3,     # A_base proximity to define "peers"

    # Investment decision (audit fix 4)
    'investment_base_weight': 0.3,         # fixed component weight in base_rate
    'investment_profit_weight': 0.7,        # profitability signal weight in base_rate
    'competitive_pressure_coeff': 2.0,     # how strongly competitive gap drives investment
    'cash_investment_cap': 0.5,            # max fraction of cash spent on AI per tick

    # Quality dynamics (audit fix 5)
    'quality_depreciation': 0.005,         # natural quality decay per tick (no mean-reversion)

    # AI labor displacement (audit fix 3)
    'ai_labor_displacement': 0.15,          # how much AI reduces labor marginal product

    # Cost multiplier (audit fix 9)
    'ai_cost_experience_coeff': 0.15,       # AI experience -> marginal cost reduction

    # Experience accumulation (audit fix 10)
    'experience_investment_scale': 5.0,     # normalization factor for investment -> experience

    # Adoption readiness S-curve
    'adoption_readiness_speed': 0.02,       # per-tick natural readiness growth (info diffusion)
    'adoption_peer_influence': 0.05,        # peer adoption rate -> readiness boost
    'adoption_threshold': 0.8,              # readiness > this to start investing

    # Counterfactual switch (audit 2026-10-07): freeze ai_strategy entirely
    # (no imitation shift AND no exploration noise). Used by the strategy-lock
    # counterfactual; False in all baseline runs.
    'strategy_freeze': False,
}

_DEMAND_FEEDBACK = {
    # Keynesian demand externality (layer 2)
    'demand_shock_coefficient': 0.08,  # strength of demand externality
    'demand_recovery_rate': 0.2,       # how fast demand recovers
    'demand_shock_max': 0.5,           # hard cap on demand shock magnitude
    'demand_markup_sensitivity': 0.5,   # how demand shock reduces perceived markup (Scheme A)
    'demand_investment_sensitivity': 0.3,  # how demand shock reduces profit signal (Scheme B)
}

_LABOR = {
    # Worker agents (layer 3, added in Phase 3C).
    # matching_efficiency / matching_elasticity removed (dead, audit B3).
    'num_workers': 500,
    'skill_learning_rate': 0.01,
    'skill_depreciation_rate': 0.005,
    'min_wage': 0.5,
    'retirement_age': 65,
    'age_new_worker': 22,

    # Labor market dynamics (audit fixes 3,7,8)
    'labor_adjustment_speed': 0.05,              # L smoothing speed toward optimal
    'base_separation_prob': 0.02,                 # baseline job separation rate
    'demand_separation_sensitivity': 0.1,        # how demand shocks increase separation
    'skill_employment_weight': 0.5,              # skill weight in employment probability
    'compat_employment_weight': 0.3,             # AI compatibility weight in employment prob
}

_SIM = {
    'num_ticks': 300,
    'sample_interval': 1,
    'ticks_per_year': 12,
}


def _defaults() -> dict:
    cfg = {}
    for group in (_FIRM, _CONSUMER, _INDUSTRY_HETEROGENEITY, _MARKET, _AI, _DEMAND_FEEDBACK, _LABOR, _SIM):
        cfg.update(group)
    return cfg


def cfg(config: dict, key: str):
    return config.get(key, _defaults().get(key))


def merge_configs(*overrides: dict) -> dict:
    base = _defaults()
    for o in overrides:
        if o:
            base.update(o)
    return base
