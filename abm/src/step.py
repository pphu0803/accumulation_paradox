"""Core simulation step for AI Productivity Paradox ABM v2.0.

Each tick:
  S1: AI investment decision (strategic, continuous, with demand externality awareness)
  S2: AI learning curve (experience + quality/cost accumulation with J-curve)
  S3: Production (Cobb-Douglas + product quality + AI augmentation)
  S4: Market clearing (CES consumer choice -> endogenous markup + HHI)
  S5: Profit calculation and financial settlement
  S6: Investment / disinvestment (capital adjustment)
  S7: Entry / exit
  S8: Demand feedback (Keynesian externality -> next period market size)

v2.0 changes from v0.2:
  - No exogenous epsilon per industry. Market structure emerges from CES consumer choice.
  - AI investment is continuous (not binary adopt/not).
  - Markup is endogenous: depends on firm's market share via Kimball demand elasticity.
  - Demand externality: layoffs reduce aggregate demand, creating arms-race dynamics.
"""

import types
import math

import jax
import jax.numpy as jnp

from .state import FirmState, MarketState, WorkerState, SimState
from .config import cfg as _cfg


class _FrozenConfig(types.SimpleNamespace):
    """Hashable namespace for static JIT config."""
    def __hash__(self):
        import numpy as np
        def _make_hashable(v):
            if hasattr(v, 'shape'):
                if v.ndim == 0:
                    return ('scalar', float(v), v.dtype.str)
                else:
                    return ('array', v.shape, v.dtype.str, tuple(v.tolist()))
            if v is None:
                return None
            return v
        items = tuple(sorted((k, _make_hashable(v))
                             for k, v in vars(self).items()))
        return hash(items)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _make_static_config(config):
    return _FrozenConfig(
        num_firms=_cfg(config, 'num_firms'),
        num_workers=_cfg(config, 'num_workers'),
        alpha_CD=_cfg(config, 'alpha_CD'),
        # NOTE: global 'sigma' removed (dead key, 2026-10-07) — dynamics use
        # per-firm firms.sigma_i only.
        quality_weight=_cfg(config, 'quality_weight'),
        market_size=_cfg(config, 'market_size'),
        worker_mpc=_cfg(config, 'worker_mpc'),
        owner_mpc=_cfg(config, 'owner_mpc'),
        total_labor_supply=_cfg(config, 'total_labor_supply'),
        # sigma_i re-draw parameters for entrants (S7) — audit 2026-10-07:
        # newborn slots used to inherit the dead firm's sigma_i while drawing
        # a fresh A_base, breaking the sigma_i = f(A_base) structure.
        sigma_base=_cfg(config, 'sigma_base'),
        sigma_firm_spread=_cfg(config, 'sigma_firm_spread'),
        sigma_firm_A_sensitivity=_cfg(config, 'sigma_firm_A_sensitivity'),
        sigma_min=_cfg(config, 'sigma_min'),
        sigma_max=_cfg(config, 'sigma_max'),
        # NOTE (audit 2026-10-07, external review B3): matching_efficiency,
        # matching_elasticity and labor_supply_elasticity were loaded here but
        # never read by any dynamics — dead parameters, removed. The worker
        # layer records employment flows; it does not feed back into firm
        # dynamics (verified twice, see E9 layer ablation).
        ai_efficiency_gain=_cfg(config, 'ai_efficiency_gain'),
        ai_max_investment_rate=_cfg(config, 'ai_max_investment_rate'),
        ai_fixed_cost=_cfg(config, 'ai_fixed_cost'),
        ai_maintenance_cost=_cfg(config, 'ai_maintenance_cost'),
        j_curve_depth=_cfg(config, 'j_curve_depth'),
        j_curve_length=_cfg(config, 'j_curve_length'),
        ai_quality_elasticity=_cfg(config, 'ai_quality_elasticity'),
        ai_cost_elasticity=_cfg(config, 'ai_cost_elasticity'),
        ai_spread_rate=_cfg(config, 'ai_spread_rate'),
        ai_intro_tick=_cfg(config, 'ai_intro_tick'),
        operating_cost_rate=_cfg(config, 'operating_cost_rate'),
        investment_rate=_cfg(config, 'investment_rate'),
        depreciation_rate=_cfg(config, 'depreciation_rate'),
        firm_entry_rate=_cfg(config, 'firm_entry_rate'),
        firm_exit_cash_threshold=_cfg(config, 'firm_exit_cash_threshold'),
        initial_K_mean=_cfg(config, 'initial_K_mean'),
        initial_K_std=_cfg(config, 'initial_K_std'),
        initial_L_mean=_cfg(config, 'initial_L_mean'),
        initial_L_std=_cfg(config, 'initial_L_std'),
        initial_cash_mean=_cfg(config, 'initial_cash_mean'),
        initial_cash_std=_cfg(config, 'initial_cash_std'),
        A_base_mean=_cfg(config, 'A_base_mean'),
        A_base_std=_cfg(config, 'A_base_std'),
        demand_shock_coefficient=_cfg(config, 'demand_shock_coefficient'),
        demand_recovery_rate=_cfg(config, 'demand_recovery_rate'),
        demand_shock_max=_cfg(config, 'demand_shock_max'),
        demand_markup_sensitivity=_cfg(config, 'demand_markup_sensitivity'),
        demand_investment_sensitivity=_cfg(config, 'demand_investment_sensitivity'),
        skill_learning_rate=_cfg(config, 'skill_learning_rate'),
        skill_depreciation_rate=_cfg(config, 'skill_depreciation_rate'),
        min_wage=_cfg(config, 'min_wage'),
        retirement_age=_cfg(config, 'retirement_age'),
        age_new_worker=_cfg(config, 'age_new_worker'),
        strategy_learning_rate=_cfg(config, 'strategy_learning_rate'),
        strategy_observation_range=_cfg(config, 'strategy_observation_range'),
        investment_base_weight=_cfg(config, 'investment_base_weight'),
        investment_profit_weight=_cfg(config, 'investment_profit_weight'),
        competitive_pressure_coeff=_cfg(config, 'competitive_pressure_coeff'),
        cash_investment_cap=_cfg(config, 'cash_investment_cap'),
        quality_depreciation=_cfg(config, 'quality_depreciation'),
        ai_labor_displacement=_cfg(config, 'ai_labor_displacement'),
        ai_cost_experience_coeff=_cfg(config, 'ai_cost_experience_coeff'),
        experience_investment_scale=_cfg(config, 'experience_investment_scale'),
        labor_adjustment_speed=_cfg(config, 'labor_adjustment_speed'),
        base_separation_prob=_cfg(config, 'base_separation_prob'),
        demand_separation_sensitivity=_cfg(config, 'demand_separation_sensitivity'),
        skill_employment_weight=_cfg(config, 'skill_employment_weight'),
        compat_employment_weight=_cfg(config, 'compat_employment_weight'),
        adoption_readiness_speed=_cfg(config, 'adoption_readiness_speed'),
        adoption_peer_influence=_cfg(config, 'adoption_peer_influence'),
        adoption_threshold=_cfg(config, 'adoption_threshold'),
        strategy_freeze=bool(_cfg(config, 'strategy_freeze')),
    )


# ============================================================
# S1: AI Investment Decision (strategic, continuous)
# ============================================================

def _ai_investment_decision_jit(firms, market, sc, key, tick, F):
    """Firms decide AI investment intensity via social learning.

    Investment rate: driven by profitability signal + competitive pressure,
    gated by adoption readiness (S-curve diffusion).
    No demand externality awareness — firms don't understand the macro
    externality (this is the arms-race premise from Falk & Tsoukalas 2026).

    Strategy choice: imitation learning — firms observe which strategy
    (quality vs cost) is more profitable among similar peers and slowly
    adjust. No preset rule about firm size determining strategy.
    """
    k_innov, k_strat = jax.random.split(key, 2)
    ai_available = tick >= sc.ai_intro_tick

    alive = firms.alive
    has_invested_before = firms.ai_experience > 0

    # --- Adoption readiness update (S-curve mechanism) ---
    # Natural diffusion: readiness grows slowly each tick after AI introduction
    readiness_growth = jnp.float32(sc.adoption_readiness_speed)
    # Peer influence: seeing other firms adopt boosts readiness
    n_alive = jnp.maximum(jnp.sum(alive.astype(jnp.float32)), 1.0)
    peer_adoption_rate = jnp.sum((firms.ai_experience > jnp.float32(0.1)) & alive) / n_alive
    readiness_growth = readiness_growth + jnp.float32(sc.adoption_peer_influence) * peer_adoption_rate
    # Only grow after AI introduction, cap at 1.0
    new_readiness = jnp.where(
        ai_available,
        jnp.clip(firms.adoption_readiness + readiness_growth, jnp.float32(0.0), jnp.float32(1.0)),
        firms.adoption_readiness)

    # Innovator seeding: random firm starts investing at ai_intro_tick
    should_seed = (tick == sc.ai_intro_tick) & ai_available
    noise = jax.random.uniform(k_innov, (F,), dtype=jnp.float32)
    masked = jnp.where(alive, noise, jnp.float32(-1.0))
    innovator = jnp.argmax(masked)
    just_started = jnp.where(
        should_seed,
        jnp.where(jnp.arange(F) == innovator, True, False),
        jnp.zeros(F, dtype=jnp.bool_))

    # Profitability signal for investment willingness
    n_investing = jnp.sum((firms.ai_experience > 0) & alive).astype(jnp.float32)
    n_profitable = jnp.sum(
        (firms.ai_experience > 0) & alive & (firms.profit > 0)).astype(jnp.float32)
    profitable_frac = jnp.where(
        n_investing > 0, n_profitable / jnp.maximum(n_investing, 1.0), jnp.float32(0.0))

    # Competitive pressure: invest more when behind competitors
    avg_ai_exp = jnp.where(
        n_investing > 0,
        jnp.sum(firms.ai_experience * alive) / jnp.maximum(jnp.sum(alive.astype(jnp.float32)), 1.0),
        jnp.float32(0.0))
    competitive_gap = jnp.maximum(avg_ai_exp - firms.ai_experience, jnp.float32(0.0))

    # --- Investment rate ---
    base_rate = sc.ai_spread_rate * (
        jnp.float32(sc.investment_base_weight)
        + jnp.float32(sc.investment_profit_weight) * profitable_frac)
    pressure_multiplier = jnp.float32(1.0) + jnp.float32(sc.competitive_pressure_coeff) * competitive_gap

    # Variant B: demand shock amplifies investment pressure (paradox mechanism)
    # When demand falls, firms perceive lower margins and invest more to cut costs
    demand_invest_adj = jnp.float32(1.0) + market.demand_externalit_shock * jnp.float32(sc.demand_investment_sensitivity)
    pressure_multiplier = pressure_multiplier * demand_invest_adj

    # Cash constraint: cap by revenue fraction and cash reserve
    max_invest = jnp.minimum(
        sc.ai_max_investment_rate * jnp.maximum(firms.revenue, jnp.float32(0.01)),
        firms.cash * jnp.float32(sc.cash_investment_cap))

    desired_rate = jnp.minimum(base_rate * pressure_multiplier, sc.ai_max_investment_rate)
    desired_rate = jnp.where(alive & ai_available, desired_rate, jnp.float32(0.0))

    # Readiness gate: must exceed threshold to invest (innovator exempt)
    ready = new_readiness >= jnp.float32(sc.adoption_threshold)
    can_invest = ((firms.cash > sc.ai_fixed_cost) & alive & ready | just_started)
    can_invest = (can_invest | (has_invested_before & alive)) & ai_available

    desired_investment = desired_rate * jnp.maximum(firms.revenue, jnp.float32(1.0))
    investment = jnp.where(
        can_invest,
        jnp.minimum(desired_investment, max_invest),
        jnp.float32(0.0))

    cash_after_fixed = jnp.where(
        just_started & alive,
        firms.cash - sc.ai_fixed_cost,
        firms.cash)

    # --- Strategy: imitation learning among peers ---
    # For each firm i, find peers with similar A_base (|A_j - A_i| < range)
    # Compare profitability of quality-focused vs cost-focused peers
    obs_range = jnp.float32(sc.strategy_observation_range)
    lr = jnp.float32(sc.strategy_learning_rate)

    # Peer mask: alive firms with similar A_base (broadcast A_base_i across F×F)
    a_diff = jnp.abs(firms.A_base[:, None] - firms.A_base[None, :])  # [F, F]
    peer_mask = (a_diff < obs_range) & alive[None, :] & (firms.ai_experience[None, :] > jnp.float32(0.1))
    # Exclude self
    self_mask = jnp.eye(F, dtype=jnp.bool_)
    peer_mask = peer_mask & ~self_mask

    # Quality-focused peers (strategy > 0.5) vs cost-focused (strategy <= 0.5)
    quality_peer = peer_mask & (firms.ai_strategy[None, :] > jnp.float32(0.5))
    cost_peer = peer_mask & (firms.ai_strategy[None, :] <= jnp.float32(0.5))

    # Average profit rate of quality-focused and cost-focused peers
    peer_revenue = jnp.where(alive, firms.revenue, jnp.float32(1.0))  # avoid div-by-zero
    profit_rate = firms.profit / jnp.maximum(peer_revenue, jnp.float32(0.01))

    n_quality_peers = jnp.sum(quality_peer.astype(jnp.float32), axis=1)
    n_cost_peers = jnp.sum(cost_peer.astype(jnp.float32), axis=1)

    quality_peer_profit = jnp.where(
        n_quality_peers > 0,
        jnp.sum(quality_peer.astype(jnp.float32) * profit_rate[None, :], axis=1) / jnp.maximum(n_quality_peers, 1.0),
        jnp.float32(0.0))
    cost_peer_profit = jnp.where(
        n_cost_peers > 0,
        jnp.sum(cost_peer.astype(jnp.float32) * profit_rate[None, :], axis=1) / jnp.maximum(n_cost_peers, 1.0),
        jnp.float32(0.0))

    # Profit signal: positive = quality peers more profitable → shift toward quality
    denom = jnp.maximum(jnp.abs(quality_peer_profit) + jnp.abs(cost_peer_profit), jnp.float32(0.01))
    profit_signal = (quality_peer_profit - cost_peer_profit) / denom

    # Only adjust if we have enough peers to learn from
    has_peers = (n_quality_peers + n_cost_peers) > 2
    strategy_shift = lr * jnp.where(has_peers, profit_signal, jnp.float32(0.0))

    # Add exploration noise; under strategy_freeze both the imitation shift and
    # the noise are suppressed (true lock for the counterfactual experiment).
    if sc.strategy_freeze:
        new_strategy = firms.ai_strategy
    else:
        strat_noise = (jax.random.uniform(k_strat, (F,), dtype=jnp.float32) - jnp.float32(0.5)) * jnp.float32(0.05)
        new_strategy = jnp.clip(firms.ai_strategy + strategy_shift + strat_noise, jnp.float32(0.0), jnp.float32(1.0))
    new_strategy = jnp.where(can_invest, new_strategy, firms.ai_strategy)

    # Cash ledger (audit 2026-10-07): investment and maintenance are paid ONCE,
    # through the S5 profit settlement (profit already nets out r_i + m_c).
    # S1 only books the one-time adoption fixed cost here; S2 no longer touches
    # cash. The old code deducted investment here AND in S5 (double count).
    return firms._replace(
        adoption_readiness=new_readiness,
        ai_investment=investment,
        ai_strategy=new_strategy,
        cash=jnp.maximum(cash_after_fixed, jnp.float32(0.0)))


# ============================================================
# S2: AI Learning Curve (quality + cost accumulation with J-curve)
# ============================================================

def _ai_learning_jit(firms, sc, tick):
    """AI experience accumulates linearly in normalized investment.

    Audit note 2026-10-07 (external review B6): experience accumulates
    LINEARLY (new_exp += invest_norm / tau), not as 1 - exp(-t/tau); the
    J-curve productivity dip in the multiplier m(x) has depth g*d (~0.6%) and
    lasts < 1 tick at typical investment rates — a bookkeeping dip, not a
    sustained learning curve.

    Quality: additive gain + natural depreciation. Audit 2026-10-07 (external
    review S3): depreciation applies only once AI is available — it used to
    run from tick 1, dragging the PRE-AI baseline quality from 1.0 to 0.78 by
    t=50 (and to the 0.5 clip floor in the no-AI steady state), so part of
    the measured "AI quality gain" was recovery from self-inflicted decay and
    the pre-AI baseline was non-stationary.
    """
    tau = jnp.float32(sc.j_curve_length) / jnp.float32(3.0)
    invest_norm = jnp.minimum(firms.ai_investment / jnp.float32(sc.experience_investment_scale), jnp.float32(1.0))
    new_exp = firms.ai_experience + invest_norm / tau
    new_exp = jnp.minimum(new_exp, jnp.float32(1.0))

    # Product quality: additive gain + natural depreciation (no forced reversion to 1.0)
    quality_gain = firms.ai_investment * firms.ai_strategy * jnp.float32(sc.ai_quality_elasticity)
    dep = jnp.where(tick >= sc.ai_intro_tick,
                    jnp.float32(sc.quality_depreciation), jnp.float32(0.0))
    new_quality = firms.product_quality * (jnp.float32(1.0) - dep) + quality_gain
    new_quality = jnp.clip(new_quality, jnp.float32(0.5), jnp.float32(10.0))

    # Maintenance cost for active adopters is settled in S5 (inside the profit
    # ledger) — audit 2026-10-07 removed the second cash deduction here.
    return firms._replace(
        ai_experience=jnp.where(firms.alive, new_exp, firms.ai_experience),
        product_quality=jnp.where(firms.alive, new_quality, firms.product_quality))


def _ai_tfp_multiplier(ai_experience, ai_investment, sc):
    """Compute AI-augmented TFP multiplier with J-curve.

    TFP_mult(x) = 1 + ai_efficiency_gain * ((1+j_curve_depth)*x - j_curve_depth)
    At x=0: TFP_mult = 1 - ai_efficiency_gain * j_curve_depth  (dip)
    At x=1: TFP_mult = 1 + ai_efficiency_gain              (full gain)

    Investment also contributes directly to cost reduction (handled in production).
    """
    x = ai_experience
    gain = jnp.float32(sc.ai_efficiency_gain)
    depth = jnp.float32(sc.j_curve_depth)

    multiplier = jnp.float32(1.0) + gain * ((jnp.float32(1.0) + depth) * x - depth)
    return jnp.where(ai_experience > 0, multiplier, jnp.float32(1.0))


def _ai_cost_multiplier(ai_experience, ai_investment, ai_strategy, sc):
    """Compute AI-augmented marginal cost reduction.

    Cost reduction depends on experience and cost-focused investment.
    Strategy near 0 -> more cost reduction. Strategy near 1 -> less.
    """
    x = ai_experience
    s = ai_strategy  # 0 = cost-focused, 1 = quality-focused
    cost_eff = jnp.float32(sc.ai_cost_elasticity)
    invest_contribution = ai_investment * (jnp.float32(1.0) - s) * cost_eff

    # Cost multiplier: starts at 1.0, decreases with AI experience and cost investment
    cost_mult = jnp.float32(1.0) - jnp.float32(sc.ai_cost_experience_coeff) * x - invest_contribution
    return jnp.where(ai_experience > 0, jnp.maximum(cost_mult, jnp.float32(0.5)), jnp.float32(1.0))


# ============================================================
# S3: Production (Cobb-Douglas + product quality + AI)
# ============================================================

def _produce_jit(firms, market, sc):
    """Cobb-Douglas production with AI augmentation and product quality.

    Q_i = A_base_i * quality_i^theta * TFP_mult(ai) * K_i^alpha * L_i^(1-alpha)
    Marginal cost is computed in _market_clearing as
      MC_i = (w L_i + delta K_i) / Q_i * cost_multiplier(ai),
    so quality enters unit cost implicitly through Q_i (i.e. MC is divided
    by quality_i^theta). Kept in sync with manuscript Eq. (4)-(5).
    """
    alpha = jnp.float32(sc.alpha_CD)
    theta = jnp.float32(sc.quality_weight)
    K_safe = jnp.maximum(firms.K, jnp.float32(0.01))
    L_safe = jnp.maximum(firms.L, jnp.float32(0.01))

    tfp_mult = _ai_tfp_multiplier(firms.ai_experience, firms.ai_investment, sc)
    A_eff = firms.A_base * tfp_mult

    # Quality enters production as productivity multiplier
    Q = A_eff * jnp.power(jnp.maximum(firms.product_quality, jnp.float32(0.1)), theta) * \
        jnp.power(K_safe, alpha) * jnp.power(L_safe, jnp.float32(1.0) - alpha)
    Q = jnp.where(firms.alive, Q, jnp.float32(0.0))

    return firms._replace(Q=Q)


# ============================================================
# S4: Market Clearing (CES consumer choice -> endogenous markup)
# ============================================================

def _market_clearing(firms, market, sc):
    """CES monopolistic competition with endogenous markups.

    Consumer choice (CES with quality differentiation):
      s_i = (quality_i^theta * (MC_i * mu_i)^(-(sigma-1))) / sum_j(...)

    Kimball-style demand elasticity:
      eps_i = sigma - (sigma-1) * s_i   [simplified Kimball approximation]

    Markup:
      mu_i = eps_i / (eps_i - 1)

    Key insight: LARGE firms (high s_i) face LOWER demand elasticity -> HIGHER markup.
    This creates a positive feedback loop -> endogenous market structure.

    Returns updated firms (with revenue, market_share, effective_markup) and market.
    """
    alpha = jnp.float32(sc.alpha_CD)
    theta = jnp.float32(sc.quality_weight)
    n_alive = jnp.maximum(jnp.sum(firms.alive.astype(jnp.float32)), 1.0)
    alive = firms.alive

    # Per-firm substitution elasticity
    sigma_i = firms.sigma_i

    # Compute marginal cost: MC = wage * (L/Q) + depreciation * (K/Q) + operating
    # Simplified: MC per unit of output
    wage = market.wage_rate
    total_Q_safe = jnp.maximum(jnp.sum(firms.Q * alive), jnp.float32(0.01))
    Q_safe = jnp.maximum(firms.Q, jnp.float32(0.01))

    # AI cost reduction
    cost_mult = _ai_cost_multiplier(
        firms.ai_experience, firms.ai_investment, firms.ai_strategy, sc)

    # Marginal cost: (wage * L + depreciation * K) / Q, adjusted by AI
    MC = (wage * firms.L + sc.depreciation_rate * firms.K) / Q_safe * cost_mult
    MC = jnp.maximum(MC, jnp.float32(0.001))

    # Effective markup from Kimball demand elasticity (simplified)
    # eps_i = sigma_i - (sigma_i-1) * s_i  [per-firm sigma]
    # Use previous period's market share as starting point
    prev_share = jnp.maximum(firms.market_share, jnp.float32(1e-6))
    eps_i = sigma_i - (sigma_i - jnp.float32(1.0)) * prev_share
    eps_i = jnp.maximum(eps_i, jnp.float32(1.01))

    # Markup
    mu_i = eps_i / (eps_i - jnp.float32(1.0))

    # Variant A: demand shock reduces perceived markup (firms cut prices when demand falls)
    demand_shock = market.demand_externalit_shock
    demand_markup_adj = jnp.float32(1.0) - demand_shock * jnp.float32(sc.demand_markup_sensitivity)
    demand_markup_adj = jnp.maximum(demand_markup_adj, jnp.float32(0.5))  # floor
    mu_i = mu_i * demand_markup_adj

    # Price
    P_i = MC * mu_i
    P_i = jnp.where(alive, P_i, jnp.float32(0.0))

    # Consumer demand: CES share (per-firm sigma in exponent)
    # share_i = (quality_i^theta * P_i^(-(sigma_i-1))) / sum_j (...)
    quality_term = jnp.power(jnp.maximum(firms.product_quality, jnp.float32(0.1)), theta)
    price_term = jnp.power(P_i, -(sigma_i - jnp.float32(1.0)))
    # Dead firms get zero demand
    demand_weight = jnp.where(alive, quality_term * price_term, jnp.float32(0.0))
    total_weight = jnp.maximum(jnp.sum(demand_weight), jnp.float32(1e-10))

    new_share = demand_weight / total_weight
    new_share = jnp.where(alive, new_share, jnp.float32(0.0))

    # Revenue: share of total spending (market_size_effective)
    market_eff = jnp.maximum(market.market_size_effective, jnp.float32(1.0))
    revenue = new_share * market_eff

    # Update market state
    total_Q = jnp.sum(firms.Q * alive)
    P_agg = market_eff / jnp.maximum(total_Q, jnp.float32(0.01))

    # Wage: simplified (layer 1). Replaced by matching function in layer 3.
    total_L = jnp.sum(firms.L * alive)
    total_L_safe = jnp.maximum(total_L, jnp.float32(1.0))
    wage = (jnp.float32(1.0) - alpha) * P_agg * jnp.power(
        total_Q / total_L_safe, alpha)

    # HHI
    hhi = jnp.sum(new_share ** 2)

    # AI adoption rate: fraction of firms with significant AI investment
    n_ai = jnp.sum((firms.ai_experience > jnp.float32(0.1)) & alive).astype(jnp.float32)
    ai_adopt = n_ai / n_alive

    # Labor share = wage * L / revenue
    total_labor_cost = wage * jnp.sum(firms.L * alive)
    total_revenue = jnp.sum(revenue)
    labor_share = total_labor_cost / jnp.maximum(total_revenue, jnp.float32(0.01))

    new_market = MarketState(
        total_Q=total_Q,
        agg_price=P_agg,
        wage_rate=wage,
        ai_adoption_rate=ai_adopt,
        hhi=hhi,
        labor_share=labor_share,
        demand_externalit_shock=market.demand_externalit_shock,
        market_size_effective=market.market_size_effective,
        unemployment_rate=market.unemployment_rate,
        avg_wage=market.avg_wage,
    )

    return firms._replace(
        revenue=revenue,
        market_share=new_share,
        effective_markup=mu_i,
    ), new_market


# ============================================================
# S5: Profit calculation
# ============================================================

def _calc_profit_jit(firms, market, sc):
    """Calculate costs and profit.

    Cost = wage * L + depreciation * K + operating_cost * revenue + ai_investment + ai_maintenance
    Profit = revenue - cost

    Revenue is already adjusted by demand externality from market clearing.
    """
    wage = market.wage_rate

    labor_cost = wage * firms.L
    capital_cost = sc.depreciation_rate * firms.K
    operating_cost = sc.operating_cost_rate * firms.revenue
    ai_cost = firms.ai_investment + jnp.where(
        firms.ai_experience > 0, sc.ai_maintenance_cost, jnp.float32(0.0))

    total_cost = labor_cost + capital_cost + operating_cost + ai_cost
    total_cost = jnp.where(firms.alive, total_cost, jnp.float32(0.0))

    profit = firms.revenue - total_cost
    profit = jnp.where(firms.alive, profit, jnp.float32(0.0))

    # Update cash
    new_cash = firms.cash + profit

    return firms._replace(
        cost=total_cost, profit=profit, cash=jnp.maximum(new_cash, jnp.float32(0.0)))


# ============================================================
# S6: Investment (capital adjustment)
# ============================================================

def _invest_jit(firms, sc):
    """Firms invest profits into capital, depreciate existing capital.

    Audit 2026-10-07 (external review B4): the reinvested fraction used to
    stay in cash as well (cash += full profit in S5, K += iota*profit here) —
    money created from nothing. Cash ballooned until the cash constraints
    (cash > F_AI, iota-cash caps) never bound. Physical reinvestment is now
    deducted from cash in the same step.
    """
    depreciation = jnp.float32(sc.depreciation_rate)
    inv_rate = jnp.float32(sc.investment_rate)

    reinvest = inv_rate * jnp.maximum(firms.profit, jnp.float32(0.0))
    new_K = firms.K * (jnp.float32(1.0) - depreciation) + reinvest
    new_K = jnp.where(firms.alive, new_K, firms.K)
    new_cash = jnp.maximum(firms.cash - reinvest, jnp.float32(0.0))
    new_cash = jnp.where(firms.alive, new_cash, firms.cash)

    return firms._replace(K=new_K, cash=new_cash)


# ============================================================
# S7: Entry / Exit
# ============================================================

def _entry_exit_jit(firms, market, sc, key, tick, F):
    """Firm entry and exit based on financial health.

    Exit: firms with cash below threshold die.
    Entry: new firms born when profit opportunities exist (driven by avg profitability).
    New entrants have random technology level (sometimes better than incumbents).
    """
    k_entry, k_birth = jax.random.split(key)
    # Audit 2026-10-07 (external review B1): all four trait draws used to
    # share one key -> perfectly correlated entrant K/L/A/cash. Now split.
    kb_K, kb_L, kb_A, kb_cash, kb_sig = jax.random.split(k_birth, 5)

    # Exit: firms with cash below threshold die
    alive = firms.alive & (firms.cash > sc.firm_exit_cash_threshold)

    # Clear dead firms
    firms = firms._replace(
        alive=alive,
        K=jnp.where(alive, firms.K, jnp.float32(0.0)),
        L=jnp.where(alive, firms.L, jnp.float32(0.0)),
        cash=jnp.where(alive, firms.cash, jnp.float32(0.0)),
        age=jnp.where(alive, firms.age, jnp.int32(0)),
        product_quality=jnp.where(alive, firms.product_quality, jnp.float32(0.0)),
        ai_investment=jnp.where(alive, firms.ai_investment, jnp.float32(0.0)),
        ai_strategy=jnp.where(alive, firms.ai_strategy, jnp.float32(0.5)),
        ai_experience=jnp.where(alive, firms.ai_experience, jnp.float32(0.0)),
        Q=jnp.where(alive, firms.Q, jnp.float32(0.0)),
        revenue=jnp.where(alive, firms.revenue, jnp.float32(0.0)),
        cost=jnp.where(alive, firms.cost, jnp.float32(0.0)),
        profit=jnp.where(alive, firms.profit, jnp.float32(0.0)),
        market_share=jnp.where(alive, firms.market_share, jnp.float32(0.0)),
        effective_markup=jnp.where(alive, firms.effective_markup, jnp.float32(1.5)),
    )

    # Entry: new firms born with probability firm_entry_rate
    n_alive = jnp.sum(alive)
    n_dead = F - n_alive
    u = jax.random.uniform(k_entry, (F,), dtype=jnp.float32)
    born = ~alive & (u < sc.firm_entry_rate)

    # Initialize new firms. A_base lognormal (matches init_firms); K/L/cash
    # normal (matches init_firms post-audit); sigma_i redrawn from the fresh
    # A_base (audit 2026-10-07).
    n_born = jnp.sum(born)
    new_K = jax.random.normal(kb_K, (F,), dtype=jnp.float32) * jnp.float32(sc.initial_K_std) + jnp.float32(sc.initial_K_mean)
    new_K = jnp.clip(new_K, jnp.float32(0.5), jnp.float32(100.0))

    new_L = jax.random.normal(kb_L, (F,), dtype=jnp.float32) * jnp.float32(sc.initial_L_std) + jnp.float32(sc.initial_L_mean)
    new_L = jnp.clip(new_L, jnp.float32(5.0), jnp.float32(200.0))

    new_A = jnp.exp(jax.random.normal(kb_A, (F,)) * sc.A_base_std) * sc.A_base_mean
    new_A = jnp.clip(new_A, jnp.float32(0.3), jnp.float32(3.0))

    new_cash = jax.random.normal(kb_cash, (F,), dtype=jnp.float32) * jnp.float32(sc.initial_cash_std) + jnp.float32(sc.initial_cash_mean)
    new_cash = jnp.clip(new_cash, jnp.float32(1.0), jnp.float32(100.0))

    # sigma_i for newborns: same structural rule as init_firms
    new_sigma = (jnp.float32(sc.sigma_base)
                 - jnp.float32(sc.sigma_firm_A_sensitivity) * (new_A - jnp.float32(1.0))
                 + jax.random.normal(kb_sig, (F,), dtype=jnp.float32) * jnp.float32(sc.sigma_firm_spread))
    new_sigma = jnp.clip(new_sigma, jnp.float32(sc.sigma_min), jnp.float32(sc.sigma_max))

    firms = firms._replace(
        alive=alive | born,
        K=jnp.where(born, new_K, firms.K),
        L=jnp.where(born, new_L, firms.L),
        A_base=jnp.where(born, new_A, firms.A_base),
        sigma_i=jnp.where(born, new_sigma, firms.sigma_i),
        adoption_readiness=jnp.where(born, jnp.float32(0.0), firms.adoption_readiness),
        product_quality=jnp.where(born, jnp.float32(1.0), firms.product_quality),
        ai_investment=jnp.where(born, jnp.float32(0.0), firms.ai_investment),
        ai_strategy=jnp.where(born, jnp.float32(0.5), firms.ai_strategy),
        ai_experience=jnp.where(born, jnp.float32(0.0), firms.ai_experience),
        cash=jnp.where(born, new_cash, firms.cash),
        Q=jnp.where(born, jnp.float32(0.0), firms.Q),
        revenue=jnp.where(born, jnp.float32(0.0), firms.revenue),
        cost=jnp.where(born, jnp.float32(0.0), firms.cost),
        profit=jnp.where(born, jnp.float32(0.0), firms.profit),
        market_share=jnp.where(born, jnp.float32(1.0 / F), firms.market_share),
        effective_markup=jnp.where(born, jnp.float32(1.5), firms.effective_markup),
        age=jnp.where(born, jnp.int32(0), firms.age),
    )

    return firms


# ============================================================
# S8: Demand Feedback (Keynesian externality)
# ============================================================

def _demand_feedback_jit(firms, market, sc):
    """Compute demand externality shock for next period.

    Core mechanism (Falk & Tsoukalas 2026):
    When firms automate (reduce L), displaced workers lose income.
    Since worker_mpc > owner_mpc, aggregate demand falls.
    Each firm bears only 1/N of the demand loss -> over-investment in automation.

    We track L change as a proxy for automation/displacement:
    If AI-investing firms reduce L relative to their revenue, that's displacement.
    """
    alive = firms.alive
    n_alive_f = jnp.maximum(jnp.sum(alive.astype(jnp.float32)), jnp.float32(1.0))

    # Track AI-driven labor reduction
    # Firms with high AI experience that have lower L relative to revenue are "automating"
    ai_firms = alive & (firms.ai_experience > jnp.float32(0.1))
    # Labor intensity: L / revenue ratio for AI firms
    avg_labor_intensity_all = jnp.sum(firms.L * alive) / jnp.maximum(jnp.sum(firms.revenue * alive), jnp.float32(1.0))

    # The demand shock comes from the gap: worker income lost vs owner income gained
    # worker_consumption = wage * total_L * worker_mpc
    # owner_consumption = total_profit * owner_mpc
    # If labor_share drops, total consumption drops (because worker_mpc > owner_mpc)
    worker_mpc = jnp.float32(sc.worker_mpc)
    owner_mpc = jnp.float32(sc.owner_mpc)
    wage = market.wage_rate

    total_labor_income = wage * jnp.sum(firms.L * alive)
    total_profit_income = jnp.sum(firms.profit * alive)
    total_consumption = total_labor_income * worker_mpc + total_profit_income * owner_mpc

    # "Ideal" consumption if all income went to workers
    total_income = jnp.maximum(total_labor_income + total_profit_income, jnp.float32(1.0))
    ideal_consumption = total_income * worker_mpc  # if all income were wages

    # Shock = how much consumption falls short of ideal
    consumption_gap = (ideal_consumption - total_consumption) / ideal_consumption
    consumption_gap = jnp.maximum(consumption_gap, jnp.float32(0.0))

    # Accumulate shock with recovery
    prev_shock = market.demand_externalit_shock
    new_shock = consumption_gap * sc.demand_shock_coefficient \
                 + prev_shock * (jnp.float32(1.0) - jnp.float32(sc.demand_recovery_rate))
    new_shock = jnp.clip(new_shock, jnp.float32(0.0), jnp.float32(sc.demand_shock_max))

    # Adjust effective market size
    market_size = jnp.float32(sc.market_size)
    new_market_size = market_size * (jnp.float32(1.0) - new_shock)

    new_market = MarketState(
        total_Q=market.total_Q,
        agg_price=market.agg_price,
        wage_rate=market.wage_rate,
        ai_adoption_rate=market.ai_adoption_rate,
        hhi=market.hhi,
        labor_share=market.labor_share,
        demand_externalit_shock=new_shock,
        market_size_effective=new_market_size,
        unemployment_rate=market.unemployment_rate,
        avg_wage=market.avg_wage,
    )

    return new_market


# ============================================================
# S9: Labor Market Clearing (matching function)
# ============================================================

def _labor_market_clearing_jit(firms, workers, market, sc, key, F, W):
    """Labor market: endogenous L from profit maximization + worker flows.

    Firms choose optimal L from production function (profit max condition).
    AI reduces labor marginal product → firms naturally reduce employment.
    Workers find/lose jobs through probabilistic transitions.
    No preset age premium — age-wage profile emerges from skill accumulation.
    """
    alive = firms.alive
    n_alive_f = jnp.maximum(jnp.sum(alive.astype(jnp.float32)), jnp.float32(1.0))
    n_workers_alive = jnp.maximum(jnp.sum(workers.alive.astype(jnp.float32)), jnp.float32(1.0))

    alpha = jnp.float32(sc.alpha_CD)
    theta = jnp.float32(sc.quality_weight)
    wage = market.wage_rate
    agg_price = market.agg_price

    # --- Endogenous optimal L (profit maximization) ---
    # Q = A_eff * quality^theta * K^alpha * L^(1-alpha)
    # MP_L = (1-alpha) * Q / L
    # Optimal condition: wage = P * MP_L
    # => L* = ( (1-alpha) * A_eff * quality^theta * P / wage )^(1/alpha) * K
    K_safe = jnp.maximum(firms.K, jnp.float32(0.01))
    quality_safe = jnp.maximum(firms.product_quality, jnp.float32(0.1))

    tfp_mult = _ai_tfp_multiplier(firms.ai_experience, firms.ai_investment, sc)
    A_eff = firms.A_base * tfp_mult

    # AI reduces labor marginal product (substitution effect)
    ai_mp_l_factor = jnp.where(
        firms.ai_experience > jnp.float32(0.0),
        jnp.float32(1.0) - jnp.float32(sc.ai_labor_displacement) * firms.ai_experience,
        jnp.float32(1.0))

    # Optimal L from FOC
    productivity = A_eff * jnp.power(quality_safe, theta) * ai_mp_l_factor
    wage_safe = jnp.maximum(wage, jnp.float32(0.01))
    price_safe = jnp.maximum(agg_price, jnp.float32(0.01))
    L_optimal = jnp.power(
        jnp.maximum(
            (jnp.float32(1.0) - alpha) * productivity * price_safe / wage_safe,
            jnp.float32(0.001)),
        jnp.float32(1.0) / alpha) * K_safe
    # Upper bound: can't hire more than available labor
    L_max = jnp.float32(sc.total_labor_supply) / n_alive_f
    L_optimal = jnp.clip(L_optimal, jnp.float32(1.0), L_max)

    # Smooth adjustment toward optimal
    adj_speed = jnp.float32(sc.labor_adjustment_speed)
    new_L = firms.L * (jnp.float32(1.0) - adj_speed) + L_optimal * adj_speed
    new_L = jnp.where(alive, new_L, firms.L)

    # --- Employment rate from firm L ---
    total_employment = jnp.sum(new_L * alive)
    employment_rate = jnp.clip(total_employment / n_workers_alive, jnp.float32(0.0), jnp.float32(1.0))

    # --- Worker employment status ---
    avg_skill = jnp.maximum(jnp.sum(workers.skill * workers.alive) / n_workers_alive, jnp.float32(1.0))
    skill_w = jnp.float32(sc.skill_employment_weight)
    compat_w = jnp.float32(sc.compat_employment_weight)
    base_w = jnp.float32(1.0) - skill_w - compat_w

    skill_weight = workers.skill / jnp.maximum(avg_skill, jnp.float32(0.01))
    ai_adopt_rate = market.ai_adoption_rate
    compat_scaled = workers.ai_compatibility * ai_adopt_rate

    emp_prob = employment_rate * (base_w + skill_w * skill_weight + compat_w * compat_scaled)
    emp_prob = jnp.clip(emp_prob, jnp.float32(0.0), jnp.float32(0.95))

    # Separation: endogenous — higher when demand is contracting
    was_employed = workers.employer >= 0
    demand_stress = market.demand_externalit_shock
    sep_prob = jnp.float32(sc.base_separation_prob) + jnp.float32(sc.demand_separation_sensitivity) * demand_stress
    separation_prob = jnp.where(was_employed, sep_prob, jnp.float32(0.0))

    is_unemployed = workers.alive & ~was_employed

    k_emp = jax.random.split(key, 1)[0]
    rand = jax.random.uniform(k_emp, (W,), dtype=jnp.float32)

    gets_hired = is_unemployed & (rand < emp_prob)
    gets_fired = was_employed & (rand < separation_prob)

    hired_firm = jax.random.categorical(k_emp, jnp.log(
        jnp.where(alive, jnp.ones(F, dtype=jnp.float32), jnp.float32(1e-6))), shape=(W,))

    new_employer = workers.employer
    new_employer = jnp.where(gets_hired, hired_firm, new_employer)
    new_employer = jnp.where(gets_fired, jnp.int32(-1), new_employer)

    # --- Wage determination (no age premium — emerges from skill) ---
    min_wage_val = jnp.float32(sc.min_wage)

    actual_unemployed = workers.alive & (new_employer < 0)
    n_unemployed = jnp.sum(actual_unemployed.astype(jnp.float32))
    unemp_rate = n_unemployed / n_workers_alive
    bargaining_power = jnp.maximum(jnp.float32(1.0) - jnp.float32(0.8) * unemp_rate, jnp.float32(0.1))

    worker_wage = jnp.maximum(
        workers.skill * wage * bargaining_power,
        min_wage_val)
    worker_wage = jnp.where(workers.alive & (new_employer >= 0), worker_wage, jnp.float32(0.0))

    # Unemployment duration
    new_unemp_dur = jnp.where(
        new_employer < 0,
        workers.unemployment_duration + jnp.float32(1.0),
        jnp.float32(0.0))

    new_workers = workers._replace(
        employer=new_employer,
        wage=worker_wage,
        unemployment_duration=new_unemp_dur,
    )

    new_firms = firms._replace(L=new_L)

    # Update market
    avg_wage = jnp.sum(worker_wage * workers.alive) / n_workers_alive
    new_market = MarketState(
        total_Q=market.total_Q,
        agg_price=market.agg_price,
        wage_rate=wage,
        ai_adoption_rate=market.ai_adoption_rate,
        hhi=market.hhi,
        labor_share=market.labor_share,
        demand_externalit_shock=market.demand_externalit_shock,
        market_size_effective=market.market_size_effective,
        unemployment_rate=unemp_rate,
        avg_wage=avg_wage,
    )

    return new_firms, new_workers, new_market


# ============================================================
# S10: Worker Life Cycle
# ============================================================

def _worker_lifecycle_jit(workers, sc, key, W):
    """Worker aging, skill evolution, retirement, and new entry.

    - Employed workers: skill grows slowly
    - Unemployed workers: skill depreciates (faster with duration)
    - Age > retirement_age: retire (exit labor force)
    - New young workers enter to replace retirees
    """
    k_retire, k_enter, k_adapt, k_compat = jax.random.split(key, 4)
    # Audit 2026-10-07 (external review B5): entering / adaptability /
    # ai_compatibility used to share one key -> comonotone draws. Split.

    alive = workers.alive
    employed = alive & (workers.employer >= 0)
    unemployed = alive & (workers.employer < 0)

    # Aging
    new_age = workers.age + 1

    # Skill evolution
    # Employed: skill grows with learning_rate
    skill_growth = jnp.where(
        employed,
        jnp.float32(sc.skill_learning_rate) * workers.skill * workers.adaptability,
        jnp.float32(0.0))

    # Unemployed: skill depreciates
    skill_loss = jnp.where(
        unemployed,
        jnp.float32(sc.skill_depreciation_rate) * workers.skill * workers.unemployment_duration / jnp.float32(12.0),
        jnp.float32(0.0))

    new_skill = workers.skill * (jnp.float32(1.0) + skill_growth - skill_loss)
    new_skill = jnp.clip(new_skill, jnp.float32(0.1), jnp.float32(10.0))

    # Retirement (ages are in months, 1 tick = 1 month — see state.py)
    retired = alive & (new_age >= sc.retirement_age * 12)
    new_alive = alive & ~retired

    # New workers enter to replace retirees (and maintain workforce size)
    n_retired = jnp.sum(retired.astype(jnp.float32))
    # Also replace some workers who dropped out
    n_enter = jnp.minimum(jnp.floor(n_retired * jnp.float32(1.2)), jnp.float32(W * jnp.float32(0.02)))

    entering = jax.random.uniform(k_enter, (W,), dtype=jnp.float32) < (n_enter / jnp.maximum(jnp.sum((~new_alive).astype(jnp.float32)), jnp.float32(1.0)))
    entering = entering & ~new_alive  # only enter vacant slots

    # Initialize new workers (age in months); independent keys per trait (B5)
    age_min = sc.age_new_worker
    new_worker_age = jnp.full((W,), age_min * 12, dtype=jnp.int32)
    new_worker_skill = jnp.exp(jax.random.normal(k_adapt, (W,), dtype=jnp.float32) * jnp.float32(0.3))
    new_worker_skill = jnp.clip(new_worker_skill, jnp.float32(0.2), jnp.float32(3.0))
    new_worker_adapt = jax.random.uniform(k_adapt, (W,), dtype=jnp.float32) * jnp.float32(0.8) + jnp.float32(0.2)
    new_worker_adapt = jnp.clip(new_worker_adapt, jnp.float32(0.1), jnp.float32(1.0))
    new_worker_ai_compat = new_worker_adapt * jnp.float32(0.5) + jax.random.uniform(k_compat, (W,), dtype=jnp.float32) * jnp.float32(0.5)
    new_worker_ai_compat = jnp.clip(new_worker_ai_compat, jnp.float32(0.0), jnp.float32(1.0))

    # Apply entering workers
    final_alive = new_alive | entering
    final_age = jnp.where(entering, new_worker_age, new_age)
    final_skill = jnp.where(entering, new_worker_skill, new_skill)
    final_adapt = jnp.where(entering, new_worker_adapt, workers.adaptability)
    final_ai_compat = jnp.where(entering, new_worker_ai_compat, workers.ai_compatibility)
    final_employer = jnp.where(~final_alive, jnp.full((W,), -1, dtype=jnp.int32), workers.employer)
    final_wage = jnp.where(~final_alive, jnp.float32(0.0), workers.wage)
    final_unemp_dur = jnp.where(~final_alive, jnp.float32(0.0), workers.unemployment_duration)

    return workers._replace(
        alive=final_alive,
        age=final_age,
        skill=final_skill.astype(jnp.float32),
        adaptability=final_adapt.astype(jnp.float32),
        employer=final_employer,
        wage=final_wage,
        ai_compatibility=final_ai_compat.astype(jnp.float32),
        unemployment_duration=final_unemp_dur,
    )


# ============================================================
# Core JIT step
# ============================================================

@jax.jit(static_argnums=1)
def step_core(state: SimState, sc: _FrozenConfig) -> SimState:
    firms = state.firms
    workers = state.workers
    market = state.market
    tick = state.tick
    key = state.key
    F = firms.K.shape[0]
    W = workers.age.shape[0]

    keys = jax.random.split(key, 5)
    k_ai, k_entry, _, k_out, k_labor = keys

    # Timing note: after run_n_ticks(state, 250) (ticks 0..249), the first
    # step of the nominal 120-tick AI window executes at tick 250 with AI
    # still inactive; AI is active from tick 251, i.e. for 119 of the 120
    # window ticks. Uniform across every experiment and every result in the
    # paper; the manuscript's "AI introduced at tick t_AI = 251" matches
    # this behavior exactly.
    ai_available = tick >= sc.ai_intro_tick

    # S1: AI investment decision (strategic)
    firms = _ai_investment_decision_jit(firms, market, sc, k_ai, tick, F)

    # S2: AI learning + quality accumulation
    firms = _ai_learning_jit(firms, sc, tick)

    # S3: Production
    firms = _produce_jit(firms, market, sc)

    # S4: Market clearing (CES consumer choice -> endogenous markup)
    firms, market = _market_clearing(firms, market, sc)

    # S5: Profit
    firms = _calc_profit_jit(firms, market, sc)

    # S6: Investment
    firms = _invest_jit(firms, sc)

    # S7: Entry/exit
    firms = _entry_exit_jit(firms, market, sc, k_entry, tick, F)

    # S8: Demand feedback
    market = _demand_feedback_jit(firms, market, sc)

    # S9: Labor market clearing (matching)
    firms, workers, market = _labor_market_clearing_jit(firms, workers, market, sc, k_labor, F, W)

    # S10: Worker life cycle
    workers = _worker_lifecycle_jit(workers, sc, _, W)

    # S11: Aging (firms)
    firms = firms._replace(age=firms.age + 1)

    return SimState(firms=firms, workers=workers, market=market, tick=tick + 1, key=k_out)


# ============================================================
# Public API
# ============================================================

def sim_step(state: SimState, config: dict):
    sc = _make_static_config(config)
    new_state = step_core(state, sc)
    return new_state


@jax.jit(static_argnums=(1, 2))
def run_n_ticks(state: SimState, sc: _FrozenConfig, n_ticks: int) -> SimState:
    def body(carry, _):
        return step_core(carry, sc), None
    final_state, _ = jax.lax.scan(body, state, None, length=n_ticks)
    return final_state


def run_n_ticks_collect(state: SimState, sc: _FrozenConfig, n_ticks: int,
                        sample_every: int = 1):
    """Run n_ticks via scan, returning (final_state, sampled_states_pytree).

    Collects only every sample_every-th state to limit memory.
    sampled_states_pytree has leaves of shape (n_samples, *leaf_shape).
    """
    def body(carry, x):
        new_state = step_core(carry, sc)
        return new_state, new_state

    final_state, all_states = jax.lax.scan(body, state, None, length=n_ticks)

    # all_states has shape (n_ticks, ...). Subsample at every sample_every-TH
    # step (states after ticks sample_every, 2*sample_every, ...). Audit
    # 2026-10-07: indices used to start at 0, i.e. the state after tick 1 was
    # labeled as tick `sample_every`, shifting every collected time series
    # (rho(t) curves) by ~1 period.
    indices = jnp.arange(sample_every - 1, n_ticks, sample_every)
    sampled = jax.tree_util.tree_map(lambda a: a[indices], all_states)
    return final_state, sampled
