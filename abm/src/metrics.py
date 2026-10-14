"""Metrics for AI Productivity Paradox ABM v2.0.

Key outputs:
  - Micro TFP: firm-level productivity of AI-investing vs non-investing firms
  - Macro TFP: aggregate productivity (total output / effective inputs)
  - Labor share: wages / total revenue
  - HHI: Herfindahl concentration index (endogenous market structure)
  - Markup distribution: effective markup per firm (emergent from market position)
  - Paradox detection: micro TFP↑ but macro TFP unchanged
  - Demand externality shock magnitude
"""

import jax.numpy as jnp
from typing import Dict, Any

from .state import FirmState, MarketState


def gini(values) -> float:
    """Gini coefficient from sorted values."""
    n = values.shape[0]
    if n < 2:
        return 0.0
    cum = jnp.cumsum(values)
    total = cum[-1]
    if total < 1e-10:
        return 0.0
    g = (n + 1 - 2 * jnp.sum(cum) / total) / n
    return float(jnp.clip(g, 0.0, 1.0))


def compute_metrics(
    firms: FirmState, market: MarketState,
    config: dict, tick: int,
    pre_ai_snapshot: Dict = None,
) -> Dict[str, Any]:
    alive = firms.alive
    n_alive = int(jnp.sum(alive))
    if n_alive < 1:
        return {'tick': tick, 'n_firms': 0}

    result = {'tick': tick, 'n_firms': n_alive}

    # AI investment / adoption
    ai_mask = alive & (firms.ai_experience > 0.1)
    no_ai_mask = alive & (firms.ai_experience <= 0.1)
    n_ai = int(jnp.sum(ai_mask))
    n_no_ai = int(jnp.sum(no_ai_mask))
    result['n_ai_firms'] = n_ai
    result['n_no_ai_firms'] = n_no_ai
    result['ai_adoption_rate'] = float(market.ai_adoption_rate)

    # AI investment statistics
    result['avg_ai_investment'] = float(jnp.mean(jnp.where(alive, firms.ai_investment, 0.0)))
    result['total_ai_investment'] = float(jnp.sum(firms.ai_investment[alive]))
    result['avg_ai_experience'] = float(jnp.mean(jnp.where(alive, firms.ai_experience, 0.0)))

    # AI strategy distribution
    result['avg_ai_strategy'] = float(jnp.mean(jnp.where(
        alive & (firms.ai_experience > 0), firms.ai_strategy, 0.0)))
    result['std_ai_strategy'] = float(jnp.std(jnp.where(
        alive & (firms.ai_experience > 0), firms.ai_strategy, 0.0)))

    # Strategy by firm size tercile (monitor imitation learning divergence)
    if n_alive > 2:
        shares = firms.market_share[alive]
        sorted_indices = jnp.argsort(shares)
        t1 = n_alive // 3
        t2 = 2 * n_alive // 3
        strats = firms.ai_strategy
        result['strategy_small_firms'] = float(jnp.mean(strats[sorted_indices[:t1]])) if t1 > 0 else 0.0
        result['strategy_mid_firms'] = float(jnp.mean(strats[sorted_indices[t1:t2]])) if t2 > t1 else 0.0
        result['strategy_large_firms'] = float(jnp.mean(strats[sorted_indices[t2:]])) if n_alive > t2 else 0.0

    # Output
    total_Q = float(jnp.sum(firms.Q[alive]))
    result['total_Q'] = total_Q
    result['avg_Q'] = total_Q / n_alive

    # Micro TFP: Q / (K^alpha * L^(1-alpha)).
    # Audit 2026-10-07: Q is recomputed from the state (A_eff * q^theta *
    # K^alpha L^(1-alpha)) — the recorded firms.Q uses within-tick K/L while
    # the reported state's K/L are end-of-tick, and newborn entrants carry
    # Q=0, mixing vintages (3.5-6% level bias).
    from .step import _make_static_config
    sc = _make_static_config(config)
    alpha = float(sc.alpha_CD)
    theta = float(sc.quality_weight)
    K_safe = jnp.maximum(firms.K, 0.01)
    L_safe = jnp.maximum(firms.L, 0.01)
    effective_inputs = jnp.power(K_safe, alpha) * jnp.power(L_safe, 1.0 - alpha)
    gain = float(sc.ai_efficiency_gain)
    depth = float(sc.j_curve_depth)
    mult = jnp.where(
        firms.ai_experience > 0,
        jnp.float32(1.0) + gain * ((jnp.float32(1.0) + depth)
                                   * firms.ai_experience - depth),
        jnp.float32(1.0))
    a_eff = firms.A_base * mult
    q_safe = jnp.maximum(firms.product_quality, jnp.float32(0.1))
    Q_hat = a_eff * jnp.power(q_safe, theta) * effective_inputs
    tfp = jnp.where(alive & (effective_inputs > 0.01),
                    Q_hat / effective_inputs, jnp.float32(0.0))

    if n_ai > 0:
        result['micro_tfp_ai'] = float(jnp.mean(tfp[ai_mask]))
    else:
        result['micro_tfp_ai'] = 0.0
    if n_no_ai > 0:
        result['micro_tfp_no_ai'] = float(jnp.mean(tfp[no_ai_mask]))
    else:
        result['micro_tfp_no_ai'] = 0.0

    # Macro TFP: aggregate Q / aggregate effective inputs (Q recomputed, see above)
    total_eff_inputs = float(jnp.sum(effective_inputs[alive]))
    result['macro_tfp'] = float(jnp.sum(Q_hat[alive])) / max(total_eff_inputs, 0.01)

    # Prices
    result['agg_price'] = float(market.agg_price)
    result['wage_rate'] = float(market.wage_rate)

    # Revenue and costs
    total_revenue = float(jnp.sum(firms.revenue[alive]))
    total_cost = float(jnp.sum(firms.cost[alive]))
    total_profit = float(jnp.sum(firms.profit[alive]))

    result['total_revenue'] = total_revenue
    result['total_cost'] = total_cost
    result['total_profit'] = total_profit

    # Labor share (guard: at t=0 total revenue is 0 and the old
    # max(..., 0.01) floor produced garbage values ~2e6 — audit B8)
    total_labor_cost = float(market.wage_rate * jnp.sum(firms.L[alive]))
    result['labor_share'] = (total_labor_cost / total_revenue
                             if total_revenue > 1e-6 else None)
    result['profit_share'] = total_profit / max(total_revenue, 0.01)

    # Profit rates by AI status
    if n_ai > 0:
        ai_revenue = float(jnp.sum(firms.revenue[ai_mask]))
        ai_profit = float(jnp.sum(firms.profit[ai_mask]))
        result['profit_rate_ai'] = ai_profit / max(ai_revenue, 0.01)
    else:
        result['profit_rate_ai'] = 0.0

    if n_no_ai > 0:
        no_ai_revenue = float(jnp.sum(firms.revenue[no_ai_mask]))
        no_ai_profit = float(jnp.sum(firms.profit[no_ai_mask]))
        result['profit_rate_no_ai'] = no_ai_profit / max(no_ai_revenue, 0.01)
    else:
        result['profit_rate_no_ai'] = 0.0

    # === v2.0 NEW: Market structure metrics ===

    # HHI (endogenous concentration)
    result['hhi'] = float(market.hhi)

    # Effective markup distribution
    markups = firms.effective_markup[alive]
    if n_alive > 0:
        result['avg_markup'] = float(jnp.mean(markups))
        result['median_markup'] = float(jnp.median(markups))
        result['max_markup'] = float(jnp.max(markups))
        result['min_markup'] = float(jnp.min(markups))
        result['std_markup'] = float(jnp.std(markups))
        result['gini_markup'] = gini(jnp.sort(markups))
        # Markup by firm size tercile
        shares = firms.market_share[alive]
        sorted_indices = jnp.argsort(shares)
        t1 = n_alive // 3
        t2 = 2 * n_alive // 3
        result['markup_small_firms'] = float(jnp.mean(markups[sorted_indices[:t1]])) if t1 > 0 else 0.0
        result['markup_mid_firms'] = float(jnp.mean(markups[sorted_indices[t1:t2]])) if t2 > t1 else 0.0
        result['markup_large_firms'] = float(jnp.mean(markups[sorted_indices[t2:]])) if n_alive > t2 else 0.0

    # Market share distribution
    shares = firms.market_share[alive]
    result['max_market_share'] = float(jnp.max(shares))
    result['top1_share'] = float(jnp.max(shares))
    # Top-k share using partition instead of topk
    sorted_shares = jnp.sort(shares)[::-1]  # descending
    result['top5_share'] = float(jnp.sum(sorted_shares[:min(5, n_alive)])) if n_alive >= 5 else float(jnp.sum(shares))
    result['top10_share'] = float(jnp.sum(sorted_shares[:min(10, n_alive)])) if n_alive >= 10 else float(jnp.sum(shares))

    # Product quality distribution
    quality = firms.product_quality[alive]
    if n_alive > 0:
        result['avg_quality'] = float(jnp.mean(quality))
        result['std_quality'] = float(jnp.std(quality))
        result['max_quality'] = float(jnp.max(quality))
        result['min_quality'] = float(jnp.min(quality))

    # Inequality
    if n_alive > 1:
        result['gini_profit'] = gini(jnp.sort(firms.profit[alive]))
        result['gini_revenue'] = gini(jnp.sort(firms.revenue[alive]))
        result['gini_tfp'] = gini(jnp.sort(tfp[alive]))

    # Capital and employment
    result['total_K'] = float(jnp.sum(firms.K[alive]))
    result['total_L'] = float(jnp.sum(firms.L[alive]))
    result['avg_cash'] = float(jnp.mean(firms.cash[alive]))
    result['avg_age'] = float(jnp.mean(firms.age[alive].astype(jnp.float32)))

    # Demand externality
    result['demand_shock'] = float(market.demand_externalit_shock)
    result['market_size_effective'] = float(market.market_size_effective)

    # Pre-AI comparison
    if pre_ai_snapshot is not None:
        pre_macro_tfp = pre_ai_snapshot.get('macro_tfp', result['macro_tfp'])
        pre_labor_share = pre_ai_snapshot.get('labor_share', result['labor_share'])
        pre_total_Q = pre_ai_snapshot.get('total_Q', result['total_Q'])
        pre_hhi = pre_ai_snapshot.get('hhi', result['hhi'])

        result['delta_macro_tfp'] = (result['macro_tfp'] - pre_macro_tfp) / max(pre_macro_tfp, 0.01)
        result['delta_labor_share'] = result['labor_share'] - pre_labor_share
        result['delta_total_Q'] = (result['total_Q'] - pre_total_Q) / max(pre_total_Q, 0.01)
        result['delta_hhi'] = result['hhi'] - pre_hhi

        # Paradox detection: micro TFP↑ but macro TFP unchanged
        micro_gain = result['micro_tfp_ai'] - result['micro_tfp_no_ai']
        result['paradox_score'] = (
            micro_gain > 0.01  # AI firms are more productive
            and abs(result['delta_macro_tfp']) < 0.05  # but macro TFP barely changed
        )

    return result
