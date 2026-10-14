"""Entity state definitions for AI Productivity Paradox ABM v2.0.

Three entity types:
  FirmState   — enterprises with capital, labor, AI investment, product quality
  MarketState — aggregate market conditions (not an agent)
  SimState    — top-level simulation state

v2.0 changes from v0.2:
  FirmState: removed has_ai (binary) and industry (exogenous).
             Added: product_quality, ai_investment, ai_strategy, market_share, effective_markup
  MarketState: added hhi, unemployment_rate, demand_externalit_shock, etc.
  WorkerState will be added in Phase 3C (layer 3: labor market).
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

from .config import cfg as _cfg, merge_configs


class FirmState(NamedTuple):
    alive: jax.Array              # bool[F]
    K: jax.Array                  # float32[F] — capital
    L: jax.Array                  # float32[F] — labor (workers)
    A_base: jax.Array             # float32[F] — baseline TFP (heterogeneous)
    sigma_i: jax.Array            # float32[F] — per-firm substitution elasticity
    adoption_readiness: jax.Array  # float32[F] — AI adoption readiness [0, 1]
    product_quality: jax.Array    # float32[F] — product differentiation quality [0, inf)
    ai_investment: jax.Array      # float32[F] — current period AI investment amount
    ai_strategy: jax.Array        # float32[F] — strategy mix [0=cost, 1=quality]
    ai_experience: jax.Array     # float32[F] — AI learning accumulation [0, 1]
    cash: jax.Array               # float32[F] — financial reserves
    Q: jax.Array                  # float32[F] — current period output
    revenue: jax.Array           # float32[F] — current period revenue
    cost: jax.Array               # float32[F] — current period total cost
    profit: jax.Array             # float32[F] — current period profit
    market_share: jax.Array      # float32[F] — current market share (from consumer choice)
    effective_markup: jax.Array   # float32[F] — implied markup from market position
    age: jax.Array                # int32[F] — ticks since birth


class MarketState(NamedTuple):
    total_Q: jax.Array                # float32 — aggregate output
    agg_price: jax.Array              # float32 — aggregate price level
    wage_rate: jax.Array              # float32 — equilibrium wage
    ai_adoption_rate: jax.Array       # float32 — fraction of firms with significant AI investment
    hhi: jax.Array                    # float32 — Herfindahl concentration index
    labor_share: jax.Array            # float32 — wages / total revenue
    demand_externalit_shock: jax.Array # float32 — current demand externality shock magnitude
    market_size_effective: jax.Array   # float32 — market_size adjusted for demand feedback
    unemployment_rate: jax.Array      # float32 — fraction of workers unemployed
    avg_wage: jax.Array              # float32 — average wage across employed workers


class WorkerState(NamedTuple):
    alive: jax.Array              # bool[W] — active in labor force
    age: jax.Array                # int32[W] — worker age (in ticks, 1 tick ~ 1 month)
    skill: jax.Array              # float32[W] — productivity skill level
    adaptability: jax.Array      # float32[W] — speed of learning new skills
    employer: jax.Array           # int32[W] — current employer firm index (-1 = unemployed)
    wage: jax.Array              # float32[W] — current wage
    ai_compatibility: jax.Array   # float32[W] — ability to work with AI [0, 1]
    unemployment_duration: jax.Array  # float32[W] — ticks since last employment


class SimState(NamedTuple):
    firms: FirmState
    workers: WorkerState
    market: MarketState
    tick: int
    key: jax.Array


# ============================================================
# pytree registration
# ============================================================

def _flatten_firm(xs):
    return tuple(xs), None

def _unflatten_firm(aux, children):
    return FirmState(*children)

jax.tree_util.register_pytree_node(FirmState, _flatten_firm, _unflatten_firm)


def _flatten_market(xs):
    return tuple(xs), None

def _unflatten_market(aux, children):
    return MarketState(*children)

jax.tree_util.register_pytree_node(MarketState, _flatten_market, _unflatten_market)


def _flatten_worker(xs):
    return tuple(xs), None

def _unflatten_worker(aux, children):
    return WorkerState(*children)

jax.tree_util.register_pytree_node(WorkerState, _flatten_worker, _unflatten_worker)


def _flatten_sim(xs):
    return (xs.firms, xs.workers, xs.market, xs.tick, xs.key), None

def _unflatten_sim(aux, children):
    f, w, m, tick, k = children
    return SimState(f, w, m, tick, k)

jax.tree_util.register_pytree_node(SimState, _flatten_sim, _unflatten_sim)


# ============================================================
# Initialization
# ============================================================

def init_firms(config: dict, key: jax.Array) -> FirmState:
    F = _cfg(config, 'num_firms')
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    # Audit 2026-10-07 (external review S1): K/L/cash used to be drawn as
    # exp(N(0, sd)) * mean — treating the config sd (3/15/5) as a LOGNORMAL
    # sigma, which put 50-88% of the mass at the clip bounds (degenerate
    # two-point "heterogeneity" and a mass-extinction transient at t<75).
    # Now drawn as normal(mean, sd), consistent with the entry draws in
    # step.py S7 and with the config comments.
    K = jax.random.normal(k1, (F,), dtype=jnp.float32) \
        * jnp.float32(_cfg(config, 'initial_K_std')) \
        + jnp.float32(_cfg(config, 'initial_K_mean'))
    K = jnp.clip(K, 0.5, 100.0)

    L = jax.random.normal(k2, (F,), dtype=jnp.float32) \
        * jnp.float32(_cfg(config, 'initial_L_std')) \
        + jnp.float32(_cfg(config, 'initial_L_mean'))
    L = jnp.clip(L, 5.0, 200.0)

    # A_base stays lognormal (positive productivity; sd=0.2 as log-sigma).
    A_base = jnp.exp(jax.random.normal(k3, (F,)) * _cfg(config, 'A_base_std')
                ) * _cfg(config, 'A_base_mean')
    A_base = jnp.clip(A_base, 0.3, 3.0)

    cash = jax.random.normal(k4, (F,), dtype=jnp.float32) \
        * jnp.float32(_cfg(config, 'initial_cash_std')) \
        + jnp.float32(_cfg(config, 'initial_cash_mean'))
    cash = jnp.clip(cash, 1.0, 100.0)

    # Per-firm substitution elasticity from A_base + noise
    sigma_base = jnp.float32(_cfg(config, 'sigma_base'))
    sigma_spread = jnp.float32(_cfg(config, 'sigma_firm_spread'))
    sigma_A_sens = jnp.float32(_cfg(config, 'sigma_firm_A_sensitivity'))
    sigma_min = jnp.float32(_cfg(config, 'sigma_min'))
    sigma_max = jnp.float32(_cfg(config, 'sigma_max'))

    sigma_i = sigma_base - sigma_A_sens * (A_base - jnp.float32(1.0)) \
              + jax.random.normal(k5, (F,), dtype=jnp.float32) * sigma_spread
    sigma_i = jnp.clip(sigma_i, sigma_min, sigma_max)

    F_s = _cfg(config, 'num_firms')
    return FirmState(
        alive=jnp.ones(F_s, dtype=jnp.bool_),
        K=K.astype(jnp.float32),
        L=L.astype(jnp.float32),
        A_base=A_base.astype(jnp.float32),
        sigma_i=sigma_i.astype(jnp.float32),
        # Audit 2026-10-07 (external review B2): readiness used to start at
        # exactly 0 with a SCALAR growth increment, so every firm crossed the
        # gate on the same tick (a global clock, not firm-level diffusion).
        # Initial heterogeneous prior exposure U(0, 0.3) staggers the gate.
        adoption_readiness=jax.random.uniform(
            k6, (F_s,), dtype=jnp.float32) * jnp.float32(0.3),
        product_quality=jnp.ones(F_s, dtype=jnp.float32),       # start at 1.0
        ai_investment=jnp.zeros(F_s, dtype=jnp.float32),        # no investment initially
        ai_strategy=jnp.full(F_s, 0.5, dtype=jnp.float32),      # neutral strategy
        ai_experience=jnp.zeros(F_s, dtype=jnp.float32),
        cash=cash.astype(jnp.float32),
        Q=jnp.zeros(F_s, dtype=jnp.float32),
        revenue=jnp.zeros(F_s, dtype=jnp.float32),
        cost=jnp.zeros(F_s, dtype=jnp.float32),
        profit=jnp.zeros(F_s, dtype=jnp.float32),
        market_share=jnp.full(F_s, 1.0/F_s, dtype=jnp.float32),  # equal initially
        effective_markup=jnp.full(F_s, 1.5, dtype=jnp.float32),  # initial guess
        age=jnp.zeros(F_s, dtype=jnp.int32),
    )


def init_market(config: dict) -> MarketState:
    return MarketState(
        total_Q=jnp.float32(1.0),
        agg_price=jnp.float32(1.0),
        wage_rate=jnp.float32(1.0),
        ai_adoption_rate=jnp.float32(0.0),
        hhi=jnp.float32(0.0),
        labor_share=jnp.float32(0.5),
        demand_externalit_shock=jnp.float32(0.0),
        market_size_effective=jnp.float32(_cfg(config, 'market_size')),
        unemployment_rate=jnp.float32(0.0),
        avg_wage=jnp.float32(1.0),
    )


def init_workers(config: dict, key: jax.Array) -> WorkerState:
    W = _cfg(config, 'num_workers')
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Age: uniformly distributed between 22 and 62 working-age years.
    # Stored in MONTHS (1 tick = 1 month) — audit 2026-10-07 fixed the old
    # unit clash where ages in years were compared against retirement_age
    # once per tick, making everyone retire after <=65 ticks (~5 sim-years).
    age_min = _cfg(config, 'age_new_worker')
    age_max = _cfg(config, 'retirement_age') - 3  # retire at 65, so max start ~62
    age_years = jax.random.randint(k1, (W,), age_min, age_max, dtype=jnp.int32)
    age = age_years * 12

    # Skill: log-normal, centered around 1.0
    skill = jnp.exp(jax.random.normal(k2, (W,)) * jnp.float32(0.3))
    skill = jnp.clip(skill, jnp.float32(0.2), jnp.float32(5.0))

    # Adaptability: random, slightly correlated with youth
    adaptability = jax.random.uniform(k3, (W,), dtype=jnp.float32) * jnp.float32(0.8) + jnp.float32(0.2)
    # Younger workers slightly more adaptable
    age_factor = jnp.float32(1.0) - (age_years.astype(jnp.float32) - age_min) / jnp.float32(age_max - age_min) * jnp.float32(0.3)
    adaptability = adaptability * age_factor
    adaptability = jnp.clip(adaptability, jnp.float32(0.1), jnp.float32(1.0))

    # AI compatibility: random, correlated with adaptability and skill
    ai_compat = (adaptability * jnp.float32(0.4) + skill / jnp.float32(5.0) * jnp.float32(0.3)
                  + jax.random.uniform(k4, (W,), dtype=jnp.float32) * jnp.float32(0.3))
    ai_compat = jnp.clip(ai_compat, jnp.float32(0.0), jnp.float32(1.0))

    return WorkerState(
        alive=jnp.ones(W, dtype=jnp.bool_),
        age=age,
        skill=skill.astype(jnp.float32),
        adaptability=adaptability.astype(jnp.float32),
        employer=jnp.full(W, -1, dtype=jnp.int32),  # all unemployed initially
        wage=jnp.zeros(W, dtype=jnp.float32),
        ai_compatibility=ai_compat.astype(jnp.float32),
        unemployment_duration=jnp.full(W, jnp.float32(1.0), dtype=jnp.float32),  # start with 1 tick
    )


def init_simulation(config: dict, seed: int = 42) -> SimState:
    key = jax.random.PRNGKey(seed)
    k1, k2, _ = jax.random.split(key, 3)
    firms = init_firms(config, k1)
    workers = init_workers(config, k2)
    market = init_market(config)
    return SimState(firms=firms, workers=workers, market=market, tick=0, key=key)
