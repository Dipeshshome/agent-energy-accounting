
"""
Trajectory-level energy accounting model for multi-step LLM agents.

ACCOUNTING BOUNDARY (see the paper's reporting standard):
  B1 accelerator (device) energy          included
  B2 host CPU/DRAM/NIC/chassis overhead   included, as alpha_host x B1
  B3 reserved-but-idle accelerator        included, switchable (charge_idle)
  B4 tool / environment execution         included, per tool call
  B5 facility overhead (PUE)              included, as a multiplier
  B6 network transit, client device       EXCLUDED
  B7 embodied / manufacturing             EXCLUDED

All outputs are ESTIMATES. Parameter provenance is in parameters.csv; the two
per-token energies are fitted from one measured source at a matched request
configuration, and p_idle_w is the model's single unanchored parameter.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np


# ---------------------------------------------------------------- serving ---

@dataclass(frozen=True)
class Serving:
    """A serving configuration, expressed in energy per token."""
    name: str
    e_prefill: float      # J per prefill token, device level, reference model
    e_decode: float       # J per decode token, device level, reference model
    alpha_host: float     # host energy as a fraction of device energy
    p_idle_w: float       # W drawn by the reserved accelerator set while stalled
    pue: float            # facility power usage effectiveness
    sigma: float = 1.0    # model-scale energy multiplier vs the reference model
    occupancy: float = 0.0  # fraction of a stall during which the reserved
                            # accelerators serve other tenants' work. 0 = a
                            # dedicated single-tenant deployment (the full idle
                            # charge falls on this trajectory); 1 = a saturated
                            # shared fleet (no idle energy is attributable).

    @property
    def ep(self) -> float:
        return self.sigma * self.e_prefill

    @property
    def ed(self) -> float:
        return self.sigma * self.e_decode


# ---------------------------------------------------------------- pattern ---

@dataclass(frozen=True)
class Pattern:
    """One agent control-flow pattern, per attempt at a task."""
    name: str
    steps: int                      # sequential LLM calls per attempt
    branches: int = 1               # parallel samples drawn per step
    agents: int = 1                 # cooperating workers, each with its own context
    sys_tokens: int = 1200          # system prompt + tool schemas
    task_tokens: int = 400          # task statement
    decode_per_step: int = 400      # output tokens per call, incl. reasoning tokens
    tool_tokens_per_step: int = 0   # tokens of tool output appended to the context
    tool_calls_per_step: float = 0.0
    tool_energy_j: float = 0.0      # J per tool call
    stall_s_per_step: float = 0.0   # s stalled on a tool with capacity reserved
    cache_hit: float = 0.0          # prefix-cache hit rate on re-ingested context
    success_rate: float = 1.0       # P(task solved in one attempt)
    charge_idle: bool = True
    summarise_every: int = 0        # 0 = never; else compact the transcript every k steps
    summary_tokens: int = 500


def residency_is_consistent(p: Pattern, sv: Serving, tol: float = 0.25) -> bool:
    """Is this (cache hit, idle charge) pair physically realisable?

    The justification for charging idle power is that the trajectory's serving
    state stays resident across a stall. But resident KV blocks are exactly what
    makes the next step's prefix a cache hit. So a configuration cannot both
    re-prefill the whole transcript (h near 0) and charge the full idle power of
    capacity held for it (occupancy near 0). The two corners that are realisable
    are (h high, occupancy low) and (h low, occupancy high); the midpoint is not.
    """
    if p.stall_s_per_step <= 0 or not p.charge_idle:
        return True                      # no idle charge, nothing to reconcile
    # h is the fraction of state retained; (1 - occupancy) is the fraction of
    # capacity held for this trajectory. They describe the same retention and
    # must agree.
    return abs(p.cache_hit - (1.0 - sv.occupancy)) <= tol


def context_tokens(p: Pattern) -> tuple[float, float]:
    """Billed prefill and decode tokens for ONE worker's context.

    Step s re-sends the accumulated transcript. Tokens appended since the
    previous call are new and always billed; the remainder is reused and billed
    at (1 - cache_hit). Parallel branches share the prefix, so only the
    uncached fraction is re-prefilled per extra branch.
    """
    ctx = p.sys_tokens + p.task_tokens
    new = ctx
    pf = dc = 0.0
    for s in range(1, p.steps + 1):
        reused = ctx - new
        branch_factor = 1.0 + (p.branches - 1) * (1.0 - p.cache_hit)
        pf += (new + (1.0 - p.cache_hit) * reused) * branch_factor
        dc += p.decode_per_step * p.branches
        new = p.decode_per_step + p.tool_tokens_per_step
        ctx += new
        if p.summarise_every and s % p.summarise_every == 0:
            ctx = p.sys_tokens + p.task_tokens + p.summary_tokens
            new = ctx          # the compacted transcript is new to the cache
    return pf, dc


def trajectory_energy(p: Pattern, sv: Serving, sv_small: Serving | None = None,
                      small_frac: float = 0.0) -> dict:
    """Energy of one attempt at the task, in joules, by cost source."""
    pf, dc = context_tokens(p)

    if p.agents > 1:
        # workers run the pattern; a coordinator ingests every worker's output
        coord = replace(p, agents=1, branches=1,
                        tool_tokens_per_step=int(p.decode_per_step * p.agents),
                        stall_s_per_step=0.0)
        cpf, cdc = context_tokens(coord)
        pf, dc = pf * p.agents + cpf, dc * p.agents + cdc

    def blend(hi: float, lo: float) -> float:
        return hi if (sv_small is None or small_frac <= 0) else \
            (1.0 - small_frac) * hi + small_frac * lo

    ep = blend(sv.ep, sv_small.ep if sv_small else 0.0)
    ed = blend(sv.ed, sv_small.ed if sv_small else 0.0)

    e_prefill, e_decode = pf * ep, dc * ed
    e_device = e_prefill + e_decode
    e_host = sv.alpha_host * e_device
    n_calls = p.steps * p.agents
    e_tool = n_calls * p.tool_calls_per_step * p.tool_energy_j
    stall = p.steps * p.stall_s_per_step
    e_idle = ((1.0 - sv.occupancy) * sv.p_idle_w * stall
              if (p.charge_idle and stall > 0) else 0.0)

    e_it = e_device + e_host + e_tool + e_idle
    # components are reported at IT level; the facility term is the PUE
    # increment, so the six components sum exactly to E_attempt
    return {
        "prefill_tokens": pf, "decode_tokens": dc,
        "E_prefill": e_prefill, "E_decode": e_decode,
        "E_host": e_host, "E_tool": e_tool,
        "E_idle": e_idle, "E_facility": e_it * (sv.pue - 1.0),
        "E_attempt": e_it * sv.pue,
        "E_per_success": e_it * sv.pue / max(p.success_rate, 1e-9),
    }


def epst(p: Pattern, sv: Serving, **kw) -> float:
    return trajectory_energy(p, sv, **kw)["E_per_success"]


# ------------------------------------------------------------ uncertainty ---

@dataclass
class Unc:
    field: str
    low: float
    high: float
    label: str = ""
    log: bool = True

    def draw(self, rng, n):
        if self.log:
            return np.exp(rng.uniform(np.log(self.low), np.log(self.high), n))
        return rng.uniform(self.low, self.high, n)


def monte_carlo(p: Pattern, sv: Serving, unc: list[Unc], n: int = 4000, seed: int = 0,
                **kw) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = {u.field: u.draw(rng, n) for u in unc}
    out = np.empty(n)
    for i in range(n):
        out[i] = epst(p, replace(sv, **{f: float(v[i]) for f, v in draws.items()}), **kw)
    return out


def oat(p: Pattern, sv: Serving, unc: list[Unc], **kw) -> list[dict]:
    base = epst(p, sv, **kw)
    rows = []
    for u in unc:
        lo = epst(p, replace(sv, **{u.field: u.low}), **kw)
        hi = epst(p, replace(sv, **{u.field: u.high}), **kw)
        rows.append({"parameter": u.label or u.field, "low": min(lo, hi),
                     "high": max(lo, hi), "base": base,
                     "swing": (max(lo, hi) - min(lo, hi)) / base})
    return sorted(rows, key=lambda r: -r["swing"])
