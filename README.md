# Reproduction package

Regenerates every number and every figure in *Energy-Efficient Agentic AI
Systems: Measurement, Sources of Cost, and Efficiency Strategies* from the
parameter file.

## Run

    python reproduce.py --outdir out

Requires Python 3.10+, numpy, pandas, matplotlib. No other dependencies, no
network access, runs in a few seconds on a laptop.

The script prints a CHECKS table comparing each regenerated value against the
figure quoted in the manuscript and exits non-zero if any check fails. A clean
run reports `18/18 checks reproduce the manuscript`.

## Contents

| File | What it is |
|---|---|
| `agent_energy.py` | The accounting model: Equations (1) and (2) of the paper, plus the one-at-a-time and Monte Carlo uncertainty routines. |
| `parameters.csv` | Table 3. Every parameter with its value, basis, source anchor, bibkey, status (MEASURED / DERIVED / ASSUMED / SWEPT) and caveat. |
| `calibration_anchors.csv` | The 81 published values extracted from the literature, each with units, hardware, measurement method, measured-vs-estimated flag, source and caveat. Two rows are `UNFILLED`: no retrieved source supplies them. |
| `reproduce.py` | Driver. Performs the two-point calibration, computes all tables, calls the figure module, runs the checks. |
| `figures.py` | Regenerates Figures 1-4. |
| `figstyle.py` | Figure rcParams, so the figures regenerate with no external style dependency. |

## Outputs

`results_patterns.csv` (Fig 1), `results_scaling.csv` (Fig 2),
`results_strategies.csv` (Fig 3), `results_sensitivity.csv` and
`results_uncertainty.csv` (Fig 4), `results_boundaries.csv` (the B1-B5
accounting-boundary comparison in Section 3.2), and the four figures as PDF
and PNG.

## What the calibration does

`calibrate()` reads anchors ET-01 and ET-02, which report energy per output
token at 10 and at 512 output tokens for the **same** request configuration
(Llama-3.2-1B on an H200, batch 16, 4K context). Writing total request energy
as `F + m*n` and solving the two equations separates a fixed per-request cost
`F = 68.74 J` from a marginal cost `m = 0.5857 J` per decode token. Dividing `F`
by that configuration's 4096-token context gives `e_p = 0.0168 J` per prefill
token, an upper bound because `F` also contains generation setup.

Nothing else in the model is fitted.

## Honest limits, restated here so they travel with the code

- **Every output is an estimate.** The measured quantities are the inputs in
  `calibration_anchors.csv`. Absolute magnitudes inherit the reference serving
  configuration above and should not be quoted as the energy of an agent.
- **Four parameters have no source.** `p_idle_w`, `tool_energy_j`,
  `stall_s_per_step` and `cache_hit` are assumptions, flagged as such in
  `parameters.csv`. Propagated jointly they alone produce a 4.3x interval on
  energy per successful task, comparable to the 4.8x from model scale.
- **The component-level attribution is untested.** The split of trajectory
  energy between decode, prefill, host and reserved idle capacity is the
  paper's central claim and rests on the accounting structure, not on any
  measurement we could compare against.
- **The multi-agent configuration is a calibration, not a validation.** Worker
  and round counts were selected from a twelve-point grid to match a published
  multiplier; `reproduce.py` does not present that as a check.

## Operating points

The model carries a residency constraint: the justification for charging idle
power is that serving state stays resident across a stall, but resident KV blocks
are exactly what makes the next step's prefix a cache hit. A configuration
therefore cannot both re-prefill the whole transcript and charge the full idle
power of capacity held for it. `agent_energy.residency_is_consistent` enforces
`|h - (1 - u)| <= 0.25`, and `reproduce.py` asserts it for every reported point.
Two corners satisfy it:

- **OP-A** dedicated, state retained: `h = 0.9`, `u = 0`.
- **OP-B** shared, state evicted: `h = 0`, `u = 0.9`.

Every table is produced at both. The configuration used in an earlier version of
this work (`h = 0`, `u = 0`) is rejected by the constraint; `reproduce.py` checks
that it is rejected, so the defect cannot silently return.

## Changing assumptions

Every row of `parameters.csv` is read by the run. `reproduce.py` calls
`Params.assert_all_consumed()` before writing figures and aborts if any row was
not consumed. Editing any row makes the CHECKS table fail by construction: the
checks assert agreement with the manuscript's published configuration, not with
whatever the file currently holds, so a failing check after an edit is the
intended signal that you have left that configuration.

`op_cache_hit` holds the two operating points' hit rates and is read as a pair;
`u` is derived from it by the residency constraint rather than stored
separately. Strategy effect sizes (`SPEC_X` 2.5, `QUANT_X` 3.25, `DVFS_F` 0.854,
`SMALL_X` 8.6, `ROUTE_FRAC` 0.6, compaction interval 6) and pattern shapes live
in `reproduce.py`, because they define the strategies and workloads rather than
the system being modelled.

## Honest limits, carried with the code

- **Every output is an estimate.** The measured inputs are in
  `calibration_anchors.csv`; absolute magnitudes inherit a 1B-class model on an
  H200 at batch 16 and 4K context.
- **Three parameters have no source**: `p_idle_w`, `tool_energy_j`,
  `stall_s_per_step`. Propagated jointly with the swept cache hit they produce a
  4.3x interval on energy per successful task, marginally more than model scale.
- **The headline ordering is conditional.** Agent-specific sources exceed decode
  in 53% of the swept (idle power, stall time) space at OP-A and 32% at OP-B.
  `results_contour.csv` contains the full grid.
- **Which operating point a deployment occupies is not measured.** We argue from
  KV-state size that OP-A fits the reference configuration; we have no
  measurement of cache residency under an agent workload.
- **The component attribution is untested** by any comparison available to us.
