#!/usr/bin/env python3
"""Regenerate every number and figure in the revised paper from parameters.csv.

    python reproduce.py [--outdir DIR]

Prints a CHECKS table against the figures quoted in the manuscript and exits
non-zero if any disagrees. Requires numpy, pandas, matplotlib. No network.
"""
import argparse, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, dataclasses as dcs
import matplotlib; matplotlib.use("Agg")
import agent_energy as ae

COMP = ['E_prefill','E_decode','E_host','E_tool','E_idle','E_facility']
ORD  = ['Single turn','Chain of thought','Extended thinking','Self-consistency 8',
        'Multi-agent (2x3)','ReAct loop (12)','Reflexion (24)']
OP   = {'A': (0.9, 0.0), 'B': (0.0, 0.9)}      # (cache hit h, occupancy u)
SPEC_X, QUANT_X, DVFS_F, SMALL_X, ROUTE_FRAC = 2.5, 3.25, 0.854, 8.6, 0.6


class Params:
    """parameters.csv reader that tracks which rows the run consumes."""
    def __init__(self, path="parameters.csv"):
        self.df = pd.read_csv(path); self._used = set()
    def _raw(self, s):
        hit = self.df.loc[self.df.symbol == s, "value"]
        if hit.empty: raise KeyError(f"parameters.csv has no row for {s!r}")
        self._used.add(s); return str(hit.iloc[0]).strip()
    def scalar(self, s): return float(self._raw(s))
    def pair(self, s):   return [float(x) for x in self._raw(s).split(';')]
    def assert_all_consumed(self):
        unused = sorted(set(self.df.symbol) - self._used)
        if unused:
            raise AssertionError("parameters.csv rows never read: " + ", ".join(unused))


def calibrate(anchors):
    v = lambda a: float(str(anchors.loc[anchors.anchor_id == a, "value"].iloc[0]).strip())
    n1, n2, j1, j2 = 10, 512, v("ET-01"), v("ET-02")
    m = (n2*j2 - n1*j1) / (n2 - n1)
    F = n1*j1 - m*n1
    return F/4096.0, m, F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=".")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    out = lambda f: os.path.join(a.outdir, f)
    P = Params("parameters.csv")
    anchors = pd.read_csv("calibration_anchors.csv")
    e_p, e_d, F = calibrate(anchors)
    assert abs(P.scalar("e_prefill") - e_p) < 5e-4
    assert abs(P.scalar("e_decode")  - e_d) < 5e-4
    def mk(u, **kw):
        d = dict(name="ref", e_prefill=e_p, e_decode=e_d,
                 alpha_host=P.scalar("alpha_host"), p_idle_w=P.scalar("p_idle_w"),
                 pue=P.scalar("PUE"), sigma=P.scalar("sigma"), occupancy=u)
        d.update(kw)
        return ae.Serving(**d)
    TOOL_J, STALL_S = P.scalar("tool_energy_j"), P.scalar("stall_s_per_step")
    tl = dict(tool_tokens_per_step=700, tool_calls_per_step=1.0,
              tool_energy_j=TOOL_J, stall_s_per_step=STALL_S)
    PAT = {
     "Single turn":        ae.Pattern("Single turn", steps=1, decode_per_step=300),
     "Chain of thought":   ae.Pattern("Chain of thought", steps=1, decode_per_step=900),
     "Extended thinking":  ae.Pattern("Extended thinking", steps=1, decode_per_step=1332),
     "Self-consistency 8": ae.Pattern("Self-consistency 8", steps=1, branches=8,
                                      decode_per_step=900),
     "Multi-agent (2x3)":  ae.Pattern("Multi-agent (2x3)", steps=3, agents=2,
                                      decode_per_step=400, tool_tokens_per_step=300,
                                      tool_calls_per_step=1.0, tool_energy_j=TOOL_J,
                                      stall_s_per_step=STALL_S),
     "ReAct loop (12)":    ae.Pattern("ReAct loop (12)", steps=12, decode_per_step=250, **tl),
     "Reflexion (24)":     ae.Pattern("Reflexion (24)", steps=24, decode_per_step=250, **tl),
    }
    P.pair("op_cache_hit")          # the two operating points, declared in the file

    # ---- patterns at both operating points -------------------------------
    rows = []
    for op, (h, u) in OP.items():
        sv = mk(u)
        for k, p in PAT.items():
            pp = dcs.replace(p, cache_hit=h); r = ae.trajectory_energy(pp, sv)
            assert ae.residency_is_consistent(pp, sv), f"{op}/{k} violates residency"
            rows.append({'op': op, 'pattern': k, **{c: r[c] for c in COMP},
                         'E_attempt': r['E_attempt']})
    res = pd.DataFrame(rows)
    for op in OP:
        s = res[res.op == op]
        res.loc[res.op == op, 'x_single'] = (s.E_attempt/s.E_attempt.iloc[0]).values
    assert (res[COMP].sum(axis=1) - res.E_attempt).abs().max() < 1e-6
    res.to_csv(out("results_patterns.csv"), index=False)
    sh = {op: 100*res[res.op==op].set_index('pattern')[COMP]
                 .div(res[res.op==op].set_index('pattern').E_attempt, axis=0) for op in OP}
    g = lambda op, pat, c: sh[op].loc[pat, c]

    # ---- the published midpoint must be rejected -------------------------
    mid_ok = ae.residency_is_consistent(dcs.replace(PAT['ReAct loop (12)'], cache_hit=0.0), mk(0.0))

    # ---- step sweep ------------------------------------------------------
    Ss = np.arange(1, 41); sc = []
    for op, (h, u) in OP.items():
        sv = mk(u)
        for S in Ss:
            r = ae.trajectory_energy(dcs.replace(PAT['ReAct loop (12)'], steps=int(S),
                                                 cache_hit=h), sv)
            sc.append({'op': op, 'steps': int(S), 'E': r['E_attempt'],
                       'prefill_share': r['E_prefill']/r['E_attempt']})
    scal = pd.DataFrame(sc); scal.to_csv(out("results_scaling.csv"), index=False)
    expo = {op: np.polyfit(np.log(scal[(scal.op==op)&(scal.steps>=4)].steps),
                           np.log(scal[(scal.op==op)&(scal.steps>=4)].E), 1)[0] for op in OP}

    # ---- strategies at OP-A, with the residency coupling -----------------
    base_p, base_sv = dcs.replace(PAT['ReAct loop (12)'], cache_hit=0.9), mk(0.0)
    E0 = ae.trajectory_energy(base_p, base_sv)['E_attempt']
    nm = {'release':'Release stalled capacity to co-tenants',
          'compact':'Transcript compaction (every 6)','spec':'Speculative decoding (2.5x)',
          'quant':'4-bit quantisation (3.25x)','dvfs':'Kernel-level DVFS (14.6%)',
          'route':'Route 60% of calls to a small model'}
    def apply(ss):
        p, sv, kw = base_p, base_sv, {}
        for s in ss:
            if s=='release':  p = dcs.replace(p, cache_hit=0.0); sv = dcs.replace(sv, occupancy=0.9)
            elif s=='compact':p = dcs.replace(p, summarise_every=6, summary_tokens=500)
            elif s=='spec':   sv = dcs.replace(sv, e_decode=sv.e_decode/SPEC_X)
            elif s=='quant':  sv = dcs.replace(sv, sigma=sv.sigma/QUANT_X)
            elif s=='dvfs':   sv = dcs.replace(sv, e_prefill=sv.e_prefill*DVFS_F,
                                               e_decode=sv.e_decode*DVFS_F)
            elif s=='route':  kw = {'sv_small': dcs.replace(sv, sigma=sv.sigma/SMALL_X),
                                    'small_frac': ROUTE_FRAC}
        return ae.trajectory_energy(p, sv, **kw)['E_attempt']
    srow, cum = [], []
    for s in ['release','compact','spec','quant','dvfs','route']:
        ei = apply([s]); cum.append(s)
        srow.append({'strategy': nm[s], 'E_alone': ei, 'saving_alone_pct': 100*(1-ei/E0),
                     'saving_cumulative_pct': 100*(1-apply(cum)/E0)})
    strat = pd.DataFrame(srow); strat.to_csv(out("results_strategies.csv"), index=False)
    naive = 100*(1 - ae.trajectory_energy(base_p, dcs.replace(base_sv, occupancy=0.9))['E_attempt']/E0)

    # ---- uncertainty -----------------------------------------------------
    unc = [ae.Unc('e_decode',0.27,0.72,'Marginal decode energy'),
           ae.Unc('e_prefill',0.0084,e_p,'Prefill energy per token'),
           ae.Unc('alpha_host',0.695,0.818,'Host overhead share',log=False),
           ae.Unc('p_idle_w',50.,300.,'Idle reserved power (unanchored)'),
           ae.Unc('pue',1.08,1.29,'Facility PUE'),
           ae.Unc('sigma',1.,8.6,'Model-scale multiplier')]
    ref_s = dcs.replace(PAT['ReAct loop (12)'], cache_hit=0.9,
                        success_rate=P.scalar("success_rate"))
    sens = pd.DataFrame(ae.oat(ref_s, mk(0.0), unc)); sens.to_csv(out("results_sensitivity.csv"), index=False)
    S_ = {'e_decode':(0.27,0.72,1),'e_prefill':(0.0084,e_p,1),'alpha_host':(0.695,0.818,0),
          'p_idle_w':(50.,300.,1),'pue':(1.08,1.29,0),'sigma':(1.,8.6,1)}
    Pp = {'tool_energy_j':(2.,200.,1),'stall_s_per_step':(0.5,10.,1),'cache_hit':(0.0,0.9,0)}
    def mc(ks, kp, n=6000, seed=13):
        rg = np.random.default_rng(seed)
        dr = lambda lo,hi,lg: np.exp(rg.uniform(np.log(lo),np.log(hi),n)) if lg else rg.uniform(lo,hi,n)
        ds = {k: dr(*S_[k]) for k in ks}; dp = {k: dr(*Pp[k]) for k in kp}
        o = np.empty(n)
        for i in range(n):
            o[i] = ae.epst(dcs.replace(ref_s, **{k: float(v[i]) for k,v in dp.items()}),
                           dcs.replace(mk(0.0), **{k: float(v[i]) for k,v in ds.items()}))
        q = np.percentile(o,[5,50,95]); return o, q, q[2]/q[0]
    mc9,q9,r9 = mc(list(S_), list(Pp)); _,_,r6 = mc(list(S_), [])
    _,_,rs = mc(['sigma'], []); _,_,ru = mc(['p_idle_w'], list(Pp))
    pd.DataFrame([{'analysis':'6 published-range','p95_over_p5':r6},
                  {'analysis':'all 9','p5_kJ':q9[0]/1e3,'median_kJ':q9[1]/1e3,
                   'p95_kJ':q9[2]/1e3,'p95_over_p5':r9},
                  {'analysis':'model scale alone','p95_over_p5':rs},
                  {'analysis':'four unanchored alone','p95_over_p5':ru}]
                 ).to_csv(out("results_uncertainty.csv"), index=False)

    # ---- (pi, t) region in which the headline ordering survives -----------
    grid = []
    for op,(h,u) in OP.items():
        for pi in np.linspace(25,300,24):
            for t in np.logspace(-1,1,24):
                r = ae.trajectory_energy(dcs.replace(PAT['ReAct loop (12)'], cache_hit=h,
                                                     stall_s_per_step=float(t)),
                                         mk(u, p_idle_w=float(pi)))
                grid.append({'op':op,'pi':pi,'t':t,
                             'margin':100*(r['E_prefill']+r['E_idle']-r['E_decode'])/r['E_attempt']})
    gr = pd.DataFrame(grid); gr.to_csv(out("results_contour.csv"), index=False)
    hold = {op: 100*(gr[gr.op==op].margin>0).mean() for op in OP}

    # ---- break-even ------------------------------------------------------
    be = []
    for op,(h,u) in OP.items():
        sv = mk(u); e1 = ae.trajectory_energy(dcs.replace(PAT['Single turn'], cache_hit=h), sv)['E_attempt']
        for k,p in PAT.items():
            if k == 'Single turn': continue
            be.append({'op':op,'pattern':k,
                       'k_x': ae.trajectory_energy(dcs.replace(p,cache_hit=h),sv)['E_attempt']/e1})
    bed = pd.DataFrame(be); bed.to_csv(out("results_breakeven.csv"), index=False)
    kx = lambda op,pat: bed[(bed.op==op)&(bed.pattern==pat)].k_x.iloc[0]

    P.assert_all_consumed()
    import figures; figures.make_all(a.outdir, res, scal, strat, sens, mc9, q9, gr, bed, expo)

    E = lambda op,pat: res[(res.op==op)&(res.pattern==pat)].E_attempt.iloc[0]
    X = lambda op,pat: res[(res.op==op)&(res.pattern==pat)].x_single.iloc[0]
    checks = [
      ("fixed per-request energy F (J)", F, 68.74, 0.01),
      ("e_prefill (J/token)", e_p, 0.0168, 0.0005),
      ("e_decode (J/token)", e_d, 0.5857, 0.0005),
      ("decode:prefill per-token ratio", e_d/e_p, 34.9, 0.1),
      ("published midpoint rejected by residency", 0.0 if mid_ok else 1.0, 1.0, 0.0),
      ("OP-A ReAct(12) energy (kJ)", E('A','ReAct loop (12)')/1e3, 7.4, 0.05),
      ("OP-A ReAct(12) multiple of single turn", X('A','ReAct loop (12)'), 19.7, 0.1),
      ("OP-A ReAct(12) decode share (%)", g('A','ReAct loop (12)','E_decode'), 23.9, 0.1),
      ("OP-A ReAct(12) idle share (%)", g('A','ReAct loop (12)','E_idle'), 40.8, 0.1),
      ("OP-A ReAct(12) prefill share (%)", g('A','ReAct loop (12)','E_prefill'), 4.3, 0.1),
      ("OP-A Reflexion(24) energy (kJ)", E('A','Reflexion (24)')/1e3, 15.1, 0.05),
      ("OP-B ReAct(12) energy (kJ)", E('B','ReAct loop (12)')/1e3, 6.4, 0.05),
      ("OP-B ReAct(12) decode share (%)", g('B','ReAct loop (12)','E_decode'), 27.6, 0.1),
      ("OP-B ReAct(12) prefill share (%)", g('B','ReAct loop (12)','E_prefill'), 21.6, 0.1),
      ("growth exponent OP-A", expo['A'], 1.04, 0.01),
      ("growth exponent OP-B", expo['B'], 1.39, 0.01),
      ("OP-B prefill share at S=40 (%)",
       100*scal[(scal.op=='B')&(scal.steps==40)].prefill_share.iloc[0], 35.9, 0.1),
      ("release-capacity saving, coupled (%)", strat.saving_alone_pct.iloc[0], 13.6, 0.1),
      ("release-capacity saving, cache ignored (%)", naive, 40.0, 0.1),
      ("overstatement factor of the uncoupled view", naive/strat.saving_alone_pct.iloc[0], 2.9, 0.05),
      ("sum of individual strategy savings (%)", strat.saving_alone_pct.sum(), 112.0, 0.2),
      ("cumulative strategy saving (%)", strat.saving_cumulative_pct.iloc[-1], 87.2, 0.1),
      ("MC p95/p5, 6 corrected ranges", r6, 5.11, 0.05),
      ("MC p95/p5, all nine", r9, 6.28, 0.05),
      ("interval from model scale alone", rs, 4.22, 0.05),
      ("interval from four unanchored alone", ru, 4.29, 0.05),
      ("headline holds in x% of (pi,t) space, OP-A", hold['A'], 53.0, 1.0),
      ("headline holds in x% of (pi,t) space, OP-B", hold['B'], 32.0, 1.0),
      ("break-even multiple, ReAct(12) at OP-A", kx('A','ReAct loop (12)'), 19.7, 0.1),
      ("break-even multiple, Reflexion(24) at OP-A", kx('A','Reflexion (24)'), 40.5, 0.1),
    ]
    print(f"\n{'CHECK':46s} {'regenerated':>12s} {'in paper':>10s}  status")
    print("-"*84); bad = 0
    for n_, got, want, tol in checks:
        ok = abs(got-want) <= tol; bad += (not ok)
        print(f"{n_:46s} {got:12.4f} {want:10.4f}  {'ok' if ok else 'MISMATCH'}")
    print("-"*84); print(f"{len(checks)-bad}/{len(checks)} checks reproduce the manuscript")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
