# Replication Package — Fiscal Multiplier SVAR-IV

## Run order

```
a_main_irfs.m       →  main text IRFs  +  saves results/sim.mat
b_appendix_irfs.m   →  appendix robustness IRFs
d_simulation_msw.m  →  MSW + delta-method MC coverage  (requires results/sim.mat)
```

## Outputs

| Script | Files saved to results/ |
|---|---|
| `a_` | `IRF.eps`, `IRF.png`, `sim.mat` |
| `b_` | `IRF_2.eps`, `IRF_2.png` |
| `d_` | `d_coverage_noncum.{eps,png}`, `d_coverage_cum.{eps,png}`, `d_simulation_results.mat` |

## Main vs appendix

| | Main (`a_`) | Appendix (`b_`) |
|---|---|---|
| Growth | 1-period | 2-period |
| Instrument | `Dpredict_spend` | `Dpredict_spend2` |
| VAR lags | p = 1 | p = 2 |
| inst lags | 1 | 1:2 |

## Simulation (`d_`)

Loads `results/sim.mat` from `a_`. Replicates the MSW MC design:
structural matrix B decomposed from Σ via `optimizeSigma`, instrument
generated as `z = θ₀(1)·ε₁ + noise` with strength controlled by `auxparamMC`.
Coverage tracked for MSW (CS^AR) and delta-method (CS^plug-in), cumulative
and non-cumulative. Requires the **Global Optimization Toolbox**.

## Requirements

MATLAB R2021b+, Statistics and Machine Learning Toolbox,
Global Optimization Toolbox (`d_` only).

## Structure

```
replication/
├── +pvar/               ← SVAR-IV toolbox
├── matlab_functions/    ← MSWfunction, RForm_PVAR, MARep, etc.
├── data/
│   ├── regionfiscal.csv
│   └── regionfiscal_appendix.csv
├── results/             ← created at runtime
├── a_main_irfs.m
├── b_appendix_irfs.m
├── d_simulation_msw.m
└── README.md
```
