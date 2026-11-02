# Replication package

"Identification, estimation and inference in panel vector autoregressions using external instruments"

This package reproduces every figure and table of the paper and of its supplementary material. All code is Python.

Availability: this repository; the archived version is on Zenodo, DOI TODO. License: Creative Commons Attribution 4.0
International (see `LICENSE.md`).

Data availability statement: the only data file is an extract from the public replication package of Chodorow-Reich,
Nenov and Simsek (2021). It is not included in this repository: download it as explained in
`applications/04_chodorowreich/data/DATA_SOURCE.md` and place it in that folder before running step 1 or 2. The
stored results of the paper (`results_paper/`) are included, so every table and figure can be checked without the data.

## Contents

| Folder | Contents |
|---|---|
| `package/pvariv/` | The estimation and inference library used throughout: PVAR-IV by the within estimator with the half-panel jackknife correction or by FOD-GMM, impulse responses and cumulated responses, Anderson-Rubin and CLR sets, one- and two-way clustered covariances, diagnostics. `pvariv.spillover` adds the extension of Section S5 of the supplement (exposures through a known weight matrix); it is not used for the results of the paper. The scripts add `package/` to the Python path, so no installation is needed. |
| `applications/04_chodorowreich/` | The empirical application (Section 7): data, code, and the results reported in the paper (`results_paper/`). |
| `simulations/` | The Monte Carlo simulations (Section 6 and Sections S3-S4 of the supplement): code and the results reported in the paper (`results_paper/`). |
| `paper/` | LaTeX sources of the paper and the supplement, with the figures and tables they include. |

## Software

Python 3.9 or later with `numpy`, `scipy`, `pandas` and `matplotlib` (`pip install -r requirements.txt`).
TODO: state the hardware and the running times of the two steps below.

## Data

`applications/04_chodorowreich/data/row9_dta.xpt`, from the replication package of Chodorow-Reich, Nenov and Simsek
(2021, AER), not included here. See `applications/04_chodorowreich/data/DATA_SOURCE.md` for where to download it.

## How to reproduce the results

Run the two steps in this order, from the folders indicated, keeping the folder layout of the package:
`simulations/` reads the data and the data module of `applications/`. The results reported in the paper are stored in
the `results_paper` folders. A new run writes to the `results` folders, so the two can be compared file by file.

### 1. Empirical application (Figures 1-2, Table 3, Tables S1-S2)

```
cd applications/04_chodorowreich/python
python run_main_application.py --growth
```

The script first replicates Table 3, rows 9 and 10, of Chodorow-Reich, Nenov and Simsek (2021) and their local
projections (stored in `results/main_application/lp_iv.csv`; add `--reuse-lp` to read them instead of recomputing
them), and then estimates the PVAR-IV in the stock wealth shock and payroll growth for every specification of the
paper. `--figures-only` redraws the figures and tables from the csv files already in the output folder, without
estimating. Outputs, in `applications/04_chodorowreich/results/main_application_growth/`:

| File | In the paper |
|---|---|
| `fig_main.png` | Figure 1 |
| `fig_robust.png` | Figure 2 |
| `table_shock.tex` | Table 3 |
| `table_diagnostics.tex` | Table S1 (supplement) |
| `table_irf.tex` | Table S2 (supplement) |
| `summary.csv`, `irf.csv`, `lags.csv`, `resid_diag.csv`, `inst_diag.csv` | all estimates, sets, lag criteria and diagnostics behind the figures and tables |

### 2. Monte Carlo simulations (Tables 1-2, Tables S3-S6)

```
cd simulations
python cns_mc.py all --growth
```

`all` runs the whole pipeline: it calibrates the design to the application (reading the data of step 1), tunes the
common first-stage noise until the median two-way first-stage F statistic of the calibrated design matches the
application, runs the five designs of Section 6, FOD-GMM on the calibrated design, the simulation with several
instruments (AR and CLR), and the comparison with local projections, each with 500 replications, and writes the
tables. Every replication has a fixed seed, so the results do not depend on the number of worker processes
(`--workers`). The run is resumable: if it is interrupted, the same command continues from the last saved chunk, and
files produced under a different calibration are set aside automatically. Each part can also be run separately with
`python cns_mc.py run --growth --mode main|gmm|clr|lp`, followed by `python cns_mc.py table --growth`.
Outputs:

| File | In the paper |
|---|---|
| `tables/table_cns_mc.tex` | Table 1 (calibrated and weak designs) |
| `tables/table_cns_mc_clr.tex` | Table 2 (several instruments, AR and CLR) |
| `tables/table_cns_mc_gmm.tex` | Table S3 (FOD-GMM) |
| `tables/table_cns_mc_county.tex` | Table S4 (clustering by county) |
| `tables/table_cns_mc_persistent.tex` | Table S5 (persistent shift) |
| `tables/table_cns_mc_lp.tex` | Table S6 (comparison with local projections) |
| `results/cns_calibration_growth.json` | calibrated parameters and true responses |
| `results/cns_mc_growth/`, `cns_mc_growth_gmm/`, `cns_mc_growth_clr/`, `cns_mc_growth_lp2/` | one row per replication |
| `results/cns_mc_summary.csv`, `results/cns_mc_lp_summary.csv` | coverage, set length and bias by design and horizon |

Small differences in the lengths of nearly unbounded Anderson-Rubin sets in the weak designs can arise across
machines and library versions from floating-point rounding; they do not affect any coverage rate or table entry.

### 3. Paper

Copy the outputs of steps 1 and 2 into `paper/` (the application files directly, the simulation tables into
`paper/tables/`), then compile `pvar_iv_jbes.tex` and afterwards `pvar_iv_jbes_supplement.tex`, which reads the
cross-references of the paper. In both files, `\anontrue` produces the anonymous version.
