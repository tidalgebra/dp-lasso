# Differentially private Lasso under three privacy placements — experiment code

Code and results for the numerical study in *Differentially Private Lasso under Three
Privacy Placements: ISTA-Based Procedures with Finite-Iteration Guarantees*.

Three thresholded first-order procedures are compared with a published baseline in the
same privacy placement:

| Placement | Our procedure | Baseline |
|---|---|---|
| Label-local | Label-LDP Lasso | Label-LDP-IHT, Wang and Xu (2021) |
| Central | Thresholded Central-DP Lasso | DP-IHT, Cai, Wang and Zhang (2021) |
| Full-record local | Sequential Full-Record LDP Proximal Lasso | Sequential LDP-IHT, Zhu et al. (2024) |

## What produces what

| Figure | Plotting script | Experiment script | Results file |
|---|---|---|---|
| Label-local, simulation | `plot_label_v1format.py` | `exp_label_v1format.py` | `results/label_v1format_confirm.csv` |
| Central, simulation | `plot_central.py` | `exp_central.py` | `results/central_confirm.csv` |
| Full-record, simulation | `plot_full_record.py` | `exp1_full_record.py` | `results/exp1_confirm.csv` |
| Central diagnostics | `plot_central_diagnostic.py` | see note below | `results/figure2a_geometry.csv`, `results/confirm_figure2bc*.csv` |
| Central, ACS records | `plot_central_real.py` | `exp_central_real.py` | `results/central_real_confirm.csv` |
| Full-record, ACS records | `plot_fullrecord_national.py` | `exp_empirical_fullrecord.py` | `results/fullrecord_real_confirm.csv` |

`results/` holds experiment output, not input: each file is written by the experiment
script beside it, and the plotting scripts read it back. `data/` holds the one input the
experiments need, the ACS extract, and is built by `fetch_acs_national.py` rather than
committed.

The three files behind the central diagnostics are an exception. They come from an
earlier round of the same study, whose runner is not included here because it depends on
a larger internal package; the results themselves and the code that plots them are both
provided, so that figure reproduces from this repository like the others. Their settings
are stated in the paper and in the header of `plot_central_diagnostic.py`.

## Reproducing the figures

The results are committed, so every figure redraws without rerunning anything:

```
pip install -r requirements.txt
python plot_label_v1format.py        # label-local, simulation
python plot_central.py               # central, simulation
python plot_full_record.py           # full-record, simulation
python plot_central_diagnostic.py    # central geometry and error decomposition
python plot_central_real.py          # central, ACS records
python plot_fullrecord_national.py   # full-record, ACS records
```

Each writes a `.pdf`, a `.png` and a `_source_data.csv` of the plotted values into
`figures/`. All commands run from the repository root.

## Rerunning the experiments

Each experiment script has a `--mode` flag. `confirm` reproduces the reported numbers on
the confirmation seeds; the other modes are the exploration sweeps used to fix the
constants, on seeds that appear in no reported result.

```
python exp_label_v1format.py --mode confirm
python exp_central.py --mode confirm
python exp1_full_record.py --mode confirm
python exp_central_real.py --mode confirm --multiplier 0.1
python exp_empirical_fullrecord.py --mode confirm --multiplier 0.05
```

The simulations need no external data. The two real-data scripts need the ACS extract
described below. The full-record simulation runs to six million observations and takes
roughly an hour; the others are minutes.

## Data

The real-data experiments use the 2018 American Community Survey one-year public use
microdata sample, published by the United States Census Bureau at

```
https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/
```

The raw files are about 5.7 GB uncompressed and are not redistributed here. Build the
extract with

```
python fetch_acs_national.py
```

which downloads each state's person file, keeps ten columns, applies the income-task
filter of Ding et al. (2021), and writes `data/acs2018_income.parquet`, about 10 MB. It
takes a few minutes and never writes the raw archives to disk. Puerto Rico is excluded,
following the folktables definition of the task, and the pooled national file is skipped
because it would double count the states.

The task predicts whether personal income exceeds 50,000, with responses coded in
`{-1, +1}`. Records are kept when age exceeds 16, income exceeds 100, usual hours worked
is positive, the person weight is at least one, and occupation is recorded.

One caveat on exact reproduction. The reported real-data numbers were produced from a
pool assembled in a particular record order, and a fresh download orders the records
differently. Because each trial draws its sample by index, per-seed values will differ
slightly from those in `results/`. The pooled record set, the conclusions and the
interval widths are unaffected; the plotted values are in `figures/*_source_data.csv`
for direct comparison.

## What is in here

| File | Role |
|---|---|
| `dp_primitives.py` | Thresholding operators, the analytic Gaussian scale of Balle and Wang (2018), the Peeling operator of Cai, Wang and Zhang (2021), and the label-release scale of Wang and Xu (2021) |
| `ldp_vector.py` | The bounded-vector pure local randomizer of Duchi, Jordan and Wainwright (2018), shared by both full-record procedures |
| `figstyle.py` | Figure conventions |
| `exp_*.py` | Experiments |
| `plot_*.py` | Figures |
| `fetch_acs_national.py` | Builds the ACS extract |
| `results/` | Confirmation results, one CSV per experiment plus a JSON recording its grid |

## How the comparisons are set up

Each comparison is matched at the point where a mismatch could otherwise decide the
outcome, and the matching is different for each placement.

Under label-local privacy one privatized response vector is drawn per trial and both
arms consume it, so the comparison isolates the estimator rather than two noise
constants. Under central privacy both arms receive the same clipped statistics, the same
total budget and the same iteration count. Under full-record privacy both arms use the
same disjoint batches and the same bounded-vector randomizer at the same pure epsilon,
and each then applies the clipping radius its own analysis prescribes.

In every panel the baseline is credited with the tuning cell that minimises its own
error, swept over its threshold level and, where applicable, its truncation constant.
Our own constants are fixed on exploration seeds before any confirmation seed is run,
and the two seed sets are disjoint. Shaded bands are pointwise 95% Student-t intervals
over eight confirmation seeds.

## Requirements

Python 3.10 or later with numpy, scipy, pandas, scikit-learn, matplotlib and pyarrow.
See `requirements.txt`.

## License

MIT, see `LICENSE`.
