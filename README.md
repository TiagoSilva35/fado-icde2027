# Adaptive Fairness Constraints for Online Learning Under Concept Drift

Code, experiment configurations, per-seed results, and the scripts that generate every table
and number of the paper *Adaptive Fairness Constraints for Online Learning Under Concept Drift*
(submitted to IEEE ICDE 2027).

**FADO** (Fairness-Aware Drift-Aware Online Learning) is an online learner that keeps a bound
on demographic parity, `DP ≤ ε`, at every point of a drifting stream. It combines three
mechanisms:

* an **adaptive fairness constraint**: the weight of the fairness penalty is the Lagrange
  multiplier of the bound, learned online by projected stochastic dual ascent on an unbiased
  per-sample (Horvitz–Thompson) estimate of each group's disparity, instead of a constant
  fixed in advance;
* **drift detection with forgetting**: a dual-threshold ADWIN detector on the error stream and
  a per-group ADWIN fairness monitor on the disparity signal reset the penalty's running
  statistics when the stream changes;
* a **reaction controller** that raises the learning rate and sharpens a routing temperature
  after a detected drift, and anneals both back during recovery.

FADO's predictive model is the fair soft oblique decision forest of
[Aranyani](https://github.com/brcsomnath/Aranyani) (Chowdhury et al., ICLR 2024).

---

## Repository layout

| Path | Contents |
|---|---|
| `src/main.py` | Entry point: runs the prequential (test-then-train) pipeline for the selected methods and seeds; holds the default hyperparameters (Table II) |
| `src/models/forest/` | The forest model, FADO's evaluator (`evaluator.py`), the adaptive constraint (`lambda_controller.py`), the per-sample disparity signal (`fairness_signal.py`), the detectors, and the controller presets used in the ablation (`controller.py`) |
| `src/models/forest/baseline_evaluator.py` | The fixed-weight Aranyani baseline |
| `src/models/arf/`, `src/models/rfr/` | The ARF and RFR baselines |
| `src/drift/` | Injection of the four drift types into the Adult stream |
| `src/configs/sweeps/` | The sweep configuration behind every reported run, and `launch.sh` |
| `data/adult/` | The Adult dataset (UCI) |
| `files/experiments/wandb_arm_metrics/` | Per-seed metrics of every reported run (one CSV per sweep) |
| `DOCS/paper_icde/paper.pdf` | The paper |
| `DOCS/paper_icde/scripts/` | Scripts that generate the paper's tables and numbers |
| `DOCS/paper_icde/tables/` | The generated tables, as included in the paper |
| `DOCS/paper_icde/supplementary/` | Full significance report, per-scenario detection split, ablation, dual-step, and intersectional reports, and the derivation of every number quoted in the text |
| `DOCS/paper_icde/data/f1_store.csv` | F1 of every run, computed from its per-sample predictions |

---

## Setup

Python 3.11 or 3.12, and Java 17 or newer on the `PATH` (needed by CapyMOA, which injects the
drifts; check with `java -version`).

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Datasets.** Adult is included in `data/adult/`. Folktables (ACS Income, California) is
downloaded by the `folktables` library on first use and cached in `data/acs-folktables/`;
training is the 2015 survey and the stream is 2017 followed by 2018.

---

## Reproducing the tables (offline, a few seconds)

The per-seed metrics of every reported run are included, so the paper's tables and statistical
tests can be regenerated without re-running any experiment:

```bash
python DOCS/paper_icde/scripts/make_tables.py build
python DOCS/paper_icde/scripts/paper_numbers.py
```

On Windows, set `PYTHONUTF8=1` first. `make_tables.py build` writes:

* `tables/results.tex` (Table III): F1, demographic parity, and equalized odds over 30 seeds,
  over the whole stream and after the drift onset, with the outcome of each paired test;
* `tables/overall.tex` (Table IV): FADO's significant wins and losses against each method;
* `tables/detection.tex` (Table V): how often and how fast FADO detects each injected drift;
* `tables/ablation.tex`: the ablation of the drift components;
* `supplementary/significance.md`: every test behind the marks of Table III (paired Wilcoxon
  signed-rank test, Holm-corrected across the methods compared with FADO and, as a stricter
  check, across the whole table), with the Welch t-test alongside and the number of seeds on
  which FADO is better;
* `supplementary/ablation.md`, `dual_step.md`, `intersectional.md`, `detection_split.md`.

`paper_numbers.py` recomputes every number quoted in the text that is not in a table and
writes `supplementary/paper_numbers.md`, one row per claim with its computation and source.
A few claims need the per-sample curves of the run folders, which are too large to include;
those rows say which runs they cover.

---

## Re-running the experiments

All methods run under the same prequential test-then-train protocol with batch size 1. Within a
seed they share the data split, the forest initialization, and the stream order: the NumPy,
Python, and TensorFlow generators are pinned per seed, TensorFlow op determinism is enabled,
and `PYTHONHASHSEED=0`.

**Seeds.** 30 reporting seeds, disjoint from the seeds used during development:

```text
66,77,88,99,101,111,122,133,144,155,166,177,188,199,202,213,224,235,246,257,268,279,291,303,314,325,336,347,358,369
```

**One run** (one seed, one drift, all four methods), without W&B:

```bash
PYTHONHASHSEED=0 python -m src.main --pipeline_dataset=adult \
    --pipeline_model=fado,aranyani,arf,rfr --drift_scenario=sim_x_permutations --seeds=66
```

Use `--pipeline_dataset=folktables --folktables_subsample_fraction=0.1` for Folktables (no
`--drift_scenario`: its drift is the natural shift between survey years).

**All runs.** The reported runs are the W&B grid sweeps in `src/configs/sweeps/`:

| Config | Paper |
|---|---|
| `sweep_all_arms_adult.yaml`, `sweep_all_arms_folktables.yaml` | Table III: FADO, Aranyani, ARF, RFR; the Folktables ablation of the adaptive constraint |
| `sweep_aranyani_stream_lambda_adult.yaml`, `..._folktables.yaml` | Table III: Aranyani with leaked λ |
| `sweep_ablation_adult.yaml`, `sweep_ablation_folktables.yaml` | Ablation of the drift components |
| `sweep_dual_step_adult.yaml` | Demographic parity and the target ε (larger dual step) |
| `sweep_intersectional_folktables.yaml` | Intersectional groups (sex × race) on Folktables |

```bash
src/configs/sweeps/launch.sh all_arms_adult all_arms_folktables   # validates and registers
wandb agent <entity>/<project>/<sweep-id>                          # one agent per free CPU core
```

Each run is single-threaded; run one agent per free core. Per-seed results are written to
`files/experiments/wandb/<run id>/`. To build the tables from your own sweeps, put their ids
in the `SWEEPS` dictionary of `make_tables.py` and run `make_tables.py crawl`, then
`make_tables.py build --refresh`.

---

## License and acknowledgments

MIT License (see `LICENSE`). The forest model is adapted from
[Aranyani](https://github.com/brcsomnath/Aranyani) and the RFR baseline from the code of
Jiang et al. (NeurIPS 2023); both are MIT-licensed, see `THIRD_PARTY_NOTICES.md`.

This work is funded by national funds through FCT – Foundation for Science and Technology, I.P.,
within the scope of the research unit UID/00326 – Centre for Informatics and Systems of the
University of Coimbra, <https://doi.org/10.54499/UID/00326/2025>.
