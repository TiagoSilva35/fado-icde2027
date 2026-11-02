# Sweeps

The W&B grid sweeps behind every reported run. All use the 30 reporting seeds and log to the
W&B project `fado-ablations`; `DOCS/paper_icde/scripts/make_tables.py` turns their per-seed
metrics into the paper's tables and reports.

**Datasets and drift.**
- **Adult**: the four abrupt injected-drift scenarios (`sim_*`, `src/drift/dataset_generator.py`).
- **Folktables**: its natural drift (train on 2015, stream 2017 then 2018), 10% subsample per
  seed (about 18.7k training and 18.7k stream samples).

| File | Question | Runs | Methods per run |
|---|---|---|---|
| `sweep_all_arms_{adult,folktables}` | main results (Table III) | 120 / 30 | FADO, Aranyani, ARF, RFR (+ `fado_no_lambda` on Folktables) |
| `sweep_aranyani_stream_lambda_{adult,folktables}` | Aranyani with the weight FADO reaches (leaked λ) | 120 / 30 | Aranyani |
| `sweep_ablation_{adult,folktables}` | which drift component contributes? | 120 / 30 | `fado_dual`, `fado_lambda_only`, `fado_no_reset` |
| `sweep_dual_step_adult` | demographic parity and the target ε (dual step η = 0.03, 0.1) | 120 | FADO |
| `sweep_intersectional_folktables` | four groups (sex × race) | 30 | FADO, Aranyani, ARF |

The ablation arms are presets in `src/models/forest/controller.py`. The ablation, dual-step,
and intersectional sweeps pair by seed with the FADO and Aranyani runs of the main sweeps,
which are deterministic per seed, and set `--pretrain_cache_dir`, so each machine pre-trains a
seed once (`TESTS/check_pretrain_cache.py` checks that a cached forest is bit-identical to a
fresh one). Each run evaluates all its methods on its seed, so its cost is one pre-training
plus one pass of the stream per method.

## Running a sweep

```bash
python -m TESTS.check_sweeps                       # every config must print "ok"
src/configs/sweeps/launch.sh all_arms_adult        # validates and registers; prints the agent command
wandb agent <entity>/fado-ablations/<sweep-id>     # one agent per free CPU core
```

Set `WANDB_ENTITY` to register the sweeps under your own W&B entity. Java 17 or newer must be
on the `PATH`. Each run is single-threaded, so run one agent per free core; several agents on
the same sweep split its grid, and each run writes to its own `files/experiments/wandb/<run id>/`.

On Windows, W&B's agent expands `${env}` in a sweep's `command` to nothing: replace the first
three entries of `command` (`${env}`, `PYTHONHASHSEED=0`, `${interpreter}`) by the full path of
the virtual environment's `python.exe`, and set `PYTHONHASHSEED=0` in the shell before
starting the agent.
