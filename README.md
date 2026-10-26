# MF + DP numerical reproduction

Numerical experiments for 3D/4D Keller–Segel (KS), 2D/3D KPP, two-step MF, and a cubic reaction model. The source-only workflow generates physical datasets, trains models, evaluates errors and produces figures. It requires no manuscript, rebuttal, precomputed NPZ data or pretrained weights.

## Install

Use Python 3.10 and CUDA for physical simulation, training and inference. The local validation environment uses Windows, Python 3.10.19, PyTorch 2.8.0+cu129 and an RTX 5080. Linux and other hardware have not been validated.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r paper_reproduction/requirements.txt
python -m paper_reproduction check --source-only
```

The wheel command follows the [official PyTorch 2.8 installation instructions](https://pytorch.org/get-started/previous-versions/#v280). The source-only check verifies source syntax and physical-source identities; it does not run numerical experiments.

## Reproduce without downloading data or models

Run from this repository root:

```bash
python -m paper_reproduction reproduce --models ks3d ks4d kpp2d kpp3d cubic --from-scratch --workers 4 --out reproduced/full
```

Use a new or empty output directory. This runs all physical simulations at their full particle counts and can take substantial time and disk space. `--workers` controls simultaneous KS exact-OT solves; reduce it when memory is limited.

Outputs include:

- `data/`: new physical training, evaluation and reference data, with settings, seeds and hashes.
- `training/`: new models for seeds 88, 89 and 90, with dataset identities and completion records.
- `results/`: new KS, KPP, cubic, two-step MF and KPP FK results, plus inference timing.
- `figures/`: 21 figures made from the new results, with an input manifest.
- `reproduction_run.json`: stage states and elapsed time. Failed runs preserve their outputs for inspection.

The `--from-scratch` workflow does not read old binary data or weights. Distributed JSON/CSV files describe original configurations and historical metrics for comparison. New data signatures are checked against the models trained on them. An old model cannot silently substitute for a new model.

The complete source-only run was validated on the environment above in 79.5 minutes, producing about 1.85 GiB of outputs. It generated all physical data, 60 new checkpoints, 708 error/reference records, 480 FK trace rows and 21 figures. All trained model parameters matched the historical models. KPP, cubic and KS4D error values matched; KS3D and two-step MF values changed by at most about 0.00220 because of the explicitly regenerated evaluation samples described below. Full details are in `VALIDATION.json`.

## Run individual stages

```bash
python -m paper_reproduction data --models ks3d ks4d kpp2d kpp3d cubic --out reproduced/data
python -m paper_reproduction train --models ks3d ks4d kpp2d kpp3d cubic --data-run reproduced/data --out reproduced/training
python -m paper_reproduction evaluate --models ks3d ks4d kpp2d kpp3d cubic --model-run reproduced/training --with-auxiliary --workers 4 --out reproduced/evaluation
python -m paper_reproduction timing --model-run reproduced/training --out reproduced/timing
python -m paper_reproduction figures --results reproduced/evaluation --out reproduced/figures
```

A completed training run records its data run. Evaluation and timing infer it when `--data-run` is omitted. Pass `--data-run` explicitly after relocating data; the identity must still match. Keep data, training and results together when moving an experiment.

To repeat training and evaluation using a completed new data run:

```bash
python -m paper_reproduction reproduce --models ks3d ks4d kpp2d kpp3d cubic --data-run reproduced/data --workers 4 --out reproduced/repeat
```

To generate and retain prediction clouds for separate evaluation:

```bash
python -m paper_reproduction generate --models ks3d ks4d kpp2d kpp3d --model-run reproduced/training --out reproduced/generated
python -m paper_reproduction evaluate --models ks3d ks4d kpp2d kpp3d --model-run reproduced/training --generated reproduced/generated --workers 4 --out reproduced/scores
```

Clouds are checked against their checkpoint hashes and physical-data identities. Cubic inference is integrated into evaluation. Additional experiments also have separate commands:

```bash
python -m paper_reproduction multistep --model-run reproduced/training --out reproduced/multistep
python -m paper_reproduction warm-start --models kpp2d kpp3d --model-run reproduced/training --out reproduced/warm
```

## Physical data and random seeds

**KS:** each dimension uses eight 15,000-particle training conditions, twenty 10,000-particle native evaluation conditions and twenty 100,000-particle interacting references. The regularized interaction kernel, split integrator, final time 0.02 and time step 0.005 are retained.

KS3D training uses the recovered Torch seed 88 initial cloud and a continuous NumPy RandomState(88) Brownian stream over eight conditions. The original native evaluation RNG sequence was not fully recoverable. New 10,000-particle evaluation sources use explicit seeds `999 + 1009*i`. These form the first 10,000 points of the 100,000-particle prediction sources; the remaining 90,000 retain seed `2026100901 + sigma`. Physical 100k reference initialization/noise retain seeds `2026100902 + sigma` and `2026100903 + sigma`.

KS4D retains initial seeds 666/999/10999 and condition-specific Brownian offsets `1009*i`. New KS generation uses one CPU thread. The historical KS3D 100k sampler used four, which can produce a few float32 rounding differences even with the same seed. Recipes are recorded; historical array equality is not assumed.

**KPP:** each dimension generates eight 20,000-particle training conditions with seed 666, eleven 20,000-particle evaluation conditions with seed 999, and eleven 100,000-particle references with seed 10999. Original physical batches are preserved. All run for 4096 forcing periods with 256 substeps per period. Original potentials, period-end weighting and resampling are preserved. Qualitative figures and the FK horizontal reference use newly generated 20k evaluation data. Raw physical outputs are retained for provenance.

**Cubic:** nine diffusion values `.25, .375, .5, .75, 1, 1.5, 2, 2.5, 3` each use 5,000,000 particles, final time 20 and time step .5. Initial/noise seeds are 666/667. D<=2 uses the original half-width-60 box and 100³ grid; D=2.5 and 3 use half-width 96 and 160³. Sampling seeds 12345/22345/33345/43345/53345 and original sampling sizes are retained. Only D=.25,.5,1,2 supply training targets.

## Training, metrics and timing

Body models use MF1000 / DP1000 / MF500+DP500. Cubic uses MF2000 / DP2000 / MF1000+DP1000. All use seeds 88/89/90 and recorded architecture/optimizer settings. New-data training validates new checkpoints without loading historical weights.

Accuracy uses all 100,000 particles and full-dimensional exact 16-bin-per-axis histogram OT. KPP costs are periodic; cubic uses coordinate/error normalization scale 5. Mass and optimality certificates are checked. Historical metrics are descriptive comparisons, not an equality requirement imposed on new Monte Carlo samples.

GPU inference timing uses 100,000 particles, chunks of 10,000, ten warm-ups and forty synchronized measurements per model/seed/method. Timing depends on hardware and load.

## Optional retained-data workflows

For an existing local installation containing historical data and weights:

```bash
python install_data.py /path/to/reproduction-data-v1.zip
python -m paper_reproduction check --load-models
python -m paper_reproduction reproduce --models ks3d ks4d kpp2d kpp3d cubic --strict-replay --workers 4 --out reproduced/retained-data-replay
python -m paper_reproduction figures --out reproduced/frozen-figures
```

These commands require saved inputs. The 224-file NPZ archive does not contain the 60 historical checkpoints. `--strict-replay` compares historical weights and is incompatible with newly generated data or `--from-scratch`. Source-only reproduction does not need it.

The separate `references` command still generates KPP 100k references for a retained-data installation. Source-only reproduction should use `data` or `reproduce --from-scratch`, which also prepare training and qualitative-reference data.

## Publishing

Upload this repository's code and small JSON/CSV configuration files. No data archive, manuscript, rebuttal or pretrained model is required for the source-only workflow. `.gitignore` excludes NPZ/NPY data, PT/PTH weights and `reproduced/` outputs. Retain small files under `output/`: commands import numerical kernels and configurations from those historical directory names.

Choose a project license before public distribution. The reproduction commands do not upload anything to GitHub.

`VALIDATION.json` records the actual executions, observed agreement and platform limitations. It does not imply bitwise equality on other hardware.
