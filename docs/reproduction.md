# Reproduction guide

Run commands from the repository root unless a working directory is explicitly changed. Python 3.12 is the reference packaging interpreter. Historical measurements and compiler details are preserved under `results/recorded/`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Native binary128 benchmarks target Linux with GCC and libquadmath; the provided C++ program is a fixed N=8, r=5 benchmark, not a general input-size solver. Do not use `-ffast-math` when reproducing accuracy comparisons. The historical extended mode used a 64-bit significand; other platforms can differ.

## Quick checks

```bash
OPENBLAS_NUM_THREADS=1 python examples/quick_check.py
OPENBLAS_NUM_THREADS=1 python experiments/positive-lattice/verify.py
```

The first prints JSON and writes no result file. The second runs two independent small-model checks of normalization, derivatives and tails, then updates `experiments/positive-lattice/verification.json`. The original record remains in `results/recorded/positive-lattice/verification.json`.

## Native smoke check

```bash
g++ -O3 -std=c++17 experiments/high-precision/evaluate.cpp -lquadmath -o experiments/high-precision/evaluate
python scripts/check_native.py
```

This checks both native methods at the recorded 1e-9 settings against a freshly recomputed 100-digit enumeration reference. It also reports enumeration runtime as a small-instance baseline. It does not remeasure the full historical table.

## Full experiments

```bash
OPENBLAS_NUM_THREADS=1 python experiments/positive-lattice/positive.py
OPENBLAS_NUM_THREADS=1 python experiments/critical-behavior/experiment.py
python experiments/high-precision/benchmark.py
```

The last command requires the compiled native executable. Included `input.txt`, `calibration.json` and `rules/` allow the benchmark to run without repeating node preparation. To regenerate the reference, candidate calibration and quadrature nodes first:

```bash
python experiments/high-precision/prepare.py
python experiments/high-precision/benchmark.py
```

Full high-precision tensor runs can take several minutes or longer. Reruns update the experiment's own `results.json`; archived originals under `results/recorded` are never overwritten by these commands. Node generation and setting selection are excluded from the historical timings. A new run may select different settings if native rounding differs.

## Supporting historical experiments

```bash
OPENBLAS_NUM_THREADS=1 python experiments/sequential-residues/solver.py
OPENBLAS_NUM_THREADS=1 python experiments/rank-one-descent/run.py
```

Read each experiment's report before interpreting its output. These are earlier, narrower reconstructions; their approximations and timing conventions differ. `legacy_residue.py` in the positive-lattice folder is an imported comparator, not a standalone reproduction entry point.

## Figures and report

```bash
python -m pip install -r requirements-report.txt
python scripts/make_figures.py
python scripts/build_brief.py
```

These consume archived measurements and regenerate `figures/` and `docs/scientific-brief.pdf`. Historical benchmark numbers in the README remain explicitly identified as recorded measurements.
