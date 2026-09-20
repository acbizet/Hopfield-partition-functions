# Packaging validation - 20 September 2026

## Executed successfully

1. `examples/quick_check.py`: a five-spin integer-weight case agrees with complete enumeration for log Z, mean overlap and susceptibility; the observed normalization loss respects the evaluated tail bound up to the stated rounding allowance.
2. `experiments/positive-lattice/verify.py`: both independent cases passed, at beta = 0.7 and 12. The checks cover mixed parity, weights beyond +/-1, fields, derivatives and Gaussian-tail branches.
3. The native C++ evaluator compiled with `g++ -O3 -std=c++17 ... -lquadmath`.
4. `scripts/check_native.py`: a fresh 100-digit enumeration agrees with the saved reference to the required 95-digit threshold. Native lattice K = 9 and Gauss-Hermite q = 18 have log-Z errors of approximately 1.297e-10 and 7.291e-10, both below 1e-9.
5. `experiments/critical-behavior/experiment.py`: the full script completed, reproducing grouped sums through N = 384, critical-window comparisons, brute-force checks and residues at 80 and 100 digits. Both residue susceptibilities agreed with the independent high-precision reference within the script's 1e-35 tolerance.
6. Figure and PDF generators executed successfully. The four-page PDF was rendered and visually inspected.
7. All Python sources parsed; document links were checked; original result snapshots were preserved.

## New records

- `results/quick-check.json`
- `results/native-check.json`
- `results/critical-check.json`
- `results/packaging-environment.json`

The new native smoke check measured approximately 0.0115 seconds for the positive method and 0.0102 seconds for Gauss-Hermite at the 1e-9 settings. These are local checks, separate from the historical precision sweep.

The 100-digit Python enumeration baseline took approximately 0.0059 seconds on this small model. Because its language and precision differ from the binary64 native kernels, this is not a matched-implementation speed ranking. It illustrates why enumeration is a relevant baseline at N = 8.

## Not revalidated in this packaging pass

The complete six-target high-precision timing sweep, quadrature-node regeneration and earlier rank-one/sequential experiments were not rerun. Their historical measurements remain available with their original methodology. No clean pip installation or GitHub-hosted Actions run was performed.

The local environment provided NumPy 2.3.5 and SciPy 1.17.0. mpmath 1.4.1 was loaded from the existing session dependency source with a fresh bytecode cache after its old cache proved unreadable. Published scripts use ordinary package imports and pinned requirements; they do not depend on that temporary folder.

These checks validate the documented examples; they are not a proof of correctness for arbitrary inputs or a floating-point error certificate.
