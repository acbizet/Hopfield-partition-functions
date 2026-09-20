# Rank-five comparison at higher numerical precision

This extends the previous comparison of the exact **positive lattice formula** with tensor Gauss-Hermite evaluation of the original integral. It does not benchmark the older oscillatory complex-residue formula.

The model remains N=8, r=5, beta=2, with exactly the same integer weight matrix and biases. The original IEEE binary64 bias values are now treated as exact rational numbers. This avoids changing the model when requesting errors smaller than double precision. Input precision here defines the numbers; the error targets concern evaluation of that fixed mathematical model.

## Methods and fairness

The positive representation is

$$
Z=G_N(\beta)^{-r}\sum_{\mathbf k\in\mathbb Z^r}
e^{-N\beta\|x_k\|^2/2}\prod_i2\cosh[\beta(b_i+W_i\cdot x_k)],
\quad x_{a,k}=(p_a+2k)/N,
$$
$$
G_N(\beta)=\sum_{j\in\mathbb Z}e^{-2\beta j^2/N},\quad
p_a=(\sum_iW_{ia})\bmod2.
$$

The positive evaluator retains k=-K,...,K for even parity and k=-K,...,K-1 for odd parity. This model has four odd-parity axes and one even-parity axis, giving (2K)^4(2K+1) retained points.

For the original integral, set m=sqrt(2/(N beta)) y. The Gaussian weight is exp(-|y|^2), and the tensor Gauss-Hermite rule uses q nodes in every direction, or q^5 points. Its normalization is pi^(-5/2).

Both evaluators use the same native C++ implementation pattern:

1. Read already-prepared input numbers and quadrature rule data before timing.
2. Within each timed call, construct the model-dependent per-axis exponential factors and weights.
3. Traverse the complete rank-five Cartesian product, performing nested positive summation. At every leaf, evaluate the full cosh product through e+1/e, using per-axis exponential products. No spin-state expansion or reference value is present in the timed program.
4. Return log Z, including the method's normalization.

The precomputation reduces repeated transcendental evaluations for **both** methods. It changes the common implementation relative to the previous NumPy benchmark. Therefore absolute seconds in this report should not be compared directly with the earlier Python timings; the within-row ratios are the relevant comparison.

The arithmetic mode is identical for the two methods at each target: binary64, extended precision with a 64-bit significand, or IEEE binary128 with a 113-bit significand. The latter supplies approximately 34 decimal digits and is implemented in software by this toolchain. These are fixed working-precision modes, not arbitrary-precision arithmetic inside the timed kernel. Arbitrary precision is used for the reference and for generating the quadrature nodes.

Each reported time is the median of three full evaluations after one full warm evaluation, on one CPU thread. Quadrature-node generation, compilation, reference generation and cutoff selection are excluded. The raw samples are saved. Both model-dependent setup and tensor traversal are included.

## Accuracy checks

The independent reference sums the 256 spin configurations with arbitrary precision. It is calculated at 100 and 140 decimal digits and the results agree to at least 98 digits. The biases are reconstructed from their exact binary64 numerator/denominator pairs in both calculations.

Gauss-Hermite rules are generated at 85 decimal digits, checked against their zeroth moment, and saved to 80 significant digits. They are then rounded directly into the actual working type. The high-precision rules are not obtained by casting double-precision nodes into a wider type.

For untimed setting selection only, a finite-spin expansion factorizes the mathematical tensor sum into one-dimensional moments. This inexpensive diagnostic computes the truncation error of candidate lattice cutoffs and quadrature orders at 85 digits. It does not supply values to the timed C++ program, which actually visits every tensor point. The same diagnostic and same exact reference are used for both methods.

The returned native log Z is then checked directly against the 140-digit reference. A mathematically adequate setting can still fail due to rounding; such a failure is recorded, and its order is increased. Finer native evaluations check that accuracy persists. For the 1e-24 target, the finer mathematical tensor value is checked at 85 digits; at 1e-30, an additional full native finer-order evaluation checks both methods.

The absolute error target is |log Z_computed - log Z_reference|, not relative error in Z and not the count of printed digits. These are independently validated errors for this model, not general certified error bounds for arbitrary input parameters.

## Results

| Absolute log-Z target | Working precision | Positive formula (s) | Gauss-Hermite (s) | GH / positive time |
|---:|---|---:|---:|---:|
| 1e-9 | binary64 (~16 digits) | 0.011135 | 0.010235 | 0.919 |
| 1e-12 | binary64 (~16 digits) | 0.019338 | 0.027260 | 1.410 |
| 1e-15 | extended (~19 digits) | 0.103304 | 0.185221 | 1.793 |
| 1e-18 | extended (~19 digits) | 0.156143 | 0.379543 | 2.431 |
| 1e-24 | binary128 (~34 digits) | 12.127928 | 30.994965 | 2.556 |
| 1e-30 | binary128 (~34 digits) | 17.088090 | 69.869946 | 4.089 |

A ratio above one means the positive formula was faster. The large jump in absolute time on entering binary128 reflects software quadruple-precision arithmetic as well as the larger sums.

| Target | Lattice K | Lattice points | GH order q | GH points | Actual lattice log-Z error | Actual GH log-Z error |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-9 | 9 | 1,994,544 | 18 | 1,889,568 | 1.297e-10 | 7.291e-10 |
| 1e-12 | 10 | 3,360,000 | 22 | 5,153,632 | 1.691e-13 | 3.130e-13 |
| 1e-15 | 11 | 5,387,888 | 25 | 9,765,625 | 8.974e-17 | 6.544e-16 |
| 1e-18 | 12 | 8,294,400 | 29 | 20,511,149 | 4.636e-19 | 4.038e-19 |
| 1e-24 | 14 | 17,825,024 | 34 | 45,435,424 | 3.660e-29 | 9.157e-25 |
| 1e-30 | 15 | 25,110,000 | 40 | 102,400,000 | 8.809e-34 | 2.998e-31 |

The initial run checked even GH orders. Intermediate odd orders were then tested: q=25 replaces q=26 at 1e-15, and q=29 replaces q=30 at 1e-18. The superseded timings are retained in results.json for transparency. The current preparation and benchmark scripts search consecutive orders directly. At 1e-18, q=28 failed its actual-error check despite having a mathematical truncation error slightly below the requested tolerance; rounding pushed it over the limit.

## Interpretation and limitations

On this fixed example, a speed advantage emerges as the required error decreases. At 1e-30, the positive formula needs 25,110,000 points, while Gauss-Hermite needs 102,400,000: about 4.08 times as many. The measured runtime advantage is 4.09 times. The difference is primarily the number of tensor points, because both paths use the same arithmetic and evaluation structure.

The Gaussian lattice tail decays rapidly with the squared cutoff distance. At this precision, the positive formula retains 30 or 31 points per direction, compared with 40 quadrature nodes per direction. Raising these per-direction counts to the fifth power explains the sizable total difference. This is an explanation of the observed case, not a universal asymptotic complexity result.

The advantage need not increase monotonically at every tolerance: cutoff orders change in integer steps, arithmetic modes change at precision thresholds, and the achieved errors can be substantially smaller than their respective targets.

The result concerns this fixed small model and these two full tensor evaluators. It does not establish that the lattice method wins on every model, at larger N, against adaptive integration or against other optimized solvers. At N=8, direct enumeration of 256 spin states is itself very inexpensive; it is deliberately used as an independent accuracy reference here, rather than presented as an industrial workload.

The positive formula remains a finite approximation to an exact infinite sum. Its cancellation advantage does not eliminate truncation or rounding error. Its Cartesian-product implementation still has exponential dependence on the number of auxiliary coordinates.

## Files and reproduction

- `evaluate.cpp`: the complete timed implementation of both evaluators.
- `prepare.py`: reference calculation, high-precision quadrature rules and untimed setting calibration.
- `benchmark.py`: execution-time measurements and actual-error validation.
- `model.json`: original model parameters.
- `input.txt`: high-precision decimal encodings of exactly the original binary64 inputs.
- `rules/`: high-precision quadrature nodes and weights.
- `calibration.json`: independent reference, candidate errors, settings and node-generation times.
- `results.json`: actual returned values, measured errors, timing samples and finer checks.
- `environment.json`: compiler and platform information.

Install mpmath as specified in requirements.txt, then run:

```bash
g++ -O3 -std=c++17 evaluate.cpp -lquadmath -o evaluate
python prepare.py
python benchmark.py
```

GCC and libquadmath are required for the binary128 mode. The high-precision full tensor runs take substantially longer than the double-precision runs. To reuse the provided high-precision nodes and input preparation, skip `prepare.py`.
