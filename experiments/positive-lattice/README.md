# Removing cancellation from the integer-weight Hopfield formula

The same integer-weight Hopfield partition function admits an exact positive lattice representation. This removes the sign cancellation of the previous complex-residue series. It is an equivalent Gaussian-summation representation, not direct evaluation of the original complex residues, and it does not establish a new Bethe-specific identity or a general cure for exponential dependence on rank.

## Exact identity

For N spins, rank parameter r, real b, beta>0, and integer W, retain the same diagonal-energy convention as the previous calculation:

$$
Z=\sum_{\sigma\in\{-1,1\}^N}
\exp\left[\beta b\cdot\sigma+\frac{\beta}{2N}\|W^T\sigma\|^2\right].
$$

Define

$$
I(x)=e^{-N\beta\|x\|^2/2}\prod_i 2\cosh[\beta(b_i+W_i\cdot x)],\qquad
p_a=\left(\sum_i W_{ia}\right)\bmod 2,
$$
$$
x_{a,k}=(p_a+2k)/N,\qquad
G_N(\beta)=\sum_{j\in\mathbb Z}e^{-2\beta j^2/N}.
$$

Then

$$
\boxed{Z=G_N(\beta)^{-r}\sum_{\mathbf k\in\mathbb Z^r}I(x_{1,k_1},\ldots,x_{r,k_r}).}
$$

Every summand and the normalizer are positive. The summation condition number for Z, sum of absolute terms divided by absolute sum, is therefore exactly one in exact arithmetic. This removes sign cancellation; overflow, underflow, function-evaluation error and ordinary summation rounding still need numerical care.

This is an exact infinite lattice identity. The actual implementation retains finitely many lattice points. The lattice is on the real integration domain, so this can also be viewed as a specially normalized lattice quadrature with an exact infinite-sum property. We do not label it as unchanged residue evaluation or claim that it avoids all grid-like computation.

## Short proof

Expand the cosh product only for the proof. For each spin configuration, h_sigma=W^T sigma is integer and h_sigma,a has parity p_a. Completing the square gives

$$
I(x)=\sum_\sigma
e^{\beta b\cdot\sigma+\beta\|h_\sigma\|^2/(2N)}
e^{-N\beta\|x-h_\sigma/N\|^2/2}.
$$

On the indicated lattice, x_a-h_sigma,a/N = 2(k_a-j_sigma,a)/N for an integer j_sigma,a. Thus the lattice sum of the last Gaussian is G_N(beta)^r for every sigma, independent of the Gaussian center. Summing the positive terms proves the identity. There is no enumeration of spin configurations in the positive evaluator.

This also exposes the link with classical theta identities:

$$
G_N(\beta)=\sqrt{\frac{\pi N}{2\beta}}\,
\vartheta_3\left(0,e^{-N\pi^2/(2\beta)}\right).
$$

The standard theta series and transformation formulas are documented in [DLMF 20.2](https://dlmf.nist.gov/20.2) and [DLMF 20.7](https://dlmf.nist.gov/20.7). The code chooses whichever of the direct Gaussian series and transformed series converges faster. Both use positive terms and an explicit bound on the remaining geometric tail.

The earlier residue representation and this formula evaluate the same Z. The relationship does not allow simply deleting the imaginary shift from the old formula while retaining its old normalization: G, or the corresponding theta_3 factor, is essential.

## Stable derivatives

For a field b_i(h)=b_i+h a_i, define normalized positive lattice weights w_k=I_h(x_k)/sum I_h(x_k), and

$$
T_k=\sum_i a_i\tanh u_{i,k},\qquad
V_k=\sum_i a_i^2\operatorname{sech}^2u_{i,k},\quad
u_{i,k}=\beta(b_i+h a_i+W_i\cdot x_k).
$$

For the observable M=(sum_i a_i sigma_i)/N,

$$
\langle M\rangle=\frac{\mathbb E_w T}{N},\qquad
\boxed{\chi=\frac\beta N[\operatorname{Var}_w(T)+\mathbb E_w V].}
$$

This is a sum of nonnegative variance contributions. The code combines batches using the law of total variance, instead of subtracting two large raw second moments. It evaluates sech^2(u) as 4 exp(-2|u|)/(1+exp(-2|u|))^2 to avoid the cancellation in 1-tanh^2(u). These are derivatives with respect to real fields; the identity does not justify differentiating a fixed integer-weight lattice formula with respect to arbitrary real W.

## Selecting coordinate cutoffs without an exact reference

The completed-square proof shows that the normalized lattice distribution is a mixture of product discrete Gaussians. Its centers are h_sigma,a/N, with

$$
|h_{\sigma,a}|\leq A_a,\qquad A_a=\sum_i|W_{ia}|.
$$

Choose a symmetric interval per coordinate: k=-K_a,...,K_a for even parity, and k=-K_a,...,K_a-1 for odd parity. For a symmetric unimodal discrete Gaussian, the largest omitted mass over the possible centers occurs at an endpoint center, h_a=+A_a or -A_a. This follows by sliding the fixed-length interval across a symmetric decreasing sequence.

Let alpha=2 beta/N, j_a=(A_a-p_a)/2, m_right=K_a+1-p_a-j_a and m_left=K_a+1+j_a. When both m values are positive, each Gaussian tail obeys

$$
\sum_{j=m}^{\infty}e^{-\alpha j^2}
\leq \frac{e^{-\alpha m^2}}{1-e^{-\alpha(2m+1)}}.
$$

Divide the sum of the two tails by G_N(beta) to obtain B_a. A union bound, valid for every mixture weight and therefore independent of b, gives

$$
0\leq\frac{Z-Z_K}{Z}\leq\delta=\sum_a B_a,
\qquad
0\leq\log Z-\log Z_K\leq-\log(1-\delta)
$$

when delta<1. Allocate delta_target=1-exp(-epsilon) across coordinates and increase each cutoff until its bound fits. The implementation allocates an equal error budget per axis but allows different cutoffs.

These are analytic truncation bounds. The code evaluates them in float64 and controls the tiny normalizer-series tail separately; it is not an outward-rounded interval-arithmetic certificate covering every numerical operation.

The bounds were independently checked on small integer-weight models with entries beyond +/-1, both parity types, nonzero fields, and inverse temperatures 0.7 and 12. The derivative checks compare against complete positive spin sums. These cases also exercise both branches of the normalizer implementation.

## Fresh timing comparison

The rank-five model is exactly the previous N=8, r=5, beta=2 model, with its unrounded arrays in rank5_input.json. All methods compute log Z. Each table entry is the median of five warm timing batches in the same process, with one BLAS thread. Model-independent Gauss-Hermite nodes and weights are cached; model evaluations and sum construction remain inside the timings. Derivation and tolerance-setting selection are excluded.

The positive and Gauss-Hermite implementations share batching and log-cosh evaluation structure. The previous complex-residue implementation is preserved as legacy_residue.py. These are implementation-level comparisons on a small test case, not a general complexity or industrial-performance claim. Exact spin enumeration is itself very inexpensive at N=8.

| Absolute log-Z target | Original complex residues | Positive formula | Gauss-Hermite integral |
|---:|---:|---:|---:|
| 1e-03 | 123.1 ms | 54.6 ms | 24.1 ms |
| 1e-06 | 232.5 ms | 227.1 ms | 123.4 ms |
| 1e-09 | 2220.5 ms | 394.1 ms | 389.8 ms |

All positive-formula and Gauss-Hermite entries use float64. The original residue evaluator requires longdouble for the 1e-9 target. The positive formula is about 5.6 times faster than the original residue evaluator at that target and about equal in speed to this Gauss-Hermite implementation. At looser targets Gauss-Hermite remains faster.

For this timing table, settings are calibrated against the independent exact reference and checked at a finer cutoff, as in the earlier benchmark. The analytic-bound-selected settings below are a separate evaluation and generally more conservative.

| Requested log-Z error | Cutoffs by coordinate | Positive terms | Analytic log-Z truncation bound | Observed log-Z error |
|---:|---|---:|---:|---:|
| 1e-03 | [7, 6, 7, 6, 6] | 366,912 | 6.769e-04 | 3.595e-04 |
| 1e-06 | [9, 8, 9, 8, 8] | 1,410,048 | 3.043e-08 | 1.614e-08 |
| 1e-09 | [10, 9, 10, 9, 9] | 2,462,400 | 4.570e-11 | 2.425e-11 |

The critical susceptibility test uses the previous N=12, rank-three model at beta=0.6 and pairwise correlation 1/3. The positive formula gives chi=2.873524158356025 in float64, within about 9e-16 of the high-precision reference rounded to double. At K=24 it retains 117649 positive terms. The preceding complex-residue calculation lost about 29 digits and used 80- and 100-digit arithmetic. This demonstrates removal of that cancellation, not a formal guarantee of machine accuracy for every model.

## Reducing the number of terms: what is solved and what is not

1. **Different cutoffs per coordinate.** The implemented bounds avoid imposing the largest cutoff on every direction. The bounded choices in the table are selected without looking at the exact reference. They do not necessarily use fewer points than a reference-calibrated, unbounded cutoff.
2. **Bound and discard regions.** Positivity makes every omitted region an omitted positive mass. The mixture interpretation supplies a route to bounds on such regions. A general adaptive region-pruning solver has not been implemented here.
3. **Find a separable approximation.** If the retained tensor admits I(k_1,...,k_r) approximately equal to sum_{j=1}^R product_a f_{j,a}(k_a), its sum contracts as sum_j product_a sum_{k_a} f_{j,a}(k_a). Evaluation then costs order R r L for L points per coordinate, after the representation has been constructed. The total approximation error is at most the sum of absolute tensor errors. Finding a small R and bounding that error are substantive additional tasks; neither follows automatically from the residue formula or from positivity. Tensor-train forms offer a related direction, with their own approximation ranks and construction costs.
4. **Exploit exact structure when present.** Symmetry, repeated pattern rows and genuinely decoupled factors can reduce the work. These are model-dependent reductions. Expanding the complete cosh product into 2^N spin terms merely moves the expensive computation elsewhere and is not a general solution.

The straightforward positive evaluator still visits a Cartesian product, so rank scaling remains exponential in this implementation. Higher precision, GPU execution and batching can change constants or stability; they do not establish a low-rank tensor representation or remove that exponential scaling.

## Reproduction

Install the versions in requirements.txt, then run:

```bash
OPENBLAS_NUM_THREADS=1 python positive.py
OPENBLAS_NUM_THREADS=1 python verify.py
```

The first command recomputes the scan, comparisons, critical observable and bound-selected evaluations. The second performs the independent small-model checks. Numerical results are in results.json and verification.json. The raw complex-residue comparator is imported by the benchmark; it need not be run as a standalone program.
