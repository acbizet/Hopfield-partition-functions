# Model, positive identity and scope

Assume $W\in\mathbb Z^{N\times r}$, $b\in\mathbb R^N$, and $\beta>0$. Retain the diagonal contribution in $\|W^T\sigma\|^2$. Removing it changes $\log Z$ by a known constant and must be done consistently across methods.

Define $h_\sigma=W^T\sigma$ and

$$
A_\sigma=\exp\left[\beta b^T\sigma+\frac{\beta}{2N}\|h_\sigma\|^2\right].
$$

The Hubbard-Stratonovich integrand satisfies

$$
I(x)=\sum_\sigma A_\sigma\exp\left[-\frac{N\beta}{2}\left\|x-\frac{h_\sigma}{N}\right\|^2\right].
$$

Let $p_a=(\sum_iW_{ia})\bmod2$. Since $\sigma_i\equiv1\pmod2$, every $h_{\sigma,a}$ has parity $p_a$. Write $h_\sigma=p+2j_\sigma$ for an integer vector $j_\sigma$. At $x_k=(p+2k)/N$,

$$
\sum_{k\in\mathbb Z^r}\exp\left[-\frac{N\beta}{2}\left\|x_k-\frac{h_\sigma}{N}\right\|^2\right]
=\prod_{a=1}^r\sum_{k_a\in\mathbb Z}e^{-2\beta(k_a-j_{\sigma,a})^2/N}=G_N(\beta)^r.
$$

All terms are nonnegative and the spin sum is finite. Interchanging the sums proves

$$
\sum_k I(x_k)=G_N(\beta)^r\sum_\sigma A_\sigma=G_N(\beta)^r Z.
$$

The spin expansion is a proof device and an independent reference for small models; the lattice evaluator computes the cosh product directly.

## Numerical consequences

Positivity gives summation condition number one for $Z$ in exact arithmetic. It removes cancellation between signed terms, but does not remove finite-grid truncation, floating-point rounding, underflow or overflow. The current Cartesian product has exponential cost in rank.

The normalized lattice mass is a mixture of discrete Gaussians. The overlap centers obey $|h_{\sigma,a}|\le\sum_i|W_{ia}|$. Bounding each coordinate's omitted Gaussian tail and applying a union bound gives

$$
0\le (Z-Z_K)/Z\le\delta,\qquad 0\le\log Z-\log Z_K\le-\log(1-\delta),\quad\delta<1.
$$

See the [positive-lattice derivation](../experiments/positive-lattice/README.md) for the explicit cutoff formulas. Bounds evaluated in ordinary floating point are not full interval-arithmetic certificates.

## Field observables

For $b_i(h)=b_i+h a_i$, let $M=N^{-1}\sum_i a_i\sigma_i$, $u_i=\beta(b_i+h a_i+W_i\cdot x)$, $T=\sum_i a_i\tanh u_i$, and $V=\sum_i a_i^2\sech{u_i}^2$. With positive normalized lattice weights,

$$
\langle M\rangle=\mathbb E[T]/N,\qquad
\chi=\frac\beta N\left(\operatorname{Var}(T)+\mathbb E[V]\right).
$$

The implementation combines variances across batches and avoids subtraction in evaluating $\operatorname{sech}^2$. These formulas concern real-field derivatives. Arbitrary real-weight differentiation is not justified by an identity restricted to integer weights.

## Residues and criticality

The residue and positive sums are representations of the same partition function with different numerical conditioning. The positive sum is not unchanged complex-residue evaluation. The rank-one descendant action $\pi^2/\beta$ remains positive at the homogeneous physical transition $\beta=1$; this diagnostic therefore does not locate that transition.

For three correlated patterns with pairwise correlation $c>0$, the Gram matrix has collective eigenvalue $1+2c$, giving $\beta_c=(1+2c)^{-1}$. For the specific $c=1/3$ sequence, the recorded finite-size analysis gives

$$
\chi_N=0.9142635791469034\sqrt N-0.3444428364160962+O(N^{-1/2}).
$$

This is an asymptotic result for that structured sequence, with no explicit nonasymptotic remainder constant supplied. The critical-behavior README details the quartic limiting law, massive transverse modes and auxiliary-noise correction.

## Related sources

- NIST DLMF, [theta series definitions, section 20.2](https://dlmf.nist.gov/20.2).
- NIST DLMF, [theta transformations, section 20.7](https://dlmf.nist.gov/20.7).
- The [critical-behavior note](../experiments/critical-behavior/README.md#7-what-this-contributes-and-what-remains-open) records related mean-field fluctuation literature. A full novelty review is still outstanding.
