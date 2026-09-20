# Critical fluctuations of three correlated Hopfield patterns

This is a worked analytic calculation and reproducible numerical check, not a claim of a new universality class or established publication priority. It extends the residue calculation from the partition function to a physical observable: the susceptibility of the collective pattern overlap. A conventional critical-point expansion supplies an explicit finite-size approximation, which is checked independently.

Run `OPENBLAS_NUM_THREADS=1 python experiment.py` after installing NumPy, SciPy and mpmath. The script writes `results.json`. It evaluates residues and analytic field derivatives directly; it does not numerically integrate descendants. The independent grouped spin sum is only a reference.

## 1. Model and observable

Use the Hamiltonian, retaining its constant diagonal contribution,

$$
H_h(\sigma)=-\frac{1}{2N}\|W^T\sigma\|^2-hN M(\sigma),\qquad
M=\frac{R_1+R_2+R_3}{\sqrt3},\quad
R_a=\frac1N\sum_i W_{ia}\sigma_i.
$$

There are three bipolar patterns with exactly equal pairwise empirical correlation c, with 0<c<1 fixed as N increases. Their Gram matrix is

$$
C=N^{-1}W^TW=(1-c)I_3+c\mathbf1\mathbf1^T.
$$

One explicit realization has four row types and group sizes

| Row of W | Fraction of spins |
|---|---:|
| (+1,+1,+1) | (1+3c)/4 |
| (-1,+1,+1) | (1-c)/4 |
| (+1,-1,+1) | (1-c)/4 |
| (+1,+1,-1) | (1-c)/4 |

Take a sequence of N for which these group sizes are integers. Where each group size is even, splitting it equally between the listed row and its negative produces individually balanced patterns. The zero-field model and the stated overlap are unchanged by the corresponding spin gauge transformation.

The numerical example is c=1/3, with group sizes N/2,N/6,N/6,N/6 and N a multiple of 12.

Let v=(1,1,1)/sqrt(3) and a_i=W_i dot v. The physical susceptibility is defined by the field in the Hamiltonian:

$$
\chi_N=\left.\partial_h\langle M\rangle_h\right|_{h=0}
=\beta N\operatorname{Var}_0(M).
$$

Global spin reversal gives mean M=0 at every finite N. There is no singularity at finite N; the critical inverse temperature below refers to the large-N limit.

## 2. What follows exactly from the residue formula

Put b_i=h a_i in the integer-weight residue representation. Its poles z_k and correction Psi are independent of h:

$$
Z(h)=\mathcal P\sum_{\mathbf k\in\mathbb Z^3} I_h(\mathbf z_{\mathbf k}),\qquad
\mathcal P=\left(\frac{\sqrt{N\beta/(2\pi)}}{N\Psi}\right)^3,
$$
$$
I_h(z)=e^{-N\beta\sum_a z_a^2/2}\prod_i 2\cosh[\beta(W_i\cdot z+h a_i)].
$$

The Gaussian envelope bounds the differentiated series locally uniformly in h, so derivatives may be taken term by term. Therefore

$$
\boxed{\chi_N=\frac1{\beta N}
\left[\frac{\sum I_0''(z_k)}{\sum I_0(z_k)}-
\left(\frac{\sum I_0'(z_k)}{\sum I_0(z_k)}\right)^2\right].}
$$

Primes here mean derivatives with respect to h, not z. The prefactor, including Psi, cancels. Away from removable zeros one can write

$$
I_h''=\beta^2I_h\left[
\left(\sum_i a_i\tanh u_i\right)^2+
\sum_i a_i^2\operatorname{sech}^2u_i\right],\quad
u_i=\beta(W_i\cdot z+h a_i).
$$

The implementation instead differentiates products polynomially, avoiding divisions at zeros. At the poles every matter factor is proportional to sinh, since all row sums are odd. For a group with multiplicity n and field coefficient a, set s=2sinh(u), q=2cosh(u), and G=s^n. Then

$$
G'=n\beta a\,q s^{n-1},\qquad
G''=n(\beta a)^2s^n+n(n-1)(\beta a)^2q^2s^{n-2}.
$$

All tested group counts are at least two. Repeated product rules produce Z, Z' and Z'' in the same residue summation. Permutation symmetry reduces the number of evaluated pole tuples without replacing the residue sum by a spin-state expansion.

## 3. Critical distribution and susceptibility amplitude

The Hubbard-Stratonovich action at h=0, with its additive constant removed, is

$$
\Phi_\beta(m)=\frac\beta2\|m\|^2-
\frac1N\sum_i\log\cosh(\beta W_i\cdot m).
$$

Its Hessian at zero is beta(I-beta C). The eigenvalue of C along v is lambda=1+2c, while the two transverse eigenvalues are 1-c. Hence

$$
\beta_c=(1+2c)^{-1}.
$$

This is the onset of collective ordering. It is not a demonstrated threshold for reliable retrieval of one particular memory. The inequality log cosh(x)<=x^2/2 proves that zero is the unique global minimum at beta_c: the inequality is strict whenever any W_i dot m is nonzero. It also excludes an earlier first-order transition for this zero-field model. The two transverse modes remain massive for fixed c>0.

The row moments needed below are

$$
\mu_2=\overline{a^2}=1+2c,\qquad
\mu_4=\overline{a^4}=\frac{7+20c}{3},\qquad
\mu_6=\overline{a^6}=\frac{61+182c}{9}.
$$

Write m=N^{-1/4}x v+N^{-1/2}y, with y perpendicular to v. At beta_c, the leading action is

$$
N\Phi_{\beta_c}=A_c x^4+\frac{d_c}{2}|y|^2+o(1),\quad
A_c=\frac{\beta_c^4\mu_4}{12},\quad
d_c=\beta_c[1-\beta_c(1-c)].
$$

The auxiliary variable satisfies exactly m=M_vec+eta, where M_vec=W^T sigma/N and eta is an independent centered Gaussian with covariance I/(N beta), in the joint Hubbard-Stratonovich construction. Its noise disappears at the N^{1/4} critical scale. Thus the physical variable X_N=N^{1/4}M converges weakly to the density

$$
\boxed{p_c(x)=\frac{2A_c^{1/4}}{\Gamma(1/4)}e^{-A_c x^4}.}
$$

This is a limiting density: the finite-N spin overlap itself remains discrete. Its second moment yields

$$
\boxed{\chi_N=C(c)\sqrt N+O(1),\quad
C(c)=\frac{6(1+2c)}{\sqrt{7+20c}}
\frac{\Gamma(3/4)}{\Gamma(1/4)}.}
$$

Positive fixed correlations select one collective critical mode. The case c=0 has three critical modes and is excluded from this one-mode asymptotic expansion. In particular, the expansion is not uniform as c tends to zero.

## 4. First finite-size correction, derived without fitting

At order N^{-1/2}, the effective marginal density of the scaled *auxiliary* critical coordinate has the correction

$$
e^{-A_c x^4}\left[1+N^{-1/2}(D_c x^6-B_c x^2)+O(N^{-1})\right],
$$

understood after normalization and in the integrated-moment expansion, not as a globally positive finite-N density. The constants are

$$
D_c=\frac{\beta_c^6\mu_6}{45},\qquad
B_c=\frac{\beta_c^4(3\mu_2-\mu_4)}{2d_c}
=\frac{1-c}{9c(1+2c)^2}.
$$

The x^6 term comes from the sixth-order expansion of log cosh. The x^2 term comes from integrating the quartic coupling to the two transverse Gaussian modes. The possible x^3 y term vanishes by permutation symmetry: the perpendicular projection of the vector average of a_i^3 W_i is zero.

For a quartic density, define

$$
q_k=A_c^{-k/4}\frac{\Gamma((k+1)/4)}{\Gamma(1/4)}\quad (k\text{ even}).
$$

The exact auxiliary-noise identity gives chi_N=beta_c N Var(m dot v)-1, and consequently

$$
\boxed{\chi_N=\beta_c q_2\sqrt N+
\beta_c\left[D_c(q_8-q_2q_6)-B_c(q_4-q_2^2)\right]-1+O(N^{-1/2}).}
$$

The minus one is essential: it subtracts the auxiliary Gaussian contribution, which is absent from the physical spin susceptibility.

The standard degenerate Laplace argument controls this expansion for fixed c>0: zero is the unique global minimum, the action is coercive, the transverse quadratic form is positive, and the critical quartic coefficient is positive. The region outside a fixed neighborhood has exponentially small weight; inside, Taylor expansion and rescaling give the stated moment expansion. This note does not supply an explicit constant for the O(N^{-1/2}) remainder, so it is not a nonasymptotic numerical error certificate for that approximation.

For c=1/3,

$$
\beta_c=\frac35,\quad A_c=\frac{123}{2500},\quad
B_c=\frac2{25},\quad D_c=\frac{219}{15625},
$$
$$
\boxed{\chi_N=0.9142635791469034\sqrt N-0.3444428364160962+O(N^{-1/2}).}
$$

The coefficients are evaluated from the displayed gamma-function formulas. They were not fitted to numerical data.

## 5. Numerical evidence

The independent reference sums over the four group magnetizations S_g=-n_g,-n_g+2,...,n_g with multiplicity product_g binom(n_g,(S_g+n_g)/2). It includes every spin configuration and has no state-space cutoff. It is evaluated in floating point with stable positive weights. At N=12 it agrees with ordinary enumeration of all 4096 configurations; a separate 100-digit grouped sum checks the residue output.

| N | Full finite-state sum: susceptibility | Analytic formula including constant | Relative error |
|---:|---:|---:|---:|
| 12 | 2.873524158 | 2.822659105 | 1.77013% |
| 24 | 4.166360923 | 4.134515682 | 0.76434% |
| 48 | 6.008626915 | 5.989761046 | 0.31398% |
| 96 | 8.624231210 | 8.613474201 | 0.12473% |
| 192 | 12.329986127 | 12.323964928 | 0.04883% |
| 384 | 17.574759793 | 17.571391238 | 0.01917% |

The following two direct residue evaluations both give chi = 2.873524158356024125814297772083539917198 at N=12 and beta=0.6:

| Pole cutoff K | Working digits | Full pole tuples | Symmetry-reduced evaluations | One-run elapsed time |
|---:|---:|---:|---:|---:|
| 40 | 80 | 531,441 | 91,881 | 3.868 s |
| 42 | 100 | 614,125 | 105,995 | 4.686 s |

These elapsed times are execution records, not a warmed speed benchmark. The grouped positive spin sum is much cheaper on this specially structured small example; no computational advantage is claimed.

At N=192 the Kolmogorov distance between the exact discrete distribution of N^{1/4}M and the limiting quartic CDF is about 0.006233. Both left and right limits of the discrete CDF were checked, so the discreteness was not hidden by a smooth histogram.

The critical window also has an explicit prediction. For beta_N=beta_c+s/sqrt(N),

$$
p_s(x)=\frac{\exp(sx^2/2-A_c x^4)}{\int_{\mathbb R}\exp(st^2/2-A_c t^4)dt},\qquad
\frac{\chi_N}{\sqrt N}\longrightarrow\beta_c\int x^2p_s(x)dx.
$$

For s=-1,0,+1 the limiting susceptibility amplitudes are respectively 0.435867, 0.914264 and 2.533783. The script checks N=48,96,192 on each of these sequences. Convergence away from s=0 is slower; at N=192 the corresponding values are 0.389563, 0.889840 and 2.346301. These are leading limits, not uniformly high-accuracy finite-N formulas throughout the window. The one-dimensional integrals here evaluate the limiting crossover function, not the Bethe descendants.

## 6. Analytic error control for the residue observable

For the example, the pole real parts are x_a=2k_a/N. With |k_a|<=K, put alpha=N beta/2, spacing d=2/N and L=2(K+1)/N>1. Completing the Gaussian envelope square gives

$$
|I_0(z_k)|\leq 2^N e^{3N\pi^2/(8\beta)+3N\beta/2}
\prod_{a=1}^3 e^{-\alpha(|x_a|-1)^2}.
$$

A full one-coordinate sum is bounded by F=2[1+sqrt(pi)/(d sqrt(alpha))]. Each two-sided omitted tail is bounded by

$$
T\leq2\left[e^{-\alpha(L-1)^2}+\frac{\sqrt\pi}{2d\sqrt\alpha}
\operatorname{erfc}(\sqrt\alpha(L-1))\right].
$$

The three-dimensional tail is bounded by 3TF^2 times the displayed prefactor and the residue normalization. Divide by the Jensen lower bound Z>=2^N exp(3 beta/2) to obtain delta_0, a relative-Z tail bound.

Each field derivative of a matter factor is bounded by beta |a_i| times the same elementary exponential envelope. Therefore the second-derivative tail is at most beta^2 (sum_i |a_i|)^2 times the Z envelope. Since sum_i |a_i|=2N/sqrt(3), a susceptibility bound, using the exact zero-field symmetry, is

$$
|\chi_N-\chi_{N,K}|\leq
\left(\frac{4\beta N}{3}+|\chi_{N,K}|\right)\delta_0.
$$

This bound concerns truncation of exact sums. At N=12, beta=0.6, K=40 the evaluated bound is 1.87e-16; K=42 reduces it to 1.04e-22. Ordinary high-precision arithmetic was used, not outward-rounded interval arithmetic. Accordingly these are evaluations of analytic bounds, not full certificates incorporating all floating-point error. The formula loses about 29 digits through cancellation in this example. Evaluations at 80 and 100 digits were checked against a separate 100-digit positive reference and agree to all 40 digits stored for the residue result. The tail of Psi after the 30 terms used here is negligible at these parameters, and Psi cancels identically from susceptibility.

## 7. What this contributes and what remains open

This calculation produces an exact residue formula for a physical response, an analytic finite-size susceptibility prediction with no fitted coefficients, and numerical validation. It demonstrates that the residue representation can reproduce a nontrivial observable near a critical point.

The leading quartic law and square-root susceptibility scaling are established types of mean-field critical behavior. The asymptotic calculation above uses the real Hubbard-Stratonovich action; it has not been extracted more efficiently by the descent hierarchy alone. The residue method supplies an exact finite-N representation and a separate validation route. These roles should not be conflated.

Relevant prior work includes:

1. B. Gentz and M. Lowe, *Fluctuations in the Hopfield Model at the Critical Temperature*, Markov Processes and Related Fields 5 (1999), 423-449. The paper establishes non-Gaussian, pattern-dependent critical fluctuation laws in a random-pattern Hopfield setting. [Journal abstract](https://math-mprf.org/journal/articles/id856/).
2. M. Fedele and P. Contucci, *Scaling Limits for Multispecies Statistical Mechanics Mean-Field Models*, Journal of Statistical Physics (2011), DOI 10.1007/s10955-011-0334-4. It studies Gaussian and higher-order limiting laws for multispecies mean-field models. [Author manuscript](https://arxiv.org/abs/1011.3216).
3. W. Kirsch and G. Toth, *Critical Regime in a Curie-Weiss Model with two Groups and Heterogeneous Coupling*, Journal of Theoretical Probability 33 (2020), 2001-2026. It establishes non-Gaussian N^(3/4)-scaled total-magnetization limits in a two-group setting. [Author manuscript](https://arxiv.org/abs/1807.05020).

These references concern related settings, not a verified prior appearance of the exact coefficient -0.3444428364 for the particular pattern proportions here. A targeted search did not establish the priority of that specialization. We therefore make no novelty claim for the coefficient or its derivation. A stronger original contribution would require a new theorem, a demonstrably new regime, or an advantage that existing mean-field analysis does not provide.
