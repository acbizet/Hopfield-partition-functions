# Sequential residue integration: a working implementation

This implementation eliminates one integration variable at a time by residue
sums. It does not use quadrature to evaluate the descendant integrals. The
same calculation is provided both as literal successive callables F0, F1,
..., Fr and as a batched evaluation of those nested residue sums.

## Scope and assumptions

The model is

    Z = c_beta^r integral_R^r I0(m) dm,
    c_beta = sqrt(N*beta/(2*pi)),
    I0(m) = exp[-N*beta*sum_a m_a^2/2]
            product_i 2*cosh[beta*(b_i + sum_a W_ia m_a)].

This version uses integer W, real b, beta>0, the shift Delta=i*pi/beta,
and the lower strip -pi/beta < Im(z) < 0. These choices make all required
pole locations analytical and independent of the remaining coordinates.
The diagonal self interaction is retained. This is a concrete specialization
of the displayed descent recurrence, not a claim to recover every detail of
the user's unseen general algorithm. It is not yet a solver for the earlier
generic real-weight random matrices.

## Step 1: identify the residues for the first variable

Holding all other coordinates fixed (they may already be complex), integer
weights give

    Q0,a(z) = (-1)^P_a exp[-i*pi*N*z + N*pi^2/(2*beta)],
    P_a = sum_i W_ia.

Thus the Bethe roots in the lower strip are

    z_(a,k) = (P_a+2*k)/N - i*pi/(2*beta),  k in Z.

Using P_a modulo 2 merely relabels the same infinite pole set. Their Jacobian
is Q0,a'(z_(a,k))=-i*pi*N. The first contour contribution is therefore

    B0,a[F] = (2/N) sum_k F(z_(a,k), remaining coordinates).

At a canceled pole the integrand residue is zero; evaluating the resulting
formula automatically handles this case.

## Step 2: include the next descent contribution

For J_j=I_j/(Q_j-1), the displayed recurrence can be rewritten

    I_(j+1)(z) = I_j(z-Delta)+J_j(z-Delta)-J_j(z).

With the lower contour running left to right,

    integral I_j = -2*pi*i sum_strip Res(I_j+J_j) + integral I_(j+1).

This formula includes inherited poles of I_j. In the present specialization,
write a=exp[-N*pi^2/(2*beta)], u=a*(-1)^P exp(-i*pi*N*z). Then

    Q1 = u(u-1)/(u-a^4).

Its new Q1=1 roots are outside the strip, but inherited poles contribute
B1,a=-B0,a/2. Consequently the sum of the first two contour contributions is

    T_a[F](remaining) = (c_beta/N) sum_k F(z_(a,k), remaining).

The factor 1/N rather than 2/N is essential. Summing only new roots at the
second descent level would omit the inherited-pole contribution.

## Step 3: form the reduced function and integrate the next variable

Define F0=I0 and then

    F1(m2,...,mr) = T1[F0](m2,...,mr),
    F2(m3,...,mr) = T2[F1](m3,...,mr),
    ...,
    Z_hat = Fr = Tr[...T2[T1[I0]]...].

Each operation eliminates one continuous variable. F_a has the required
quasi-period in every remaining coordinate because every term in its residue
sum has the same remaining-coordinate Q. Finite, coordinate-independent pole
cutoffs preserve that quasi-period too.

The literal implementation is `build_successive_integrals` in solver.py.
It returns callables with one fewer argument at each step. The batched
implementation evaluates exactly the same nested sums, with the first
variable summed for each fixed tuple of the other pole positions. It uses
analytical pole indices and residue prefactors. No Gauss-Hermite, Legendre,
adaptive integration or other real-domain quadrature grid is used.

The roots happen to form lattices in this specialization. A Cartesian set of
known poles should not be confused with a quadrature grid, but enumerating
all pole combinations can still be expensive.

## Step 4: account for the retained-order truncation

Let

    S = sum_(l>=1) exp[-N*pi^2*l*(l+1)/(2*beta)].

For one coordinate, the infinite two-level residue sum equals (1+S) times
the exact normalized integral. This identity remains valid with complex
bias shifts from earlier residues. Repeating it gives

    Z_hat_infinite = (1+S)^r Z,
    log Z_hat_infinite - log Z = r*log(1+S).

The code reports this predicted truncation; it does not divide it away or
silently call the two-level approximation exact. This is two descent
contributions per variable, not an arbitrary-depth implementation.

Finite pole-sum tails and numerical rounding are separate error sources.
A Gaussian envelope determines the cutoffs. Its absolute omitted-residue
bound is compared with the Jensen lower bound

    Z >= 2^N exp[beta*||W||_F^2/(2*N)].

The requested relative residue-tail target is 1e-12. The envelope is
conservative, and its evaluation uses floating point, not interval arithmetic.
Therefore no full numerical certificate is claimed. Long-double arithmetic
has 18 decimal digits of precision on this machine. Cancellation can consume
many of those digits even when the analytical truncation is tiny.

## Measured results

Exact enumeration is an independent reference and is never used by the
residue evaluator. N=4/r=2 uses a coupled integer matrix
[[1,1],[1,-1],[-1,1],[0,1]]. Other integer matrices are sampled with
default_rng seed 7200+N+r; all biases use the same generator and scale 0.15.
The complete arrays are retained in results.json.

| N | Rank | beta | Residue logZ | Absolute logZ error | Predicted two-level truncation | Pole combinations | Runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 1 | 3.269584006447 | <1e-13 | 7.157e-18 | 21 | 0.001031 s |
| 4 | 2 | 1 | 4.012766126319 | <1e-13 | 1.431e-17 | 529 | 0.001396 s |
| 4 | 2 | 4 | 10.910838826443 | 1.034e-04 | 1.034e-04 | 169 | 0.0006192 s |
| 6 | 3 | 2 | 7.608754197742 | 4.157e-13 | 4.151e-13 | 12,167 | 0.01052 s |
| 8 | 5 | 2 | 13.944814081142 | 1.403e-13 | 3.579e-17 | 39,135,393 | 38.62 s |

Times are single CPU observations, including cutoff selection and evaluation,
with one BLAS thread. They exclude reference enumeration, partial-function
checks and refinement. The 39-million-term rank-5 evaluation completed;
straightforward enumeration has not demonstrated a speed advantage here.
This is not evidence that all residue implementations require this many
terms: no pole pruning beyond the envelope cutoff, symbolic compression,
or optimized compiled implementation was used. The small N values make
enumeration inexpensive and are for validation, not commercial benchmarking.

At rank 5, about nine digits are lost to cancellation. Its observed logZ
error is therefore larger than the tiny analytical truncation and tail
estimate. A certificate would have to include numerical summation error.
For ranks 1–3, enlarging every pole cutoff by two left logZ unchanged to
the displayed precision. The full high-rank cutoff was not enlarged again.

## Inspect the actual first integration

For N=4, r=2, beta=1, here are evaluations of F1(m2), after eliminating m1
by residues. The reference independently integrates the first Gaussian
analytically inside the finite spin sum; no quadrature is used.

| Remaining m2 | F1 from residues | Independent partial-integral reference |
|---:|---:|---:|
| -1 | 11.326206073308 | 11.326206073308 |
| -0.5 | 20.065706250099 | 20.065706250099 |
| 0 | 29.147382150338 | 29.147382150338 |
| 0.5 | 33.775769185824 | 33.775769185824 |
| 1 | 26.079377549125 | 26.079377549125 |

Applying the m2 residue operation to F1 gives logZ=4.012766126319322; the independent reference is 4.012766126319322. The literal nested callables agree with the batched evaluation.


## Reproduce

Python 3.12, NumPy 2.3.5, SciPy 1.17.0 were used.

```bash
OPENBLAS_NUM_THREADS=1 python solver.py
python report.py
```

The rank-5 case takes tens of seconds. `solve` supports a pole-combination
budget and returns an explicit budget-exceeded result instead of silently
changing the method. `build_successive_integrals` exposes the literal stages:

```python
stages = build_successive_integrals(W, b, beta, cutoffs)
F1 = stages[1]
value_at_m2 = F1([0.5])          # rank-two example
Z_hat = stages[2]([])           # second variable eliminated
```

Integer weights are validated. Generic learned real weights do not share
these analytical poles; extending the sequential design to them requires
actual root continuation/enumeration and handling singularities introduced
in the reduced functions. That extension has not been substituted by a grid
calculation in this implementation.

