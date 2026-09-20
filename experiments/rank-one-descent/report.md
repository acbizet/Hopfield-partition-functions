# Bethe/descent comparison: measured results and scope

The reconstructed rank-one residue method works numerically and has a
predictable truncation error. A vectorized float64 implementation is fast on
the smallest examples, but loses accuracy through cancellation as N grows.
Higher precision recovers accuracy at substantial cost. This run does not
establish a general speed advantage or a commercially scalable solver.

## What was actually implemented

No complete user solver, multidimensional prescription, or other normalization
model definition was available in the supplied conversation or workspace.
Earlier-context retrieval was disabled. This report therefore benchmarks a
specific reconstruction of the visible recurrence, not the user's full
unseen algorithm. It does not replace it with belief propagation/Bethe free
energy, which would be a different method.

The implemented specialization has rank r=1, integer W_i, arbitrary real b_i,
positive beta, shift Delta=i*pi/beta, and the lower horizontal strip
-pi/beta < Im(z) < 0. The real contour runs left to right. Vertical sides are
taken to infinity along sequences avoiding poles. All crossed poles, including
inherited ones, contribute. All calculations retain the diagonal self energy.

For Q_j(z)=I_j(z+Delta)/I_j(z), let F_j=I_j/(Q_j-1). Then

    I_(j+1)(z) = I_j(z-Delta) + F_j(z-Delta) - F_j(z).

Consequently, with C=sqrt(N*beta/(2*pi)), the contour contribution is

    B_j = -2*pi*i*C * sum_strip Res[I_j + F_j].

For the entire initial integrand, only F_0 contributes. Define

    a = exp[-N*pi^2/(2*beta)],
    s = (-1)^sum(W_i) * exp[-i*pi*N*z],
    u = a*s.

Then Q_0=u/a^2 and Q_1=u(u-1)/(u-a^4). The first roots are

    z_k = (sum(W_i)+2*k)/N - i*pi/(2*beta),  k integer,
    B_0 = (2*C/N) * sum_k I_0(z_k).

The new Q_1=1 roots lie outside this strip. However, the inherited poles
inside it give B_1=-B_0/2. Ignoring inherited poles would get the next step
wrong. The evaluated two-level approximation is therefore

    Z_hat = B_0+B_1 = (C/N) * sum_k I_0(z_k).

This code evaluates that residue sum independently of the reference Z. It
does not define Bethe estimates by subtracting numerically integrated
remainders from an already known partition function. Analytically known root
locations avoid root-finding costs; generic weights would not inherit that
advantage automatically.

## A testable, exact error identity for this specialization

Gaussian expansion of the spin sum gives <s^l>=a^(l^2) for integer weights,
including arbitrary real biases. If S=sum_{l>=1} a^[l(l+1)], then

    E_0/Z = -(1+2*S),
    E_1/Z = -S,
    Z_hat/Z = 1+S,
    |log Z_hat - log Z| = log(1+S),
    d = |E_1/E_0| = S/(1+2*S).

Thus d is not literally the relative Z error or the logZ error. Here there
is an exact conversion: S=d/(1-2*d), and the logZ error is
log[(1-d)/(1-2*d)]. This establishes a useful mathematical relation in the
restricted family; it is not an error bound for generic parameters or a
certificate for finite-precision code. Roundoff and finite residue-sum tails
must still be controlled. Direct numerical integrals of I_0, I_1, I_2 at
N=4 and beta=4,8,16 checked the signs and this identity independently.

## Rank-one accuracy and timing

CPU wall times below are milliseconds, median of three calls in one process.
Residue high precision uses Python Decimal with 75 digits; float64 uses
vectorized NumPy. Gauss-Hermite uses 64 nodes and float64, and includes bias
and W gradients. The residue timings compute logZ only. These are research
implementations, not optimized libraries or equal-work/precision benchmarks.
The reported timings do not include validation runs. Uniform-model reference
values use a grouped N+1-term binomial sum; the heterogeneous reference uses
all 256 spin configurations. That grouped sum is an additional strong baseline
for the symmetric special case and its timings are retained in results.json.

| Model | N | beta | Residue logZ error, 75 digits | Residue float64 logZ error | Residue float64 ms | Residue 75-digit ms | GH64 ms | GH64 logZ error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform_rank1 | 4 | 0.8 | below 1e-13 | below 1e-13 | 0.055 | 7.366 | 0.434 | below 1e-13 |
| uniform_rank1 | 4 | 1 | below 1e-13 | below 1e-13 | 0.040 | 7.278 | 0.411 | below 1e-13 |
| uniform_rank1 | 4 | 1.2 | below 1e-13 | below 1e-13 | 0.035 | 6.723 | 0.292 | below 1e-13 |
| uniform_rank1 | 4 | 4 | 5.172e-05 | 5.172e-05 | 0.036 | 4.501 | 0.319 | below 1e-13 |
| uniform_rank1 | 4 | 8 | 7.167e-03 | 7.167e-03 | 0.034 | 4.003 | 0.302 | below 1e-13 |
| uniform_rank1 | 4 | 16 | 8.196e-02 | 8.196e-02 | 0.033 | 3.539 | 0.321 | below 1e-13 |
| uniform_rank1 | 16 | 1 | below 1e-13 | 1.779e-10 | 0.049 | 49.074 | 0.302 | below 1e-13 |
| uniform_rank1 | 64 | 1 | below 1e-13 | failed (nonpositive Z) | 0.181 | 526.840 | 0.472 | below 1e-13 |
| integer_heterogeneous_rank1 | 8 | 3 | 3.713e-12 | 3.723e-12 | 0.040 | 13.409 | 0.355 | below 1e-13 |

Errors below the reference's float64 resolution are not measured as zero:
the table labels them below 1e-13. For example, at N=64, beta=1, the analytic
truncation error is about 4.74e-275; this experiment does NOT resolve that
error numerically. The absolute-sum/signed-sum condition estimate indicates
about 25 decimal digits lost to cancellation there, consistent with the
nonpositive float64 result. Increasing precision from 75 to 100 digits and
enlarging the residue cutoff left every reported float64 logZ unchanged.
That is a stability check, not a rigorous global error certificate.

The heterogeneous model uses W=(1,2,-1,0,1,-2,2,1),
b=(0.1,-0.2,0.05,0.15,-0.1,0.3,-0.05,0.2), beta=3. Centered differences of
the residue logZ with step 1e-5 agreed with exact bias gradients to maximum
absolute error 3.11e-10. Unrestricted W gradients are NOT implemented: varying
integer weights to arbitrary real values invalidates the common quasi-period.
This remains a gap for learning general real-valued low-rank models.

## Generic models: baselines established, Bethe result unavailable

These models have N=12, beta=1, and r=1,2,5. W entries are independent
Normal(0,0.65^2), and biases Normal(0,0.15^2), using NumPy default_rng seeds
4100+r. Exact arrays are stored in results.json. These are three fixed synthetic
instances, not a broad performance study. All 4096 states give the reference
logZ, bias gradients and W gradients.

Gauss-Hermite is tensor-product integration of the Hubbard-Stratonovich
representation. Two orders check refinement. AIS uses 512 particles, 256
linear inverse-temperature bridges, a uniform spin base, and one complete
single-site Gibbs sweep per bridge. Weight updates precede transitions. The
self-interaction is included in scores and excluded from each conditional
field. Five seeds (9000 through 9004) give RMSE of logZ and median runtime per
run. AIS gradient estimates use weighted final particles; they are finite-sample
self-normalized estimates. A high weight ESS does not certify gradient accuracy.
No separate plain-MCMC normalization estimate is claimed: this comparison uses
MCMC transitions inside AIS.

| Rank | Method | Absolute logZ error / AIS RMSE | Gradient max error / AIS mean max error | Runtime ms |
|---:|---|---:|---:|---:|
| 1 | Gauss-Hermite order 32 | below 1e-13 | below 1e-13 | 0.218 |
| 1 | Gauss-Hermite order 64 | below 1e-13 | below 1e-13 | 0.264 |
| 1 | AIS 512 particles x 256 bridges, 5 seeds | 8.900e-04 | 9.451e-02 | 62.472 |
| 2 | Gauss-Hermite order 20 | below 1e-13 | below 1e-13 | 0.386 |
| 2 | Gauss-Hermite order 40 | below 1e-13 | below 1e-13 | 0.657 |
| 2 | AIS 512 particles x 256 bridges, 5 seeds | 8.936e-04 | 7.674e-02 | 96.474 |
| 5 | Gauss-Hermite order 8 | 7.478e-05 | 1.369e-04 | 11.087 |
| 5 | Gauss-Hermite order 12 | 2.748e-08 | 7.820e-08 | 91.099 |
| 5 | AIS 512 particles x 256 bridges, 5 seeds | 5.709e-03 | 8.546e-02 | 97.332 |

At N=12, exact enumeration itself takes less than a millisecond in this run.
These cases are correctness anchors; they are not expensive industrial
workloads. At rank 5, order 12 already needs 248,832 integration nodes,
illustrating the tensor-grid growth. No conclusion about Bethe scalability to
rank 5, 20, or 50 follows without its multidimensional implementation.

## What the comparison supports

1. The specified rank-one construction gives correct residue approximations
   and a quantitatively verified diagnostic-to-truncation-error relation.
2. Small float64 residue sums can be faster than this quadrature implementation,
   but timings are microbenchmarks with different gradient/precision work.
3. Tiny formal descent error does not ensure accurate floating-point evaluation:
   cancellation caused the N=64 float64 implementation to fail.
4. Standard quadrature is a strong baseline here. At larger beta the fixed
   two-level residue truncation is markedly less accurate than quadrature.
5. A complete comparison for generic learned weights, higher ranks and other
   model families still requires the user's solver or full prescriptions for
   admissible shifts, contours, all pole contributions, multidimensional root
   enumeration, stopping, and parameter derivatives. Those methods were not
   invented or substituted in this run.

## Reproduce

Dependencies in this run: Python 3.12, NumPy 2.3.5, SciPy 1.17.0.

```bash
OPENBLAS_NUM_THREADS=1 python run.py > run-output.json
python make_report.py
```

run.py computes all results. make_report.py checks the residue error identity,
direct descendant integrals, independent quadrature agreement, stability checks
and bias gradients, then writes this report. There are no external data
downloads. Runtime results depend on the machine and process conditions.

## Primary references for baseline methods

- SciPy Gauss-Hermite nodes and Gaussian weight convention:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermitenorm.html
- SciPy adaptive integration / QUADPACK:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
- Radford M. Neal, Annealed Importance Sampling:
  https://arxiv.org/abs/physics/9803008

The contour formulas above are derived in this work from the recurrence in the
supplied conversation. They are not attributed to these baseline references.
