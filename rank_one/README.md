# Rank-one operator experiments

Code and recorded results for the noncommuting rank-one experiment and the
Rademacher RKHS truncation experiment in Tables 1 and 2 of the NeurIPS manuscript.
The full reference run in `results/` uses seed **20260919**; its computed means
and Monte Carlo standard errors are the values used in the updated paper.

The numerical implementation, runner, tests, and recorded CSV/NPZ outputs were
imported unchanged from the supplied reproduction package. The copied manifest's
output directory is written as `results` for portability; its recorded source
and output hashes remain valid. The runner retains the package's provenance
metadata and historical rebuttal comparisons. `rebuttal_reference.csv` and
`results/comparison_to_rebuttal.csv` contain the earlier rounded values;
`results/summary.csv` is the reference for the current manuscript.

## Run

Use Python 3.11 or newer (the reference run used Python 3.12.14). No datasets
are needed. Install the dependencies as described in the [main README](../README.md),
then run from this directory:

```bash
cd rank_one  # From the repository root.
python -m unittest discover -s tests -v
python run_experiments.py --jobs 4 --output outputs/full
```

The last command runs **all paper trials**, with failure probability `delta=0.05`
(95% confidence):

| Experiment | Ambient dimension | Sample size | Ranks | Trials |
|---|---:|---:|---|---:|
| Noncommuting rank-one updates | 1000 | 20000 | 3, 5, 10, 20 | 40 per rank |
| Rademacher RKHS truncations | rank m | 10000 | 32, 64, 128, 256 | 100 shared trials |

The supplied `results/` directory contains the full reference run. Use a new
output directory for a rerun. Existing runs are protected unless `--overwrite`
is explicitly supplied. The number of workers changes execution order, not
the samples or CSV row ordering. Numerical libraries use one thread per worker.
Memory is O(nm+m²) per worker; the largest m is 256, not 1000. Full runs can take
tens of minutes depending on CPU and numerical libraries, primarily due to MEB2.

Other commands:

```bash
# Small end-to-end check, not the full paper experiment:
python run_experiments.py --smoke --jobs 2 --output outputs/smoke

# Either experiment on its own:
python run_experiments.py --experiment noncommuting --output outputs/noncommuting
python run_experiments.py --experiment rkhs --jobs 4 --output outputs/rkhs

# Reference eigensolver; slower at high rank:
python run_experiments.py --smoke --solver dense --output outputs/dense_check

# A different reproducible Monte Carlo run:
python run_experiments.py --seed 20260920 --jobs 4 --output outputs/independent
```

`--n` and `--trials` deliberately override the prescribed sample sizes/counts;
such outputs are labeled in the manifest and comparison CSV. Sample sizes must
be multiples of four and at least eight. No observations are silently discarded.

## Files and outputs

| File | Contents |
|---|---|
| `bounds.py` | Bound formulas, exact population moments, and rank-one algebra |
| `run_experiments.py` | Fixed-seed simulation, multiprocessing, CSV reporting |
| `tests/test_bounds.py` | Independent dense and exhaustive mathematical checks |
| `requirements.txt` | Pinned numerical-library versions |
| `RUN_REPORT.md` | Completed reference-run tables and validation summary |
| `rebuttal_reference.csv` | Earlier rounded rebuttal numbers, retained for historical comparison |
| `results/trials.csv` | 560 computed rows: 160 noncommuting and 400 RKHS condition-trials |
| `results/summary.csv` | 24 rows: eight rank conditions times three methods |
| `results/comparison_to_rebuttal.csv` | Reported value, computed value, difference, rounding match |
| `results/embedding.npz` | Fixed dense 1000-by-20 matrix Q with orthonormal columns |
| `results/manifest.json` | Options, seed, versions, source/output hashes, completion status |

The trial CSV includes raw radii, exact intrinsic and ambient oracles, normalized
radii, true variance and trace, all OEB upper bounds, sample hashes, both center
errors, coverage indicators, and noncommutator counts. The general self-adjoint
OEB radius is also included for the formula distinction explained below.

Summary ratios are arithmetic means across trials. `sd_ratio` uses ddof=1;
`mc_standard_error` is SD/sqrt(trials). The 2.5% and 97.5% columns are empirical
quantiles of trial ratios, **not** confidence intervals for their mean. Coverage
uses the appropriate center for each method and includes two-sided 95%
Clopper-Pearson binomial intervals. These additional diagnostics do not substitute
for a theorem, and 40 or 100 trials do not establish exact 95% coverage.

`trials.csv` is checkpointed as tasks finish. A complete run must have
`"status": "complete"` in its manifest; an interrupted CSV can be partial. The
runner does not resume partial runs. Choose a new output directory to restart.

## Randomness and shared samples

The default master seed is **20260919**, using NumPy `PCG64` and `SeedSequence`.
Each condition/trial has its own deterministic stream:

* Noncommuting: `SeedSequence([seed, 0, r, trial])`.
* RKHS: `SeedSequence([seed, 1, 0, trial])`.
* Embedding: `SeedSequence([seed, 2])`, followed by QR.

Trial indices are zero-based. Signs are generated as an array of shape `(rank,n)`
and then transposed. A single 10000-by-256 sign array is generated for each RKHS
trial, and ranks 32, 64, and 128 use its leading columns. Thus the 100 samples
are genuinely shared across all four truncations. Changing the maximum rank
preserves smaller-rank samples when n is unchanged. Trials are independent.
The empirical methods at a given condition all use exactly the same observations.

Sample SHA-256 hashes support checking seed replay. Pinned dependencies and a
fixed environment reproduce the numerical results; different BLAS/LAPACK builds
can change the last floating-point digits. The saved Q can have QR sign differences
across platforms without changing any reported invariant statistic.

## Models and exact intrinsic oracles

Write `X_i=v_i v_i^T`, `C=E[X_i]`, and `Sigma=E[(X_i-C)^2]`.
The operator variance Sigma is distinct from the feature covariance C.

For either experiment, take independent Rademacher signs epsilon_ij and

$$
v_{ij}=\sqrt{p_j}\,\epsilon_{ij},\quad
c=\sum_jp_j,\quad C=\operatorname{diag}(p_j).
$$

Every observation has constant squared feature norm c, so
`X_i^2=c X_i`, `||X_i||=c`, and `tr(X_i^2)=c²`. Therefore, exactly,

$$
\Sigma=cC-C^2=\operatorname{diag}\{p_j(c-p_j)\},\qquad
\sigma^2=\max_jp_j(c-p_j),\quad
\tau=\sum_jp_j(c-p_j),\quad d_{\rm eff}=\tau/\sigma^2.
$$

The **exact intrinsic oracle** means the Bernstein radius evaluated at these
exact population parameters, not an optimal radius and not a numerical inversion
of the sharper Bennett bound:

$$
r_{\rm int}=\sqrt{\frac{2\sigma^2}{n}\log\frac{2d_{\rm eff}}\delta}
 +\frac{c}{3n}\log\frac{2d_{\rm eff}}\delta.
$$

All ratios in the experiment tables are `method_radius / r_int`. The ambient
oracle uses precisely the same formula with d in place of d_eff. These are
two-sided operator-norm radii. The factor c is valid because `0 <= X_i,C <= c I`
implies `||X_i-C|| <= c`. No Monte Carlo estimate is used in either oracle.

### 1. Dense noncommuting updates

For effective rank r, use `p_j=1/r`, `u_i=(epsilon_i1,...,epsilon_ir)/sqrt(r)`
and define the actual 1000-dimensional observation as

$$
X_i=Q_r u_i u_i^T Q_r^T,\qquad Q_r^TQ_r=I_r.
$$

`Q_r=Q[:,:r]` is dense. In ambient coordinates the mean is `Q_r Q_r^T/r`,
and Sigma has r nonzero eigenvalues `(r-1)/r²`. Thus d_eff is exactly r,
and c=B=1. Calculations use the active r-dimensional coordinates, because
orthogonal embedding preserves all products, traces, nonzero spectra, norms,
and commutators. **MEB1 and MEB2 still pay ambient d=1000**, not r. The dense
embedding is saved so the observations can be materialized if desired:

```python
import numpy as np
from run_experiments import random_signs
q = np.load("results/embedding.npz")["Q"][:, :10]
u = random_signs(20260919, 0, 0, 10, 20000, condition=10) / np.sqrt(10)
x0 = np.outer(q @ u[0], q @ u[0])   # first actual 1000-by-1000 update
```

Commutator frequency uses **all n-1 overlapping adjacent pairs**. For unit u,v,
`[uu^T,vv^T]=(u^T v)(uv^T-vu^T)`, nonzero precisely when
`0 < |u^T v| < 1`. The code tests the unscaled integer sign dot product against
0 and +/-r, avoiding an arbitrary floating-point tolerance. Its exact probability
is `1 - 2^(1-r) - 1_{r even} binom(r,r/2)/2^r`:

| r | Exact nonzero-commutator probability |
|---:|---:|
| 3 | 0.7500000000 |
| 5 | 0.9375000000 |
| 10 | 0.7519531250 |
| 20 | 0.8238010406 |

Zero-commutator adjacent pairs are neither discarded nor resampled. The reported
frequency is the total count divided by the total number of adjacent pairs.

### 2. Infinite-dimensional RKHS and finite-rank compressions

Let the input space be `{−1,+1}^N` with independent fair coordinate signs, and

$$
p_j=0.1(0.9)^{j-1},\quad
\phi(x)_j=\sqrt{p_j}\,x_j,\quad
k(x,y)=\sum_{j\ge1}p_jx_jy_j.
$$

The coordinate functions are orthonormal under the input distribution and have
Mercer eigenvalues p_j. All p_j are positive, so the feature space is infinite
dimensional. The series is absolutely convergent and `k(x,x)=sum p_j=1`.

For truncation m, `v=phi_m(x)` and

$$
c_m=1-0.9^m,\qquad C_m=\operatorname{diag}(p_1,\ldots,p_m),\qquad
\Sigma_m=\operatorname{diag}\{p_j(c_m-p_j)\}_{j=1}^m.
$$

The features are **not renormalized after truncation**. Each method uses the
valid, truncation-specific envelopes `c=c_m`, `B=c_m²`, and the denominator is
recomputed from Sigma_m at every rank. The ambient baselines use d=m. No
infinite-rank oracle is substituted into a finite-rank comparison, and these
radii cover C_m, not the untruncated C. As m grows, d_eff tends to `200/19`,
approximately 10.526315789. At m=256, r_int is about 0.010630694.

## Formula-to-code correspondence

OEB follows Section 5 of the manuscript and the PSD specialization documented
in its experiment appendix (Appendix D). MEB1/MEB2 follow Wang and Ramdas, *Sharp Matrix Empirical Bernstein
Inequalities*, [arXiv v5](https://arxiv.org/html/2411.09516v5), Theorem 3.1 and
Corollary 4.3. That source numbers the baseline formulas (17), (27), and (28).

### OEB: paired trace and variance estimates

The function `paired_statistics` computes

$$
Y_i=\tfrac12(X_{2i-1}-X_{2i})^2,\quad
Y'_j=\tfrac12(Y_{2j-1}-Y_{2j})^2,\quad
\widehat\Sigma=\frac2n\sum_{i=1}^{n/2}Y_i,
$$

and scalar samples `Z_i=tr(Y_i)`, `Z'_j=tr(Y'_j)`.
Here `E[Y'_j]` is the variance of Y_i; it is not generally the variance of
`(X_i-C)^2`. The code uses the variance of the paired estimator in this second level.

Define the scalar upper bound from Theorem 3.5, implemented by `scalar_eb_upper`,

$$
U(z;b,\alpha)=\bar z+
\sqrt{\frac{2s_z^2\log(2/\alpha)}{N}}
+\frac{7b\log(2/\alpha)}{3(N-1)},\quad
s_z^2=\frac{\sum_i(z_i-\bar z)^2}{N-1}.
$$

With the paper's probability allocation
`delta1=delta*(n-2)/n`, `delta2=delta3=delta/n`, the ingredients are

$$
\tau_u=U(Z;2B,\delta_2),\quad
\tau'_u=U(Z';4Bc^2,\delta_3/2),
$$

$$
\sigma_u^2=\|\widehat\Sigma\|
+\sqrt{\frac{2\tau'_u\log(4/\delta_3)}{n/2}}
+\frac{2c^2\log(4/\delta_3)}{3(n/2)}.
$$

These are equations (13), (14), and (12)/(15). In `oeb_radius`, let
`L=log((2/delta1)*max(tau_u/sigma_u²,1))`. The reported OEB radius is

$$
R_{\rm OEB}=\sqrt{2\sigma_u^2L/n}+cL/(3n).
$$

**PSD specialization:** the general self-adjoint equation (16) has
`2c L/(3n)`. For these positive semidefinite observations the centered norm
envelope is c, so Corollary 4.2 yields the final term above. This is the
specialization used in Tables 1 and 2 and described in the experiment appendix.
`oeb_general_selfadjoint_radius` separately exports equation (16) literally.
The paired-estimator envelopes `2B`, `4Bc²`, `2c²` are kept as in the paper;
no additional PSD tightening or capping is applied to those terms.
The ratio `tau_u/sigma_u²` itself need not upper-bound the true d_eff.

### MEB1: matrix plug-in bound

`meb1_radius` uses the same paired variance estimate and the two-sided
specialization of equation (17). Put
`A=log(2nd/((n-1)delta))` and `D=log(4nd/delta)`. Then

$$
R_{\rm MEB1}=\frac{cA}{3n}
+\sqrt{\frac{2\|\widehat\Sigma\|A}{n}}
+c\left(\sqrt{5/3}+1\right)\frac{\sqrt{AD}}n.
$$

The c factors come from applying the [0,I] result to X_i/c. Its center is the
ordinary sample mean, as for OEB. The 2d factors are the two-sided union bound.

### MEB2: predictable weights and their own center

Normalize `A_i=X_i/c`, and put `L=log(2d/delta)`. For k>=1 define
`Cbar_k=(1/k)sum A_i` and
`Vbar_k=(1/k)sum (A_i-Cbar_k)^2`; initialize `Cbar_0=0`, `Vbar_0=I/4`.
Equations (27)--(28) become

$$
\gamma_i=\sqrt{\frac{2L}{n\max(\|\bar V_{i-1}\|,5L/n)}},\quad
\psi(\gamma)=-\log(1-\gamma)-\gamma,
$$

$$
T=\sum_{i=1}^n\psi(\gamma_i)(A_i-\bar C_{i-1})^2,\quad
R_{\rm MEB2}=c\frac{L+\lambda_{\max}(T)}{\sum_i\gamma_i},\quad
\widehat C_\gamma=\frac{\sum_i\gamma_iX_i}{\sum_i\gamma_i}.
$$

Each weight uses only past observations; the variance floor ensures
`gamma_i <= sqrt(2/5) < 1`. Coverage is checked using
`||C_hat_gamma-C|| <= R_MEB2`, **not** the error of the unweighted mean.
No oracle variance enters this method. Applying the positive and negative
versions with error delta/2 gives the two-sided radius implemented here.

## Efficient algebra and numerical validation

The paired fourth moments retain noncommuting products. For `U=[a,b]`,
`Y=UHU^T`, where `H=[[a·a,-a·b],[-a·b,b·b]]/2`. Products of two such Y's
can be traced through 2-by-2 Gram matrices. This avoids storing n dense matrices.

MEB2 uses the identity `Vbar_k=c*Cbar_k-Cbar_k²` in unnormalized coordinates,
valid because every rank-one observation satisfies `X_i²=c X_i`.
When the largest eigenvalue of Cbar_k is at most c/2, its eigenvalue determines
the largest eigenvalue of that variance polynomial. Otherwise the code checks
the entire spectrum. The default symmetric ARPACK solve uses tolerance 1e-11,
checks the residual, and falls back to a full dense spectrum on nonconvergence.
Small dimensions and initial steps use a full dense spectrum directly.
`--solver dense` always uses the full spectrum. Both routes solve the same formula;
there is no feature-tail cutoff or approximate population oracle.

For the second MEB2 pass let `S_j=sum_{i<=j}X_i`, `w_j=psi(gamma_j)`,
`t_j=sum_{i>j} w_i/(i-1)²`, and set `w_1/(1-1)=0` by convention only in
the following coefficient. Direct expansion gives

$$
\sum_jw_j(X_j-S_{j-1}/(j-1))^2
=c\sum_j(w_j+t_j)X_j
+\sum_j\left(t_j-\frac{w_j}{j-1}\right)
  (X_jS_{j-1}+S_{j-1}X_j),
$$

where the first predictor is zero. Store `S_{j-1}v_j` in the first pass;
the right-hand side is then evaluated by weighted matrix products and divided
by c² to obtain T. This is an algebraic rearrangement of the sequential sum.

The tests independently verify paired second/fourth moments against explicit
matrix multiplication, population moments by enumerating all signs at small
rank, MEB2 against a literal sequential implementation, iterative versus dense
eigensolvers, embedding invariance, exact commutator tests, Bessel correction,
the oracle scale, and nested random samples.

## Historical comparison and scope

The manuscript uses the computed seed-20260919 results in `results/summary.csv`.
For provenance, the runner also compares new runs with the earlier rounded
rebuttal values and exports `comparison_to_rebuttal.csv`. Those historical
values are never used to generate samples, choose seeds, tune bounds, or
populate computed results. No seed search was performed.

The noncommuting experiment intentionally has a known finite active subspace;
an ambient theorem applied after an explicit reduction to that known subspace
could use d=r. The requested comparison instead retains d=1000, as in the
paper. The RKHS experiment illustrates stabilization of finite truncations;
it is not a direct simulation of infinitely many feature coordinates.
