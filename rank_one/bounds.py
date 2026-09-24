"""OEB and two-sided Wang--Ramdas baselines for constant-norm rank-one data.

Rows of features are v_i; observations are X_i = v_i v_i.T.  Never form an
n-by-d-by-d tensor.  All computation is in real float64 arithmetic.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


def symmetric(a):
    return (a + a.T) * 0.5


def psd_norm(a):
    return max(0.0, float(eigh(symmetric(a), eigvals_only=True,
                              subset_by_index=[len(a)-1, len(a)-1])[-1]))


def operator_norm(a):
    w = np.linalg.eigvalsh(symmetric(a))
    return float(max(abs(w[0]), abs(w[-1])))


def bernstein_radius(variance, complexity, n, delta, centered_bound):
    """Exact population parameters, closed-form Bernstein (not Bennett) radius."""
    if variance <= 0 or complexity < 1 or n < 1 or not 0 < delta < 1:
        raise ValueError("Require positive variance, complexity >= 1, n > 0, 0 < delta < 1")
    log_term = math.log(2 * complexity / delta)
    return math.sqrt(2 * variance * log_term / n) + centered_bound * log_term / (3*n)


def scalar_eb_upper(z, envelope, delta):
    """Paper Theorem 3.5: Bessel-corrected scalar variance (ddof=1)."""
    k = len(z)
    if k < 2 or not 0 < delta < 1:
        raise ValueError("At least two observations and delta in (0,1) required")
    l = math.log(2 / delta)
    return float(np.mean(z) + np.std(z, ddof=1)*math.sqrt(2*l/k)
                 + 7*envelope*l/(3*(k-1)))


@dataclass
class PairedStatistics:
    covariance: np.ndarray
    traces: np.ndarray
    fourth_traces: np.ndarray


def paired_statistics(features):
    """Section 5.1: Y_i=(X_2i-X_2i+1)^2/2 and Z'_j=tr((Y_2j-Y_2j+1)^2)/2.

    If U=[a,b], Y=U H U.T with H=[[a.a,-a.b],[-a.b,b.b]]/2.
    All fourth traces use 2-by-2 Gram products, including noncommuting cross terms.
    """
    v = np.asarray(features, dtype=np.float64)
    n, _ = v.shape
    if n < 8 or n % 4:
        raise ValueError("n must be a multiple of four and at least eight")
    a, b = v[::2], v[1::2]
    aa = np.einsum('ij,ij->i', a, a)
    bb = np.einsum('ij,ij->i', b, b)
    ab = np.einsum('ij,ij->i', a, b)
    cov = (a.T @ (aa[:, None]*a) + b.T @ (bb[:, None]*b)
           - a.T @ (ab[:, None]*b) - b.T @ (ab[:, None]*a)) / n
    u = np.stack((a, b), axis=2)
    h = np.empty((n//2, 2, 2))
    h[:, 0, 0], h[:, 1, 1] = aa/2, bb/2
    h[:, 0, 1] = h[:, 1, 0] = -ab/2
    gram = np.einsum('ndi,ndj->nij', u, u)
    hg = h @ gram
    trace_sq = np.einsum('nij,nji->n', hg, hg)
    cross_gram = np.einsum('ndi,ndj->nij', u[::2], u[1::2])
    cross = np.einsum('nij,nji->n', h[::2] @ cross_gram,
                       h[1::2] @ cross_gram.transpose(0, 2, 1))
    fourth = (trace_sq[::2] + trace_sq[1::2])/2 - cross
    # Only roundoff-size negative values are possible for these squared norms.
    if np.min(fourth) < -1e-12:
        raise ArithmeticError("Negative squared Hilbert--Schmidt norm")
    return PairedStatistics(symmetric(cov), (aa*aa + bb*bb)/2 - ab*ab,
                            np.maximum(fourth, 0))


def oeb_radius(stats, n, delta, c, hs_squared_bound):
    """(12)--(16), Theorem 5.6 allocation, PSD centered bound c in final step.

    Eq. (16) prints 2c for arbitrary self-adjoint data. Here 0<=X<=cI gives
    ||X-E X||<=c, so the same Corollary 4.2 derivation permits c. Both versions
    are exported; only the PSD version is used in the reported OEB column.
    """
    d1, d2, d3 = delta*(n-2)/n, delta/n, delta/n
    tau = scalar_eb_upper(stats.traces, 2*hs_squared_bound, d2)
    tau_prime = scalar_eb_upper(stats.fourth_traces, 4*hs_squared_bound*c*c, d3/2)
    l = math.log(4/d3)
    sample_variance = psd_norm(stats.covariance)
    sigma_upper = sample_variance + math.sqrt(2*tau_prime*l/(n/2)) + 2*c*c*l/(3*(n/2))
    effective_upper_ratio = max(tau/sigma_upper, 1.0)
    radius = bernstein_radius(sigma_upper, effective_upper_ratio, n, d1, c)
    general = bernstein_radius(sigma_upper, effective_upper_ratio, n, d1, 2*c)
    return {"oeb_radius": radius, "oeb_general_selfadjoint_radius": general,
            "variance_hat": sample_variance,
            "tau_upper": tau, "tau_prime_upper": tau_prime,
            "sigma_squared_upper": sigma_upper,
            "empirical_trace_variance_ratio": effective_upper_ratio}


def meb1_radius(variance_hat, n, dimension, delta, c):
    """Wang--Ramdas Theorem 3.1 (17), alpha=delta/2 and X/c in [0,I]."""
    a = math.log(2*n*dimension/((n-1)*delta))
    b = math.log(4*n*dimension/delta)
    return (c*a/(3*n) + math.sqrt(2*variance_hat*a/n)
            + c*(math.sqrt(5/3)+1)*math.sqrt(a*b)/n)


def _variance_from_sum(total, k, c, solver, start):
    """For constant norm data, Vbar_k=c*Cbar_k-Cbar_k^2 exactly.

    Above c/2, t(c-t) is decreasing, so inspect the entire spectrum there.
    Otherwise only the largest covariance eigenvalue is needed. ARPACK is a
    standard symmetric eigensolver; the dense option is the reference path.
    """
    d = len(total)
    if solver == "dense" or d <= 24 or k < 8:
        values = np.linalg.eigvalsh(total)/k
        return max(0.0, float(np.max(values*(c-values)))), start
    try:
        values, vectors = eigsh(total, k=1, which="LA", v0=start,
                                tol=1e-11, maxiter=300, ncv=min(20, d))
        lam = float(values[0])/k
        start = vectors[:, 0]
        if np.linalg.norm(total @ start - values[0]*start) > 1e-9*max(1, abs(values[0])):
            raise ArithmeticError("Eigensolver residual too large")
        if lam <= c/2:
            return max(0.0, lam*(c-lam)), start
    except (ArpackNoConvergence, ArithmeticError):
        pass
    values = np.linalg.eigvalsh(total)/k
    return max(0.0, float(np.max(values*(c-values)))), start


def meb2_radius(features, dimension, delta, c, solver="iterative", return_details=False):
    """Wang--Ramdas Corollary 4.3 (27)--(28), two-sided and rescaled by c.

    The predictor is the past unweighted mean. Weights use *past* classical
    variance, floor 5 log(2d/delta)/n, initialization Vbar_0=c^2 I/4.
    The returned center is the gamma-weighted mean, not the sample mean.
    Two passes evaluate the literal sequential formula in O(n*m^2) arithmetic
    apart from the per-step symmetric eigenproblems; see README for identity.
    """
    v = np.asarray(features, dtype=np.float64)
    n, m = v.shape
    if solver not in ("dense", "iterative"):
        raise ValueError("solver must be dense or iterative")
    if not np.allclose(np.einsum('ij,ij->i', v, v), c, rtol=1e-12, atol=1e-14):
        raise ValueError("MEB2 fast path requires ||v_i||^2 = c for every i")
    l = math.log(2*dimension/delta)
    floor = 5*l/n
    gamma = np.empty(n)
    products = np.empty_like(v)
    total = np.zeros((m, m))
    start = np.ones(m)/math.sqrt(m)
    for j, x in enumerate(v):
        variance = c*c/4 if j == 0 else _variance_from_sum(total, j, c, solver, start)
        if j:
            variance, start = variance
        gamma[j] = math.sqrt(2*l/(n*max(variance/(c*c), floor)))
        products[j] = total @ x  # S_{j-1} v_j, BEFORE updating total
        total += np.outer(x, x)
    psi = -np.log1p(-gamma) - gamma
    # g_i = psi_i / (i-1)^2, i>=2. tail[j] = sum_{i>j} g_i.
    denom = np.arange(n, dtype=float)
    g = np.zeros(n)
    g[1:] = psi[1:]/denom[1:]**2
    tail = np.zeros(n)
    tail[:-1] = np.cumsum(g[:0:-1])[::-1]
    cross_coeff = tail.copy()
    cross_coeff[1:] -= psi[1:]/denom[1:]
    diag = v.T @ ((c*(psi + tail))[:, None]*v)
    cross = v.T @ (cross_coeff[:, None]*products)
    self_normalizer = symmetric((diag + cross + cross.T)/(c*c))
    center = symmetric(v.T @ (gamma[:, None]*v)/gamma.sum())
    radius = c*(l + psd_norm(self_normalizer))/float(gamma.sum())
    if return_details:
        return radius, center, gamma, self_normalizer
    return radius, center


def exact_population(weights):
    """Rademacher features: C=diag(p), Sigma=diag(p*(sum(p)-p))."""
    p = np.asarray(weights, dtype=float)
    c = float(p.sum())
    spectrum = p*(c-p)
    variance = float(spectrum.max())
    trace = float(spectrum.sum())
    if variance <= 0:
        raise ValueError("Need at least two positive feature weights")
    return c, variance, trace, max(1.0, trace/variance)


def commutator_counts(signs):
    """Exact integer test for adjacent isotropic Rademacher rank-one updates."""
    s = np.asarray(signs, dtype=np.int64)
    dots = np.einsum('ij,ij->i', s[:-1], s[1:])
    r = s.shape[1]
    # [uu.T,vv.T]=0 exactly when u.v=0 or |u.v|=1.
    count = int(np.count_nonzero((dots != 0) & (np.abs(dots) != r)))
    theory = 1 - 2.0**(1-r)
    if r % 2 == 0:
        theory -= math.comb(r, r//2)/2**r
    return count, len(dots), theory
