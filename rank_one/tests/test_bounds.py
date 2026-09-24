"""Mathematical cross-checks against literal dense definitions, not table fitting."""
import itertools
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits
from bounds import (bernstein_radius, commutator_counts, exact_population,
                    meb1_radius, meb2_radius, oeb_radius, paired_statistics, scalar_eb_upper)
from run_experiments import random_signs


def literal_meb2(v, dimension, delta, c):
    n, d = v.shape
    observations = np.einsum('ni,nj->nij', v, v)/c
    total = np.zeros((d, d))
    weighted = np.zeros_like(total)
    a = np.zeros_like(total)
    weights = []
    l = math.log(2*dimension/delta)
    for i, x in enumerate(observations):
        past = total/i if i else np.zeros_like(total)
        if i:
            centered = observations[:i]-past
            variance = np.mean(centered @ centered, axis=0)
            vhat = np.linalg.eigvalsh(variance)[-1]
        else:
            vhat = 1/4
        gamma = math.sqrt(2*l/(n*max(vhat, 5*l/n)))
        residual = x-past
        a += (-math.log1p(-gamma)-gamma)*(residual @ residual)
        weighted += gamma*x
        weights.append(gamma)
        total += x
    return c*(l+np.linalg.eigvalsh(a)[-1])/sum(weights), c*weighted/sum(weights), weights, a


class MathematicalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limiter = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.limiter.restore_original_limits()

    def test_paired_moments_noncommuting_general_norms(self):
        v = np.random.default_rng(4).normal(size=(24, 6))/4
        x = np.einsum('ni,nj->nij', v, v)
        diff = x[::2]-x[1::2]
        y = diff @ diff/2
        dy = y[::2]-y[1::2]
        yp = dy @ dy/2
        fast = paired_statistics(v)
        assert_allclose(fast.covariance, y.mean(axis=0), atol=1e-14)
        assert_allclose(fast.traces, np.trace(y, axis1=1, axis2=2), atol=1e-14)
        assert_allclose(fast.fourth_traces, np.trace(yp, axis1=1, axis2=2), atol=1e-14)

    def test_exact_oracle_by_enumerating_entire_distribution(self):
        for p in (np.full(4, .25), .1*.9**np.arange(4)):
            signs = np.array(list(itertools.product([-1, 1], repeat=4)))
            v = signs*np.sqrt(p)
            x = np.einsum('ni,nj->nij', v, v)
            centered = x-x.mean(axis=0)
            sigma = np.mean(centered @ centered, axis=0)
            c, variance, trace, rank = exact_population(p)
            assert_allclose(x.mean(axis=0), np.diag(p), atol=1e-14)
            assert_allclose(sigma, np.diag(p*(c-p)), atol=1e-14)
            assert_allclose([variance, trace, rank],
                            [np.linalg.eigvalsh(sigma)[-1], np.trace(sigma),
                             np.trace(sigma)/np.linalg.eigvalsh(sigma)[-1]], atol=1e-14)

    def test_meb2_two_pass_matches_literal_sequential_matrices(self):
        for n, m in ((12, 3), (40, 8), (80, 32)):
            p = .1*.9**np.arange(m)
            v = random_signs(202, 1, 0, m, n)*np.sqrt(p)
            for solver in ("dense", "iterative"):
                actual = meb2_radius(v, m+9, .05, p.sum(), solver, return_details=True)
                expected = literal_meb2(v, m+9, .05, p.sum())
                for a, b in zip(actual, expected):
                    assert_allclose(a, b, rtol=1e-10, atol=1e-12)

    def test_meb2_iterative_solver_at_nontrivial_weights(self):
        m, n = 64, 1200
        p = .1*.9**np.arange(m)
        v = random_signs(311, 1, 1, m, n)*np.sqrt(p)
        dense = meb2_radius(v, m, .05, p.sum(), "dense", True)
        fast = meb2_radius(v, m, .05, p.sum(), "iterative", True)
        for a, b in zip(dense, fast):
            assert_allclose(a, b, rtol=1e-9, atol=1e-11)

    def test_embedding_invariance(self):
        rng = np.random.default_rng(12)
        q, _ = np.linalg.qr(rng.normal(size=(30, 5)))
        v = random_signs(12, 0, 0, 5, 16)/np.sqrt(5)
        active, dense = paired_statistics(v), paired_statistics(v @ q.T)
        assert_allclose(q @ active.covariance @ q.T, dense.covariance, atol=1e-14)
        assert_allclose(active.fourth_traces, dense.fourth_traces, atol=1e-14)
        ra, ma = meb2_radius(v, 30, .05, 1, "dense")
        rb, mb = meb2_radius(v @ q.T, 30, .05, 1, "dense")
        assert_allclose(ra, rb, atol=1e-14)
        assert_allclose(q @ ma @ q.T, mb, atol=1e-14)

    def test_commutator_integer_test_exhaustive(self):
        for r in (3, 4, 5):
            signs = np.array(list(itertools.product([-1, 1], repeat=r)))
            first = signs[0]/math.sqrt(r)
            x = np.outer(first, first)
            actual = sum(np.linalg.norm(x @ np.outer(s, s)-np.outer(s, s) @ x) > 1e-12
                         for s in signs/math.sqrt(r))
            _, _, theoretical = commutator_counts(signs)
            self.assertAlmostEqual(actual/len(signs), theoretical)
        s = random_signs(1, 0, 0, 10, 100)
        count, _, _ = commutator_counts(s)
        matrices = np.einsum('ni,nj->nij', s/np.sqrt(10), s/np.sqrt(10))
        comm = matrices[:-1] @ matrices[1:]-matrices[1:] @ matrices[:-1]
        self.assertEqual(count, np.count_nonzero(np.linalg.norm(comm, axis=(1, 2)) > 1e-12))

    def test_nested_samples_and_seed_replay(self):
        full = random_signs(45, 1, 2, 256, 40)
        small = random_signs(45, 1, 2, 32, 40)
        np.testing.assert_array_equal(full[:, :32], small)
        np.testing.assert_array_equal(full, random_signs(45, 1, 2, 256, 40))

    def test_radii_and_centers_scale_with_observations(self):
        n, m = 40, 5
        v = random_signs(91, 1, 0, m, n)/math.sqrt(m)
        # v -> sqrt(a)*v means X -> a*X. Every radius and center scales by a.
        scale = .37
        first = paired_statistics(v)
        second = paired_statistics(math.sqrt(scale)*v)
        o1 = oeb_radius(first, n, .05, 1, 1)
        o2 = oeb_radius(second, n, .05, scale, scale**2)
        for key in ("oeb_radius", "oeb_general_selfadjoint_radius"):
            self.assertAlmostEqual(o2[key], scale*o1[key])
        self.assertAlmostEqual(meb1_radius(o2['variance_hat'], n, 1000, .05, scale),
                               scale*meb1_radius(o1['variance_hat'], n, 1000, .05, 1))
        a = meb2_radius(v, 1000, .05, 1, "dense", True)
        b = meb2_radius(math.sqrt(scale)*v, 1000, .05, scale, "dense", True)
        assert_allclose(b[0], scale*a[0], atol=1e-12)
        assert_allclose(b[1], scale*a[1], atol=1e-12)
        assert_allclose(b[2], a[2], atol=1e-12)
        l = math.log(2*o1['empirical_trace_variance_ratio']/(.05*(n-2)/n))
        self.assertAlmostEqual(o1['oeb_general_selfadjoint_radius']-o1['oeb_radius'], l/(3*n))

    def test_meb2_weights_are_predictable(self):
        n, m = 1000, 5
        v = random_signs(222, 1, 0, m, n)/math.sqrt(m)
        changed = v.copy()
        changed[300:] = random_signs(223, 1, 0, m, n-300)/math.sqrt(m)
        original = meb2_radius(v, m, .05, 1, "dense", True)
        modified = meb2_radius(changed, m, .05, 1, "dense", True)
        # The weight on the first changed observation must also be unchanged.
        assert_allclose(original[2][:301], modified[2][:301], rtol=0, atol=0)

    def test_bessel_correction_and_oracle_value(self):
        z = np.array([.1, .4, .2, .8])
        pairvar = sum((z[i]-z[j])**2 for i in range(4) for j in range(i))/(4*3)
        expected = z.mean()+math.sqrt(pairvar*2*math.log(40)/4)+7*math.log(40)/9
        self.assertAlmostEqual(scalar_eb_upper(z, 1, .05), expected)
        p = .1*.9**np.arange(256)
        c, variance, _, rank = exact_population(p)
        oracle = bernstein_radius(variance, rank, 10000, .05, c)
        self.assertAlmostEqual(oracle, .010631, places=6)


if __name__ == "__main__":
    unittest.main()
