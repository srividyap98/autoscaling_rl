"""Numerical gradient, terminal, startup, accounting and reproducibility checks."""
import unittest
import numpy as np
from autoscaling.environment import AutoscalingEnv, Config
from autoscaling.ppo import actor_loss_gradient, MLP, gae, entropy, executed_distribution, entropy_diagnostics

class Tests(unittest.TestCase):

    def test_actor_gradient(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(5, 3))
        a = np.array([0, 1, 2, 0, 1])
        old = np.array([-1.7, -0.2, -0.8, -1, -2])
        adv = np.array([1, -1, 0.7, -0.3, 0.5])
        _, g, _ = actor_loss_gradient(z, a, old, adv)
        for ix in np.ndindex(z.shape):
            h = 1e-06
            z[ix] += h
            plus = actor_loss_gradient(z, a, old, adv)[0]
            z[ix] -= 2 * h
            minus = actor_loss_gradient(z, a, old, adv)[0]
            z[ix] += h
            self.assertAlmostEqual(g[ix], (plus - minus) / (2 * h), places=6)

    def test_unclipped_actor_gradient(self):
        rng = np.random.default_rng(4)
        z = rng.normal(size=(4, 3))
        a = np.array([2, 0, 1, 2])
        old = np.array([-3.0, -0.1, -2.5, -0.05])
        adv = np.array([1, -1, 0.5, -0.8])
        _, g, s = actor_loss_gradient(z, a, old, adv, clip=np.inf)
        self.assertEqual(s['clip_fraction'], 0)
        for ix in np.ndindex(z.shape):
            h = 1e-06
            z[ix] += h
            plus = actor_loss_gradient(z, a, old, adv, clip=np.inf)[0]
            z[ix] -= 2 * h
            minus = actor_loss_gradient(z, a, old, adv, clip=np.inf)[0]
            z[ix] += h
            self.assertAlmostEqual(g[ix], (plus - minus) / (2 * h), places=6)

    def test_executed_entropy(self):
        p = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
        valid = np.array([[True, True, True], [True, True, False]])
        q = executed_distribution(p, valid)
        np.testing.assert_allclose(q, [[0.2, 0.3, 0.5], [0.2, 0.8, 0.0]])
        d = entropy_diagnostics(p, valid)
        self.assertAlmostEqual(d['invalid_mass'], 0.25)
        self.assertLess(d['executed_entropy'], d['policy_entropy'])
        self.assertAlmostEqual(entropy(np.full(3, 1 / 3)), np.log(3))

    def test_valid_actions(self):
        e = AutoscalingEnv(Config(maximum=3))
        e.reset(0, {'initial': 1})
        np.testing.assert_array_equal(e.valid_actions(), [False, True, True])
        e.step(2)
        e.step(2)
        np.testing.assert_array_equal(e.valid_actions(), [True, True, False])

    def test_gae_lambda_extremes(self):
        r = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, 0.2, 0.1])
        nv = np.array([0.2, 0.1, 0.0])
        term = np.array([False, False, True])
        adv, _ = gae(r, v, nv, term, gamma=0.9, lam=1)
        np.testing.assert_allclose(adv, [1 + 0.9 * 2 + 0.81 * 3 - 0.5, 2 + 0.9 * 3 - 0.2, 3 - 0.1])
        adv, _ = gae(r, v, nv, term, gamma=0.9, lam=0)
        np.testing.assert_allclose(adv, r + 0.9 * nv * ~term - v)

    def test_mlp_gradient(self):
        rng = np.random.default_rng(1)
        m = MLP(9, 3, rng)
        x = rng.normal(size=(4, 9))
        dy = rng.normal(size=(4, 3))
        _, cache = m.forward(x)
        g = m.backward(dy, cache)
        for k in ['w0', 'w1', 'w2', 'b0', 'b1', 'b2']:
            ix = tuple((0 for _ in m.p[k].shape))
            h = 1e-06
            m.p[k][ix] += h
            plus = np.sum(m.forward(x)[0] * dy)
            m.p[k][ix] -= 2 * h
            minus = np.sum(m.forward(x)[0] * dy)
            m.p[k][ix] += h
            self.assertAlmostEqual(g[k][ix], (plus - minus) / (2 * h), places=6)

    def test_gae_terminal(self):
        adv, ret = gae(np.array([1.0, 2.0, 3.0]), np.zeros(3), np.array([0, 100, 0]), np.array([False, True, True]), gamma=1, lam=1)
        np.testing.assert_allclose(adv, [3, 2, 3])
        np.testing.assert_allclose(ret, adv)

    def test_bounds_startup_and_accounting(self):
        e = AutoscalingEnv(Config(seconds=40))
        e.reset(5, {'kind': 'steady'})
        e.step(2)
        self.assertEqual(e.active, 2)
        e.step(1)
        self.assertEqual(e.active, 3)
        e.step(1)
        self.assertEqual(e.active, 3)
        _, _, done, _, _ = e.step(0)
        self.assertTrue(done)
        m = e.metrics()
        self.assertEqual(m['arrivals'], m['served'] + m['failed'])
        e.reset(1)
        e.step(0)
        e.step(0)
        self.assertEqual(e.active, 1)

    def test_seed_reproducibility(self):
        a = AutoscalingEnv()
        b = AutoscalingEnv()
        a.reset(8, {'kind': 'bursty'})
        b.reset(8, {'kind': 'bursty'})
        np.testing.assert_array_equal(a.arrivals, b.arrivals)
        for act in [0, 2, 1, 2, 0]:
            ra = a.step(act)
            rb = b.step(act)
            np.testing.assert_array_equal(ra[0], rb[0])
            self.assertEqual(ra[1], rb[1])
if __name__ == '__main__':
    unittest.main()
