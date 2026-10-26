import unittest
import numpy as np
from non_kpp.histogram_ot import Grid, histogram, score


class NonKPPHistogramOTTests(unittest.TestCase):
    def test_extension_retains_outliers_without_moving_grid(self):
        grid = Grid.from_reference(np.array([[0., 0., 0.], [1., 1., 1.]]))
        origin, width = grid.origin.copy(), grid.width.copy()
        h = histogram(np.array([[-100., 0., 0.], [100., 0., 0.], [.5, .5, .5]]), grid)
        self.assertEqual(int(h['counts'].sum()), 3)
        np.testing.assert_array_equal(origin, grid.origin)
        np.testing.assert_array_equal(width, grid.width)
        self.assertAlmostEqual(h['weights'].sum(), 1.)

    def test_third_coordinate_is_evaluated(self):
        grid = Grid(np.zeros(3), np.ones(3))
        a = histogram(np.array([[.1, .1, .1]]), grid)
        b = histogram(np.array([[.1, .1, 3.1]]), grid)
        self.assertAlmostEqual(score(a, b)['w2'], 3.)

    def test_weights_are_particle_frequencies(self):
        grid = Grid(np.zeros(3), np.ones(3))
        a = histogram(np.array([[.1, 0., 0.]]*3+[[1.1, 0., 0.]]), grid)
        b = histogram(np.array([[.1, 0., 0.]]+[[1.1, 0., 0.]]*3), grid)
        self.assertAlmostEqual(score(a, b)['w2_squared'], .5)

    def test_matches_existing_high_count_evaluator(self):
        from non_kpp.high_count_ot import histogram as old_histogram, score as old_score
        rng = np.random.default_rng(95)
        ref, pred = rng.normal(size=(67, 3)), rng.normal(size=(71, 3))
        grid = Grid.from_reference(ref)
        expected = old_score(old_histogram(ref, grid.origin, grid.width),
                             old_histogram(pred, grid.origin, grid.width))
        actual = score(histogram(ref, grid), histogram(pred, grid))
        self.assertAlmostEqual(actual['w2_squared'], expected['w2_squared'], places=12)


if __name__ == '__main__':
    unittest.main()
