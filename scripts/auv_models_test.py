import unittest

import numpy as np

from auv_models import auv_1DOF_simplified, diagonal_slow


class SimplifiedSurgeAUVTest(unittest.TestCase):
    def test_auv_1DOF(self):
        x = 0
        u = 0.1
        tau = 1
        m = 10
        d = 20

        state = np.array([x, u])
        thrust = np.array([tau])
        parameters = np.array([m, d])

        result = auv_1DOF_simplified(state, thrust, parameters)
        self.assertEqual(result, [0.1, -0.1])


class SlowDiagonalModelTests(unittest.TestCase):
    def test_dimensions_and_type(self):
        eta = [0, 0, 0, 1, 0, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        state = np.array(eta + nu, dtype=np.float64)

        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [0]
        B = [0]
        COG = [0, 0, 0]
        COB = [0, 0, 0]
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)

        thrust = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)

        x_dot = diagonal_slow(state, thrust, parameters)

        self.assertEqual(type(x_dot), np.ndarray)
        self.assertEqual(len(x_dot), 13)

    def test_zeros_result(self):
        eta = [0, 0, 0, 1, 0, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        state = np.array(eta + nu, dtype=np.float64)

        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [0]
        B = [0]
        COG = [0, 0, 0]
        COB = [0, 0, 0]
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)

        thrust = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)

        x_dot = diagonal_slow(state, thrust, parameters)
        solution = np.zeros(13)
        comparison = x_dot == solution

        self.assertTrue(comparison.all())


if __name__ == "__main__":
    unittest.main()
