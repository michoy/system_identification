import unittest

import numpy as np

from auv_models import auv_1DOF_simplified, diagonal_slow
import helper


class RotationTests(unittest.TestCase):
    def test_dimension(self):
        eta = np.zeros(7)
        eta[3:7] = np.array([1, 0, 0, 0])
        Jq = helper.Jq(eta)

        self.assertEqual(Jq.shape, (7, 6))

    def test_trivial_quat(self):
        eta = np.zeros(7)
        eta[3:7] = np.array([1, 0, 0, 0])
        Jq = helper.Jq(eta)

        res = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.5, 0, 0],
                [0, 0, 0, 0, 0.5, 0],
                [0, 0, 0, 0, 0, 0.5],
            ]
        )

        comparison = Jq == res
        self.assertTrue(comparison.all())


class MseTests(unittest.TestCase):
    def test_equal_y(self):
        y_true = np.array([[1, 1, 1], [1, 1, 1]])
        y_pred = np.array([[1, 1, 1], [1, 1, 1]])

        res = helper.mean_squared_error(y_true, y_pred)
        zeros = np.zeros(3)
        comparison = res == zeros
        self.assertTrue(comparison.all())

    def test_different_y_same_content(self):
        y_true = np.array([[1, 1, 1], [1, 1, 1]])
        y_pred = np.array([[2, 3, 4], [2, 3, 4]])

        res = helper.mean_squared_error(y_true, y_pred)
        solution = np.array([1, 4, 9])
        comparison = res == solution
        self.assertTrue(comparison.all())

    def test_different_y_diff_content(self):
        y_true = np.array([[1, 1, 1], [1, 1, 1]])
        y_pred = np.array([[2, 3, 4], [1, 1, 1]])

        res = helper.mean_squared_error(y_true, y_pred)
        solution = np.array([0.5, 2, 4.5])
        comparison = res == solution
        self.assertTrue(comparison.all())


if __name__ == "__main__":
    unittest.main()
