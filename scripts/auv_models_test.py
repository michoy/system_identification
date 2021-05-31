import unittest

import numpy as np

from auv_models import auv_1DOF_simplified, diagonal_slow_without_g
from helper import degrees_to_quat_rotation


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

        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [0]
        B = [0]
        COG = [0, 0, 0]
        COB = [0, 0, 0]

        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)

        x_dot = diagonal_slow_without_g(state, thrust, parameters)

        self.assertEqual(type(x_dot), np.ndarray)
        self.assertEqual(len(x_dot), 13)

    def test_zeros_result(self):
        eta = [0, 0, 0, 1, 0, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]

        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [0]
        B = [0]
        COG = [0, 0, 0]
        COB = [0, 0, 0]

        x_dot_expected = np.zeros(13)

        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        x_dot = diagonal_slow_without_g(state, thrust, parameters)

        for a, b in zip(x_dot, x_dot_expected):
            self.assertAlmostEqual(a, b)

    def test_restoring_sinking_linear(self):
        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [10]
        B = [0]
        COG = [0, 0, 0]
        COB = [0, 0, 0]

        # input
        position = [0, 0, 0]
        roll, pitch, yaw = [0, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        tau = [0, 0, 0, 0, 0, 0]

        # expected
        eta_dot = [0, 0, 0, 0, 0, 0, 0]
        linear_acc = [0, 0, 1]
        angular_acc = [0, 0, 0]

        x_dot_expected = np.array(eta_dot + linear_acc + angular_acc)
        orientation = degrees_to_quat_rotation(roll, pitch, yaw).tolist()
        eta = position + orientation
        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array(tau, dtype=np.float64)
        x_dot = diagonal_slow_without_g(state, thrust, parameters)

        for a, b in zip(x_dot, x_dot_expected):
            self.assertAlmostEqual(a, b)

    def test_restoring_angular_y(self):
        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [10]
        B = [10]
        COG = [0, 0, 0]
        COB = [0, 1, 0]

        # input
        position = [0, 0, 0]
        roll, pitch, yaw = [0, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        tau = [0, 0, 0, 0, 0, 0]

        # expected
        eta_dot = [0, 0, 0, 0, 0, 0, 0]
        linear_acc = [0, 0, 0]
        angular_acc = [-1, 0, 0]

        x_dot_expected = np.array(eta_dot + linear_acc + angular_acc)
        orientation = degrees_to_quat_rotation(roll, pitch, yaw).tolist()
        eta = position + orientation
        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array(tau, dtype=np.float64)
        x_dot = diagonal_slow_without_g(state, thrust, parameters)

        for a, b in zip(x_dot, x_dot_expected):
            self.assertAlmostEqual(a, b)

    def test_restoring_angular_x(self):
        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [10]
        B = [10]
        COG = [0, 0, 0]
        COB = [1, 0, 0]

        # input
        position = [0, 0, 0]
        roll, pitch, yaw = [0, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        tau = [0, 0, 0, 0, 0, 0]

        # expected
        eta_dot = [0, 0, 0, 0, 0, 0, 0]
        linear_acc = [0, 0, 0]
        angular_acc = [0, 1, 0]

        x_dot_expected = np.array(eta_dot + linear_acc + angular_acc)
        orientation = degrees_to_quat_rotation(roll, pitch, yaw).tolist()
        eta = position + orientation
        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array(tau, dtype=np.float64)
        x_dot = diagonal_slow_without_g(state, thrust, parameters)

        for a, b in zip(x_dot, x_dot_expected):
            self.assertAlmostEqual(a, b)

    def test_restoring_x_rotated_roll(self):
        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [10]
        B = [10]
        COG = [0, 0, 0]
        COB = [1, 0, 0]

        # input
        position = [0, 0, 0]
        roll, pitch, yaw = [90, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        tau = [0, 0, 0, 0, 0, 0]

        # expected
        pos_dot_expected = [0, 0, 0]
        quat_dot_expected = [0, 0, 0, 0]
        linear_acc = [0, 0, 0]
        angular_acc = [0, 0, -1]

        eta_dot_expected = pos_dot_expected + quat_dot_expected
        nu_dot_expected = linear_acc + angular_acc
        x_dot_expected = np.array(eta_dot_expected + nu_dot_expected)
        orientation = degrees_to_quat_rotation(roll, pitch, yaw).tolist()
        eta = position + orientation
        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array(tau, dtype=np.float64)

        x_dot = diagonal_slow_without_g(state, thrust, parameters)
        eta_dot = x_dot[0:7]
        nu_dot = x_dot[7:13]

        for a, b in zip(x_dot, x_dot_expected):
            self.assertAlmostEqual(a, b)

    def test_restoring_x_rotated_pitch(self):
        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [0]
        B = [10]
        COG = [0, 0, 0]
        COB = [1, 0, 0]

        # input
        position = [0, 0, 0]
        roll, pitch, yaw = [0, 90, 0]
        nu = [0, 0, 0, 0, 0, 0]
        tau = [0, 0, 0, 0, 0, 0]

        # expected
        pos_dot_expected = [0, 0, 0]
        quat_dot_expected = [0, 0, 0, 0]
        linear_acc = [1, 0, 0]
        angular_acc = [0, 0, 0]

        eta_dot_expected = pos_dot_expected + quat_dot_expected
        nu_dot_expected = linear_acc + angular_acc
        x_dot_expected = np.array(eta_dot_expected + nu_dot_expected)
        orientation = degrees_to_quat_rotation(roll, pitch, yaw).tolist()
        eta = position + orientation
        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array(tau, dtype=np.float64)

        x_dot = diagonal_slow_without_g(state, thrust, parameters)
        eta_dot = x_dot[0:7]
        nu_dot = x_dot[7:13]

        for a, b in zip(x_dot, x_dot_expected):
            self.assertAlmostEqual(a, b)

    def test_restoring_z_rotated_roll(self):
        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [10]
        B = [10]
        COG = [0, 0, 0]
        COB = [0, 0, 1]

        # input
        position = [0, 0, 0]
        roll, pitch, yaw = [90, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        tau = [0, 0, 0, 0, 0, 0]

        # expected
        pos_dot_expected = [0, 0, 0]
        quat_dot_expected = [0, 0, 0, 0]
        linear_acc = [0, 0, 0]
        angular_acc = [1, 0, 0]

        eta_dot_expected = pos_dot_expected + quat_dot_expected
        nu_dot_expected = linear_acc + angular_acc
        x_dot_expected = np.array(eta_dot_expected + nu_dot_expected)
        orientation = degrees_to_quat_rotation(roll, pitch, yaw).tolist()
        eta = position + orientation
        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array(tau, dtype=np.float64)

        x_dot = diagonal_slow_without_g(state, thrust, parameters)
        eta_dot = x_dot[0:7]
        nu_dot = x_dot[7:13]

        for a, b in zip(x_dot, x_dot_expected):
            self.assertAlmostEqual(a, b)

    def test_tau_sway(self):
        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [10]
        B = [10]
        COG = [0, 0, 0]
        COB = [0, 0, 0]

        # input
        position = [0, 0, 0]
        roll, pitch, yaw = [0, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        tau = [0, 10, 0, 0, 0, 0]

        # expected
        pos_dot_expected = [0, 0, 0]
        quat_dot_expected = [0, 0, 0, 0]
        linear_acc = [0, 1, 0]
        angular_acc = [0, 0, 0]

        eta_dot_expected = pos_dot_expected + quat_dot_expected
        nu_dot_expected = linear_acc + angular_acc
        x_dot_expected = np.array(eta_dot_expected + nu_dot_expected)
        orientation = degrees_to_quat_rotation(roll, pitch, yaw).tolist()
        eta = position + orientation
        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array(tau, dtype=np.float64)

        x_dot = diagonal_slow_without_g(state, thrust, parameters)
        eta_dot = x_dot[0:7]
        nu_dot = x_dot[7:13]

        for a, b in zip(x_dot, x_dot_expected):
            self.assertAlmostEqual(a, b)

    def test_tau_heave(self):
        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [10]
        B = [10]
        COG = [0, 0, 0]
        COB = [0, 0, 0]

        # input
        position = [0, 0, 0]
        roll, pitch, yaw = [0, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        tau = [0, 0, 10, 0, 0, 0]

        # expected
        pos_dot_expected = [0, 0, 0]
        quat_dot_expected = [0, 0, 0, 0]
        linear_acc = [0, 0, 1]
        angular_acc = [0, 0, 0]

        eta_dot_expected = pos_dot_expected + quat_dot_expected
        nu_dot_expected = linear_acc + angular_acc
        x_dot_expected = np.array(eta_dot_expected + nu_dot_expected)
        orientation = degrees_to_quat_rotation(roll, pitch, yaw).tolist()
        eta = position + orientation
        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array(tau, dtype=np.float64)

        x_dot = diagonal_slow_without_g(state, thrust, parameters)
        eta_dot = x_dot[0:7]
        nu_dot = x_dot[7:13]

        for a, b in zip(x_dot, x_dot_expected):
            self.assertAlmostEqual(a, b)

    def test_restoring_rotated(self):
        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [0]
        B = [10]
        COG = [0, 0, 0]
        COB = [1, 0, 0]

        # input
        position = [0, 0, 0]
        roll, pitch, yaw = [90, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        tau = [0, 0, 0, 0, 0, 0]

        # expected
        eta_dot = [0, 0, 0, 0, 0, 0, 0]
        linear_acc = [0, -1, 0]
        angular_acc = [0, 0, -1]

        x_dot_expected = np.array(eta_dot + linear_acc + angular_acc)
        orientation = degrees_to_quat_rotation(roll, pitch, yaw).tolist()
        eta = position + orientation
        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array(tau, dtype=np.float64)
        x_dot = diagonal_slow_without_g(state, thrust, parameters)

        for a, b in zip(x_dot, x_dot_expected):
            self.assertAlmostEqual(a, b)

    def test_non_invertible_M(self):
        M = [10, 10, 0, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [0]
        B = [10]
        COG = [0, 0, 0]
        COB = [1, 0, 0]

        # input
        position = [0, 0, 0]
        roll, pitch, yaw = [90, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        tau = [0, 0, 0, 0, 0, 0]

        orientation = degrees_to_quat_rotation(roll, pitch, yaw).tolist()
        eta = position + orientation
        state = np.array(eta + nu, dtype=np.float64)
        parameters = np.array(M + D + W + B + COG + COB, dtype=np.float64)
        thrust = np.array(tau, dtype=np.float64)
        x_dot = diagonal_slow_without_g(state, thrust, parameters)

        self.assertTrue(np.isnan(x_dot).all())


if __name__ == "__main__":
    unittest.main()
