import unittest

import numpy as np
from pyquaternion.quaternion import Quaternion

from auv_models import auv_1DOF_simplified, diagonal_slow_without_g
import helper


class JTests(unittest.TestCase):
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


class RotationConversionTests(unittest.TestCase):
    def test_deg_to_quat_and_back(self):
        roll = 90
        pitch = 20
        yaw = 110

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        degrees_result = helper.quat_to_degrees(*q)
        degrees_expected = np.array([roll, pitch, yaw])

        for a, b in zip(degrees_result, degrees_expected):
            self.assertAlmostEqual(a, b, places=3)

    def test_deg_to_quat_roll(self):
        roll = 90
        pitch = 0
        yaw = 0
        q_expected = np.array([0.707, 0.707, 0, 0])

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        for a, b in zip(q, q_expected):
            self.assertAlmostEqual(a, b, places=3)

    def test_deg_to_quat_pitch(self):
        roll = 0
        pitch = 90
        yaw = 0
        q_expected = np.array([0.707, 0, 0.707, 0])

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        for a, b in zip(q, q_expected):
            self.assertAlmostEqual(a, b, places=3)

    def test_deg_to_quat_yaw(self):
        roll = 0
        pitch = 0
        yaw = 90
        q_expected = np.array([0.707, 0, 0, 0.707])

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        for a, b in zip(q, q_expected):
            self.assertAlmostEqual(a, b, places=3)


class LinearVelRotationTests(unittest.TestCase):
    def test_nu_transform_x_yaw(self):
        nu_body = np.array([2, 0, 0])
        roll, pitch, yaw = [0, 0, 90]
        nu_ned = np.array([0, 2, 0])

        orientation = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        R = helper.R(orientation)
        result = R @ nu_body

        for a, b in zip(result, nu_ned):
            self.assertAlmostEqual(a, b)

    def test_nu_transform_x_roll(self):
        nu_body = np.array([2, 0, 0])
        roll, pitch, yaw = [90, 0, 0]
        nu_ned = np.array([2, 0, 0])

        orientation = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        R = helper.R(orientation)
        result = R @ nu_body

        for a, b in zip(result, nu_ned):
            self.assertAlmostEqual(a, b)

    def test_nu_transform_x_pitch(self):
        nu_body = np.array([2, 0, 0])
        roll, pitch, yaw = [0, 90, 0]
        nu_ned = np.array([0, 0, -2])

        orientation = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        R = helper.R(orientation)
        result = R @ nu_body

        for a, b in zip(result, nu_ned):
            self.assertAlmostEqual(a, b)

    def test_nu_transform_y_roll(self):
        nu_body = np.array([0, 2, 0])
        roll, pitch, yaw = [90, 0, 0]
        nu_ned = np.array([0, 0, 2])

        orientation = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        R = helper.R(orientation)
        result = R @ nu_body

        for a, b in zip(result, nu_ned):
            self.assertAlmostEqual(a, b)

    def test_nu_transform_y_pitch(self):
        nu_body = np.array([0, 2, 0])
        roll, pitch, yaw = [0, 90, 0]
        nu_ned = np.array([0, 2, 0])

        orientation = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        R = helper.R(orientation)
        result = R @ nu_body

        for a, b in zip(result, nu_ned):
            self.assertAlmostEqual(a, b)

    def test_nu_transform_y_yaw(self):
        nu_body = np.array([0, 2, 0])
        roll, pitch, yaw = [0, 0, 90]
        nu_ned = np.array([-2, 0, 0])

        orientation = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        R = helper.R(orientation)
        result = R @ nu_body

        for a, b in zip(result, nu_ned):
            self.assertAlmostEqual(a, b)

    def test_unit_vector_rotation_x_yaw(self):
        roll = 0
        pitch = 0
        yaw = 90
        x_unit = [1, 0, 0]
        x_expected = [0, 1, 0]

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        R = helper.R(q)
        x_res = R @ x_unit

        for a, b in zip(x_res, x_expected):
            self.assertAlmostEqual(a, b, places=3)

    def test_unit_vector_rotation_y_roll(self):
        roll = 90
        pitch = 0
        yaw = 0
        x_body = [0, 1, 0]
        x_ned_expected = [0, 0, 1]

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        R = helper.R(q)
        x_ned = R @ x_body

        for a, b in zip(x_ned, x_ned_expected):
            self.assertAlmostEqual(a, b, places=3)

    def test_unit_vector_rotation_combined(self):
        roll = 90
        pitch = 0
        yaw = 0
        x_body = [0, 1, 1]
        x_ned_expected = [0, -1, 1]

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        R = helper.R(q)
        x_ned = R @ x_body

        for a, b in zip(x_ned, x_ned_expected):
            self.assertAlmostEqual(a, b, places=3)


class AngularRotationTests(unittest.TestCase):
    def test_x_roll(self):
        omega_body = np.array([0.2, 0, 0])
        roll, pitch, yaw = [90, 0, 0]
        omega_ned = np.array([0.2, 0, 0])

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        T = helper.T(q)
        q_dot_ned = T @ omega_body

        py_quat = Quaternion(*q)
        py_quat_dot = Quaternion(*q_dot_ned)
        omega_ned_res = (2 * py_quat_dot * py_quat.conjugate).elements[1:]

        for a, b in zip(omega_ned, omega_ned_res):
            self.assertAlmostEqual(a, b)

    def test_x_pitch(self):
        omega_body = np.array([0.2, 0, 0])
        roll, pitch, yaw = [0, 90, 0]
        omega_ned = np.array([0, 0, -0.2])

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        T = helper.T(q)
        q_dot_ned = T @ omega_body

        py_quat = Quaternion(*q)
        py_quat_dot = Quaternion(*q_dot_ned)
        omega_ned_res = (2 * py_quat_dot * py_quat.conjugate).elements[1:]

        for a, b in zip(omega_ned, omega_ned_res):
            self.assertAlmostEqual(a, b)

    def test_x_pitch_fast(self):
        omega_body = np.array([1.5, 0, 0])
        roll, pitch, yaw = [0, 90, 0]
        omega_ned = np.array([0, 0, -1.5])

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        T = helper.T(q)
        q_dot_ned = T @ omega_body

        py_quat = Quaternion(*q)
        py_quat_dot = Quaternion(*q_dot_ned)
        omega_ned_res = (2 * py_quat_dot * py_quat.conjugate).elements[1:]

        for a, b in zip(omega_ned, omega_ned_res):
            self.assertAlmostEqual(a, b)

    def test_x_yaw(self):
        omega_body = np.array([0.1, 0, 0])
        roll, pitch, yaw = [0, 0, 90]
        omega_ned = np.array([0, 0.1, 0])

        q = helper.degrees_to_quat_rotation(roll, pitch, yaw)
        T = helper.T(q)
        q_dot_ned = T @ omega_body

        py_quat = Quaternion(*q)
        py_quat_dot = Quaternion(*q_dot_ned)
        omega_ned_res = (2 * py_quat_dot * py_quat.conjugate).elements[1:]

        for a, b in zip(omega_ned, omega_ned_res):
            self.assertAlmostEqual(a, b)


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


class ParetoTests(unittest.TestCase):
    def test_simple_case(self):
        F = [
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.1, 2.0, 2.0],
            [2.0, 2.0, 2.2],
            [2, 2, 2],
        ]
        X = [
            [1, 2],
            [2, 1],
            [3, 3],
            [4, 4],
            [1, 1],
        ]
        pareto_X, pareto_F = helper.naive_pareto(np.array(X), np.array(F))

        expected_X = np.array(
            [
                [1, 2],
                [2, 1],
                [1, 1],
            ]
        )
        expected_F = np.array(
            [
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [2, 2, 2],
            ]
        )

        self.assertEqual(pareto_X.size, expected_X.size)
        self.assertEqual(pareto_F.size, expected_F.size)
        self.assertTrue(np.allclose(pareto_X, expected_X))
        self.assertTrue(np.allclose(pareto_F, expected_F))


if __name__ == "__main__":
    unittest.main()
