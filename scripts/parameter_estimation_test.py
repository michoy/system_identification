import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from auv_models import linear_surge, diagonal_slow_without_g, diagonal_slow
from helper import *
from parameter_estimation import *


class SyntheticDiagonalSlowWithoutGTests(unittest.TestCase):
    def test_diagonal_slow_synthetic_surge_short(self):

        M = [30, 60, 60, 10, 30, 30]
        D = [30, 60, 60, 10, 30, 30]
        W = [25]
        B = [24]
        COG = [0, 0, 0]
        COB = [0, 0, 0]
        theta = np.array(M + D + W + B + COG + COB, dtype=np.float64)

        tau = load_tau(SYNTHETIC_DIR / "random-1.csv")[0:500]

        eta_init = [0, 0, 0, 1, 0, 0, 0]
        nu_init = [0, 0, 0, 0, 0, 0]
        x0 = np.array(eta_init + nu_init, dtype=np.float64)

        y_measured = predict(
            diagonal_slow_without_g, x0, tau, theta, normalize_quaternions=True
        )

        x_lower = np.empty(20, dtype=np.float64)
        x_upper = np.empty(20, dtype=np.float64)

        x_lower[0:6] = 1  #  mass M = M_RB + M_A
        x_upper[0:6] = 100

        x_lower[6:12] = 0  # linear damping D
        x_upper[6:12] = 100

        x_lower[12] = 20  # weight W
        x_upper[12] = 35

        x_lower[13] = 20  # buoyancy B
        x_upper[13] = 30

        x_lower[14:20] = -0.2  # bounds on CG and CB
        x_upper[14:20] = 0.2

        res = calculate_pareto_front(
            diagonal_slow_without_g,
            tau,
            y_measured,
            x0,
            xl=x_lower,
            xu=x_upper,
            n_var=20,
            n_obj=13,
            normalize_quaternions=True,
            pop_size=70,
            n_max_gen=1000,
        )

        # save results
        save_dir = PARAM_EST_SIM_DIR / "diagonal_slow_noG_model" / "theta_1"
        Path.mkdir(save_dir, parents=True, exist_ok=True)
        pd.DataFrame(res.X).to_csv(save_dir / "resX.csv", header=False, index=False)
        pd.DataFrame(res.F).to_csv(save_dir / "resF.csv", header=False, index=False)

        found_params = False
        for design_point in res.X:
            if np.allclose(design_point, theta, atol=0.1):
                found_params = True
                break
        self.assertTrue(found_params)


class SyntheticDiagonalSlowTests(unittest.TestCase):
    def test_with_random_1_tau(self):
        M = [30, 60, 60, 10, 30, 30]
        D = [30, 60, 60, 10, 30, 30]
        eta_init = [0, 0, 0, 1, 0, 0, 0]
        nu_init = [0, 0, 0, 0, 0, 0]
        n_var = 12

        theta = np.array(M + D, dtype=np.float64)
        tau = load_tau(SYNTHETIC_DIR / "random-1.csv")
        x0 = np.array(eta_init + nu_init, dtype=np.float64)
        y_measured = predict(diagonal_slow, x0, tau, theta, normalize_quaternions=True)

        x_lower = np.empty(n_var, dtype=np.float64)
        x_upper = np.empty(n_var, dtype=np.float64)

        x_lower[0:6] = 1  #  mass M = M_RB + M_A
        x_upper[0:6] = 100

        x_lower[6:12] = 0  # linear damping D
        x_upper[6:12] = 100

        res = calculate_pareto_front(
            diagonal_slow,
            tau,
            y_measured,
            x0,
            xl=x_lower,
            xu=x_upper,
            n_var=n_var,
            n_obj=13,
            normalize_quaternions=True,
            pop_size=100,
            n_max_gen=100,
        )

        # save results
        save_dir = PARAM_EST_SIM_DIR / "diagonal_slow_model" / "theta_1"
        Path.mkdir(save_dir, parents=True, exist_ok=True)
        pd.DataFrame(res.X).to_csv(save_dir / "resX.csv", header=False, index=False)
        pd.DataFrame(res.F).to_csv(save_dir / "resF.csv", header=False, index=False)

        found_params = False
        for design_point in res.X:
            if np.allclose(design_point, theta, atol=0.1):
                found_params = True
                break
        self.assertTrue(found_params)

    def test_repeated_predicts(self):
        M = [30, 60, 60, 10, 30, 30]
        D = [30, 60, 60, 10, 30, 30]
        eta_init = [0, 0, 0, 1, 0, 0, 0]
        nu_init = [0, 0, 0, 0, 0, 0]
        n_var = 12

        theta = np.array(M + D, dtype=np.float64)
        tau = load_tau(SYNTHETIC_DIR / "random-1.csv")
        x0 = np.array(eta_init + nu_init, dtype=np.float64)

        y1 = predict(diagonal_slow, x0, tau, theta, normalize_quaternions=True)
        y2 = predict(diagonal_slow, x0, tau, theta, normalize_quaternions=True)
        for i in range(1000):
            y2 = predict(diagonal_slow, x0, tau, theta, normalize_quaternions=True)

        almost_equal = np.allclose(y1, y2)
        equal = (y1 == y2).all()

        self.assertTrue(almost_equal)
        self.assertTrue(equal)


class SyntheticOneDimensionalSimplifiedTests(unittest.TestCase):
    def test_m30_d30_s400(self):
        m = 30
        d = 30
        theta = np.array([m, d], dtype=np.float64)

        tau = []
        tau_break = [[0, 0, 0, 0, 0, 0] for _i in range(100)]
        tau_pos = [[10, 0, 0, 0, 0, 0] for _i in range(100)]
        tau_neg = [[-10, 0, 0, 0, 0, 0] for _i in range(100)]

        for i in range(1):
            tau += tau_pos + tau_break + tau_neg + tau_break

        tau = np.array(tau, dtype=np.float64)

        x0 = np.array([0, 0], dtype=np.float64)

        y_measured = predict(linear_surge, x0, tau, theta, False)

        x_lower = np.array([1, 1], dtype=np.float64)
        x_upper = np.array([100, 100], dtype=np.float64)

        res = calculate_pareto_front(
            linear_surge,
            tau,
            y_measured,
            x0,
            xl=x_lower,
            xu=x_upper,
            n_var=2,
            n_obj=2,
            normalize_quaternions=False,
        )

        # save results
        save_dir = PARAM_EST_SIM_DIR / "linear_surge_model" / ("m%i_d%i" % (m, d))
        Path.mkdir(save_dir, parents=True, exist_ok=True)
        pd.DataFrame(res.X).to_csv(save_dir / "resX.csv", header=False, index=False)
        pd.DataFrame(res.F).to_csv(save_dir / "resF.csv", header=False, index=False)

        found_params = False
        for design_point in res.X:
            if np.allclose(design_point, theta, atol=0.1):
                found_params = True
                break
        self.assertTrue(found_params)

    def test_if_resX_is_pareto(self):
        for i in range(5):
            m = 30
            d = 30
            theta = np.array([m, d], dtype=np.float64)

            tau = []
            tau_break = [[0, 0, 0, 0, 0, 0] for _i in range(100)]
            tau_pos = [[10, 0, 0, 0, 0, 0] for _i in range(100)]
            tau_neg = [[-10, 0, 0, 0, 0, 0] for _i in range(100)]

            for i in range(1):
                tau += tau_pos + tau_break + tau_neg + tau_break

            tau = np.array(tau, dtype=np.float64)

            x0 = np.array([0, 0], dtype=np.float64)

            y_measured = predict(linear_surge, x0, tau, theta, False)

            x_lower = np.array([1, 1], dtype=np.float64)
            x_upper = np.array([100, 100], dtype=np.float64)

            res = calculate_pareto_front(
                linear_surge,
                tau,
                y_measured,
                x0,
                xl=x_lower,
                xu=x_upper,
                n_var=2,
                n_obj=2,
                normalize_quaternions=False,
            )
            pareto_X, pareto_F = naive_pareto(res.X, res.F)

            self.assertTrue(np.allclose(res.X, pareto_X))
            self.assertTrue(np.allclose(res.F, pareto_F))

    def test_resX_and_pareto(self):

        dir = Path("results/objective_function/linear_surge")
        X = pd.read_csv(dir / "resX.csv", header=None).to_numpy()
        F = pd.read_csv(dir / "resF.csv", header=None).to_numpy()

        pareto_X, pareto_F = naive_pareto(X, F)

        self.assertTrue(np.allclose(X, pareto_X))
        self.assertTrue(np.allclose(F, pareto_F))

    def test_knee_points_contain_theta(self):
        dir = Path("results/objective_function/linear_surge/long_tau")
        X = pd.read_csv(dir / "resX.csv", header=None).to_numpy()
        F = pd.read_csv(dir / "resF.csv", header=None).to_numpy()
        theta = pd.read_csv(dir / "theta.csv", header=None).to_numpy()[:, 0]

        knee_points = get_knee_point(F)
        theta_knees = X[knee_points]

        found_theta = False
        for theta_knee in theta_knees:
            if np.allclose(theta, theta_knee, atol=0.1):
                found_theta = True
                break
        self.assertTrue(found_theta)


if __name__ == "__main__":
    unittest.main()
