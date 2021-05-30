import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from auv_models import auv_1DOF_simplified, diagonal_slow
from helper import *
from parameter_estimation import *


class SyntheticDiagonalSlowTests(unittest.TestCase):
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
            diagonal_slow, x0, tau, 0.1, theta, normalize_quaternions=True
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
            diagonal_slow,
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
        save_dir = ESTIMATED_DIR / "synthetic"
        Path.mkdir(save_dir, parents=True, exist_ok=True)
        pd.DataFrame(res.X).to_csv(save_dir / "diagonal_slow_resX.csv")
        pd.DataFrame(res.F).to_csv(save_dir / "diagonal_slow_resF.csv")

        found_params = False
        for design_point in res.X:
            if np.allclose(design_point, theta, atol=0.1):
                found_params = True
                break
        self.assertTrue(found_params)


class SyntheticOneDimensionalSimplifiedTests(unittest.TestCase):
    def test_only_surge(self):
        m = 30
        d = 30
        theta = np.array([m, d], dtype=np.float64)

        tau_1 = [[0, 0, 0, 0, 0, 0] for _i in range(10)]
        tau_2 = [[10, 0, 0, 0, 0, 0] for _i in range(100)]
        tau_3 = [[0, 0, 0, 0, 0, 0] for _i in range(10)]
        tau_4 = [[-10, 0, 0, 0, 0, 0] for _i in range(100)]
        tau_5 = [[0, 0, 0, 0, 0, 0] for _i in range(10)]
        tau = np.array(tau_1 + tau_2 + tau_3 + tau_4 + tau_5, dtype=np.float64)

        x0 = np.array([0, 0], dtype=np.float64)

        y_measured = predict(auv_1DOF_simplified, x0, tau, 0.1, theta, False)

        x_lower = np.array([10, 10], dtype=np.float64)
        x_upper = np.array([40, 40], dtype=np.float64)

        res = calculate_pareto_front(
            auv_1DOF_simplified,
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
        save_dir = ESTIMATED_DIR / "synthetic"
        Path.mkdir(save_dir, parents=True, exist_ok=True)
        pd.DataFrame(res.X).to_csv(save_dir / "only_surge_resX.csv")
        pd.DataFrame(res.F).to_csv(save_dir / "only_surge_resF.csv")

        found_params = False
        for design_point in res.X:
            if np.allclose(design_point, theta, atol=0.1):
                found_params = True
                break
        self.assertTrue(found_params)


if __name__ == "__main__":
    unittest.main()
