from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from auv_models import auv_1DOF_simplified, diagonal_slow
from helper import *
from parameter_estimation import *


class TestPrediction(unittest.TestCase):
    def test_diagonal_slow_synthetic_surge_short(self):
        x0, tau, y_measured, time = load_data(SYNTHETIC_DIR / "surge-1.csv")

        M = [30, 60, 60, 10, 30, 30]
        D = [30, 60, 60, 10, 30, 30]
        W = [25]
        B = [24]
        COG = [0, 0, 0]
        COB = [0, 0, 0]
        params = np.array(M + D + W + B + COG + COB)

        res, pareto_front = calculate_pareto_front(diagonal_slow, tau, y_measured, x0)

        found_params = False
        for design_point in res.X:
            if np.allclose(design_point, params):
                found_params = True
                break
        self.assertTrue(found_params)


if __name__ == "__main__":
    unittest.main()
