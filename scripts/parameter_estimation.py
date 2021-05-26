import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit, vectorize
from pandas.core.frame import DataFrame
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize

from auv_models import diagonal_slow
from helper import ETA_DOFS, NU_DOFS, TAU_DOFS, mean_squared_error, normalize, profile


class SlowDiagonalModel(Problem):
    def __init__(self, tau, y_true, x_0, parallelization=None):

        self.tau = tau
        self.y_true = y_true
        self.x_0 = x_0

        lower_bounds = np.empty(19)
        upper_bounds = np.empty(19)

        # bounds on mass M = M_RB + M_A
        lower_bounds[0:6] = 0
        upper_bounds[0:6] = 100

        # bounds on linear damping D
        lower_bounds[6:12] = 0
        upper_bounds[6:12] = 100

        # bounds on weight W
        lower_bounds[12] = 20
        upper_bounds[12] = 35

        # bounds on buoyancy B
        lower_bounds[13] = 20
        upper_bounds[13] = 30

        # bounds on CG and CB
        lower_bounds[14:20] = -0.2
        upper_bounds[14:20] = 0.2

        super().__init__(
            n_var=19,
            n_obj=1,
            n_constr=0,
            xl=lower_bounds,
            xu=upper_bounds,
            elementwise_evaluation=True,
            parallelization=parallelization,
        )

    def _evaluate(self, x, out, *args, **kwargs):

        # calculate y_pred
        y_pred = predict(
            state_space_equation=diagonal_slow,
            initial_state=self.x_0,
            inputs=self.tau,
            step_length=0.1,
            parameters=x,
        )

        # calculate error between y_true and y_pred
        out["F"] = mean_squared_error(self.y_true, y_pred)


@njit
def predict(
    state_space_equation: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    inputs: np.ndarray,
    step_length: float,
    parameters: np.ndarray,
) -> np.ndarray:

    states = np.empty((len(inputs), len(initial_state)))
    x = initial_state.copy()
    i = 0
    states[i, :] = x

    for u in inputs[0 : len(inputs) - 1]:

        # integrate state change
        x += state_space_equation(x, u, parameters) * step_length

        # normalize quaternions
        x[3:7] = normalize(x[3:7])

        # save current state
        i += 1
        states[i, :] = x

    return states


@profile
def estimate_parallelized(df_path: Path, save_path: Path):

    # data as numpy arrays
    df = pd.read_csv(df_path).head(100)
    tau = df[TAU_DOFS].to_numpy()
    y_true = df[ETA_DOFS + NU_DOFS].to_numpy()
    x_0 = df[ETA_DOFS + NU_DOFS].loc[0].to_numpy()

    # thread pool for parallelization
    number_of_threads = 4
    with ThreadPool(number_of_threads) as pool:

        problem = SlowDiagonalModel(
            tau, y_true, x_0, parallelization=("starmap", pool.starmap)
        )
        algorithm = NSGA2(pop_size=100)
        res = pymoo_minimize(problem, algorithm, ("n_gen", 100))

    with save_path.open("wb") as file:
        pickle.dump(res, file)

    print(res.problem.pareto_front())


@profile
def estimate(df_path: Path, save_path: Path):

    # data as numpy arrays
    df = pd.read_csv(df_path).head(100)
    tau = df[TAU_DOFS].to_numpy()
    y_true = df[ETA_DOFS + NU_DOFS].to_numpy()
    x_0 = df[ETA_DOFS + NU_DOFS].loc[0].to_numpy()

    problem = SlowDiagonalModel(tau, y_true, x_0)
    algorithm = NSGA2(pop_size=100)
    res = pymoo_minimize(problem, algorithm, ("n_gen", 100))

    with save_path.open("wb") as file:
        pickle.dump(res, file)

    print(res.problem.pareto_front())


if __name__ == "__main__":
    estimate(
        df_path=Path("data/preprocessed/surge-1.csv"),
        save_path=Path(
            "results/parameter_estimation/slow_diagonal_model/attempt-1.obj"
        ),
    )
