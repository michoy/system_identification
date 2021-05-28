import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, List
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit, prange
from pandas.core.frame import DataFrame
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

from auv_models import diagonal_slow
from helper import (
    DFKeys,
    ETA_DOFS,
    NU_DOFS,
    TAU_DOFS,
    mean_squared_error,
    normalize,
    profile,
    is_poistive_def,
)


class SlowDiagonalModel(Problem):
    def __init__(self, tau, y_measured, x0, dtype=np.float64):

        self.tau = tau
        self.y_measured = y_measured
        self.x0 = x0

        x_lower = np.empty(20, dtype=dtype)
        x_upper = np.empty(20, dtype=dtype)

        x_lower[0:6] = 0  #  mass M = M_RB + M_A
        x_upper[0:6] = 100

        x_lower[6:12] = 0  # linear damping D
        x_upper[6:12] = 100

        x_lower[12] = 20  # weight W
        x_upper[12] = 35

        x_lower[13] = 20  # buoyancy B
        x_upper[13] = 30

        x_lower[14:20] = -0.2  # bounds on CG and CB
        x_upper[14:20] = 0.2

        super().__init__(n_var=20, n_obj=6, n_constr=1, xl=x_lower, xu=x_upper)

    def _evaluate(self, designs, out, *args, **kwargs):
        out["F"], out["G"] = compiled_evaluation(
            designs=designs,
            state_space_equation=diagonal_slow,
            x0=self.x0,
            inputs=self.tau,
            y_measured=self.y_measured,
        )


# @njit(parallel=True, fastmath=True)
def compiled_evaluation(
    designs,
    state_space_equation,
    x0,
    inputs,
    y_measured,
    step_length=0.1,
    dtypte=np.float64,
):
    f = np.empty((len(designs), 13), dtype=dtypte)
    g = np.empty(len(designs), dtype=dtypte)
    for i in prange(len(designs)):
        y_predicted = predict(
            state_space_equation=state_space_equation,
            initial_state=x0,
            inputs=inputs,
            step_length=step_length,
            parameters=designs[i],
        )

        nu_predicted = y_predicted[7:13]
        nu_measured = y_measured[7:13]
        f[i] = mean_squared_error(nu_measured, nu_predicted)

        M = np.diag(designs[i][0:6])
        if is_poistive_def(M):
            g[i] = 0
        else:
            g[i] = 1
    return f, g


# @njit(fastmath=True)
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
        x_dot = state_space_equation(x, u, parameters) * step_length

        # if (np.abs(x_dot) > 100000).any():
        #     print("x_dot above max for i: " + str(i))
        x += x_dot

        # normalize quaternions
        x[3:7] = normalize(x[3:7])

        # save current state
        i += 1
        states[i, :] = x

    return states


def plot_predict(
    state_space_equation: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    inputs: np.ndarray,
    parameters: np.ndarray,
    y_measured: np.ndarray,
    timesteps: np.ndarray,
):
    y_predicted = predict(state_space_equation, initial_state, inputs, 0.1, parameters)
    dofs = ETA_DOFS + NU_DOFS
    for dof, i in zip(dofs, range(len(dofs))):
        plt.plot(timesteps, y_predicted[:, i], label="predicted")
        plt.plot(timesteps, y_measured[:, i], label="measured")
        plt.title(dof)
        plt.legend()
        plt.savefig(
            "results/parameter_estimation/slow_diagonal_model/trials/%s.jpg" % dof
        )
        plt.close()


def estimate(tau, y_true, x0, save_path: Path):
    problem = SlowDiagonalModel(tau, y_true, x0)
    algorithm = NSGA2(pop_size=100)
    stop_criteria = ("n_gen", 1000)

    res = minimize(problem, algorithm)

    with save_path.with_suffix(".obj").open("wb") as file:
        pickle.dump(res, file)

    df = pd.DataFrame(res.F)
    df.to_csv(save_path.with_suffix(".csv"))

    print(res.F)
    print("Success: " + str(res.success))


if __name__ == "__main__":
    logging.basicConfig(
        filename="application.log",
        level=logging.WARNING,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    SURGE_PATH = Path("data/preprocessed/surge-1.csv")
    SYNTHETIC_DIR = Path("data/synthetic")
    SAVE_PATH = Path(
        "results/parameter_estimation/slow_diagonal_model/synthetic-random-1.obj"
    )
    DTYPE = np.float64

    df = pd.read_csv(SYNTHETIC_DIR / "random-1.csv").head(100)

    tau = np.ascontiguousarray(df[TAU_DOFS].to_numpy(), dtype=DTYPE)
    y_measured = np.ascontiguousarray(df[ETA_DOFS + NU_DOFS].to_numpy(), dtype=DTYPE)
    x0 = np.ascontiguousarray(df[ETA_DOFS + NU_DOFS].loc[0].to_numpy(), dtype=DTYPE)
    timesteps = np.ascontiguousarray(df[DFKeys.TIME.value].to_numpy(), dtype=DTYPE)

    M = [30, 60, 60, 10, 30, 30]
    D = [30, 60, 60, 10, 30, 30]
    W = [25]
    B = [24]
    COG = [0, 0, 0]
    COB = [0, 0, 0]
    params = np.array(M + D + W + B + COG + COB, dtype=DTYPE)

    # plot_predict(diagonal_slow, x0, tau, params, y_measured, timesteps)
    estimate(tau, y_measured, x0, SAVE_PATH)
