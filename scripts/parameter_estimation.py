import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, List, Union
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit, prange
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.display import Display, MultiObjectiveDisplay

from auv_models import diagonal_slow
from data_generation import synthesize_dataset
from helper import (
    DFKeys,
    ETA_DOFS,
    NU_DOFS,
    PREPROCESSED_DIR,
    SYNTHETIC_DIR,
    TAU_DOFS,
    load_data,
    mean_squared_error,
    normalize,
    numpy_from_df,
    profile,
    is_poistive_def,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class UUVParameterProblem(Problem):
    def __init__(
        self,
        state_space_equation: Callable,
        tau: np.ndarray,
        y_measured: np.ndarray,
        x0: np.ndarray,
        n_var: float,
        n_obj: float,
        n_constr: float,
        dtype=np.float64,
    ):

        self.state_space_equation = state_space_equation
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

        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=x_lower, xu=x_upper
        )

    def _evaluate(self, designs, out, *args, **kwargs):
        out["F"] = compiled_evaluation(
            designs=designs,
            state_space_equation=self.state_space_equation,
            x0=self.x0,
            inputs=self.tau,
            y_measured=self.y_measured,
        )
        nan_count = 0
        for objectives in out["F"]:
            if np.isnan(objectives).any():
                nan_count += 1
        log.info("objectives with nan: " + str(nan_count))


class MyDisplay(MultiObjectiveDisplay):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("f_mean", np.mean(algorithm.pop.get("F")))
        self.output.append("f_min", algorithm.pop.get("F").min())
        self.output.append("f_max", algorithm.pop.get("F").max())


@njit(parallel=True)
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
    for i in prange(len(designs)):
        y_predicted = predict(
            state_space_equation=state_space_equation,
            initial_state=x0,
            inputs=inputs,
            step_length=step_length,
            parameters=designs[i],
        )
        if np.isnan(y_predicted).any():
            f[i] = np.full(13, np.nan)
        else:
            nu_predicted = y_predicted[7:13]
            nu_measured = y_measured[7:13]
            f[i] = mean_squared_error(nu_measured, nu_predicted)

    return f


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
        x_dot = state_space_equation(x, u, parameters) * step_length

        # return none if state space equation reached an illegal state
        if np.isnan(x_dot).any():
            states[:] = np.nan
            return states

        x += x_dot

        # normalize quaternions
        x[3:7] = normalize(x[3:7])

        # save current state
        i += 1
        states[i, :] = x

    return states


def calculate_pareto_front(
    state_space_equation,
    tau,
    y_measured,
    x0,
    n_var=20,
    n_obj=6,
    n_constr=0,
    pop_size=100,
    max_gen: Union[float, None] = None,
    save_dir: Union[Path, None] = None,
    verbose=True,
    save_history=True,
    display=MyDisplay(),
):

    # calculate pareto frontier
    problem = UUVParameterProblem(
        state_space_equation,
        tau,
        y_measured,
        x0,
        n_var=n_var,
        n_obj=n_obj,
        n_constr=n_constr,
    )
    algorithm = NSGA2(pop_size=pop_size)
    if max_gen:
        stop_criteria = ("n_gen", max_gen)
        res = minimize(
            problem,
            algorithm,
            stop_criteria,
            verbose=verbose,
            display=display,
            save_history=save_history,
        )
    else:
        res = minimize(
            problem,
            algorithm,
            verbose=verbose,
            display=display,
            save_history=save_history,
        )

    # save results
    if save_dir:
        Path.mkdir(save_dir, parents=True, exist_ok=True)
        save_path = save_dir / "trial"
        with save_path.with_suffix(".obj").open("wb") as file:
            pickle.dump(res, file)

    return res, problem.pareto_front()


if __name__ == "__main__":

    name = "sway-1"

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh = logging.FileHandler("logs/%s.log" % name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    M = [30, 60, 60, 10, 30, 30]
    D = [30, 60, 60, 10, 30, 30]
    W = [25]
    B = [24]
    COG = [0, 0, 0]
    COB = [0, 0, 0]
    params = np.array(M + D + W + B + COG + COB, dtype=np.float64)

    input_path = PREPROCESSED_DIR / (name + ".csv")
    save_dir = Path("results/synthetic_data/" + name)

    log.info("Synthesizing data..")
    df = synthesize_dataset(params=params, input_path=input_path)
    x0, tau, y_measured, time = numpy_from_df(df)

    log.info("calculating pareto front..")
    res, pareto_front = calculate_pareto_front(
        diagonal_slow, tau, y_measured, x0, max_gen=100, save_dir=save_dir
    )
    log.info("pareto front: " + str(pareto_front))

    found_params = False
    for design_point in res.X:
        if np.allclose(design_point, params):
            found_params = True
            break
    log.info("found parameters: " + str(found_params))
