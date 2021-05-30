import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, List, Union
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.factory import get_sampling
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.model.callback import Callback
import seaborn as sns
from numba import njit, prange
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.display import Display, MultiObjectiveDisplay

from auv_models import diagonal_slow
from helper import (
    DFKeys,
    ETA_DOFS,
    NU_DOFS,
    PREPROCESSED_DIR,
    SYNTHETIC_DIR,
    TAU_DOFS,
    load_data,
    mean_absolute_error,
    mean_absolute_error_with_log,
    mean_squared_error,
    normalize,
    numpy_from_df,
    profile,
    is_poistive_def,
    normalizer,
)

NAN_FILLER = 100000.0


class UUVParameterProblem(Problem):
    def __init__(
        self,
        state_space_equation: Callable,
        tau: np.ndarray,
        y_measured: np.ndarray,
        x0: np.ndarray,
        n_var: int,
        n_obj: int,
        n_constr: int,
        xl: np.ndarray,
        xu: np.ndarray,
        normalize_quaternions: bool,
    ):

        self.state_space_equation = state_space_equation
        self.tau = tau
        self.y_measured = y_measured
        self.x0 = x0
        self.n_obj = n_obj
        self.normalize = normalize_quaternions

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, designs, out, *args, **kwargs):
        out["F"] = compiled_evaluation(
            designs=designs,
            state_space_equation=self.state_space_equation,
            x0=self.x0,
            inputs=self.tau,
            y_measured=self.y_measured,
            n_obj=self.n_obj,
            normalize_quaternions=self.normalize,
        )


class MyDisplay(MultiObjectiveDisplay):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("f_best_mean", np.mean(algorithm.pop.get("F"), axis=1).min())
        self.output.append(
            "f_worst_mean", np.mean(algorithm.pop.get("F"), axis=1).max()
        )
        self.output.append(
            "f_mean_mean", np.mean(np.mean(algorithm.pop.get("F"), axis=1))
        )


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []

    def notify(self, algorithm):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt[0].F)


@njit(parallel=True)
def compiled_evaluation(
    designs,
    state_space_equation,
    x0,
    inputs,
    y_measured,
    n_obj: int,
    normalize_quaternions: bool,
    step_length=0.1,
    dtypte=np.float64,
):
    f = np.empty((len(designs), n_obj), dtype=dtypte)
    for i in prange(len(designs)):
        y_predicted = predict(
            state_space_equation=state_space_equation,
            initial_state=x0,
            inputs=inputs,
            step_length=step_length,
            parameters=designs[i],
            normalize_quaternions=normalize_quaternions,
        )
        if np.isnan(y_predicted).any():
            f[i] = np.full(n_obj, NAN_FILLER)
        else:
            f[i] = mean_absolute_error(y_measured, y_predicted)

    return f


@njit
def predict(
    state_space_equation: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    inputs: np.ndarray,
    step_length: float,
    parameters: np.ndarray,
    normalize_quaternions: bool,
) -> np.ndarray:

    states = np.empty((len(inputs), len(initial_state)), dtype=np.float64)
    x = initial_state.copy()
    i = 0
    states[i, :] = x

    for u in inputs[0 : len(inputs) - 1]:

        # integrate state change
        x_dot = state_space_equation(x, u, parameters) * step_length

        # return none if state space equation reached an illegal state
        if np.isnan(x_dot).any() or np.isinf(x_dot).any():
            states[:] = np.nan
            return states

        x += x_dot

        # normalize quaternions
        x = normalizer(x, normalize_quaternions)

        # save current state
        i += 1
        states[i, :] = x

    return states


def calculate_pareto_front(
    state_space_equation,
    tau,
    y_measured,
    x0,
    xl: np.ndarray,
    xu: np.ndarray,
    n_var: int,
    n_obj: int,
    normalize_quaternions: bool,
    n_constr=0,
    pop_size=100,
    n_max_gen=100,
    verbose=True,
    save_history=False,
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
        xl=xl,
        xu=xu,
        normalize_quaternions=normalize_quaternions,
    )
    algorithm = NSGA2(pop_size=pop_size)
    termination = DesignSpaceToleranceTermination(
        tol=0.0025, n_last=20, n_max_gen=n_max_gen
    )

    res = minimize(
        problem,
        algorithm,
        termination,
        verbose=verbose,
        display=display,
        save_history=save_history,
        eliminate_duplicates=True,
    )

    return res
