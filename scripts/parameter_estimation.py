import logging
import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit, prange
from numpy import float64, ndarray
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_decision_making, get_sampling
from pymoo.model.callback import Callback
from pymoo.model.problem import Problem
from pymoo.model.result import Result
from pymoo.operators.crossover.simulated_binary_crossover import (
    SimulatedBinaryCrossover,
)
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.util.display import Display, MultiObjectiveDisplay
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination

from auv_models import diagonal_slow, diagonal_slow_without_g, linear_surge
from helper import (
    ETA_DOFS,
    NU_DOFS,
    PREPROCESSED_DIR,
    SYNTHETIC_DIR,
    TAU_DOFS,
    DFKeys,
    is_poistive_def,
    load_data,
    load_tau,
    mean_absolute_error,
    mean_absolute_error_with_log,
    mean_squared_error,
    normalize,
    normalizer,
    numpy_from_df,
    profile,
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
        xl: Union[np.ndarray, None],
        xu: Union[np.ndarray, None],
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
        f[i] = objective_function(
            state_space_equation=state_space_equation,
            initial_state=x0,
            inputs=inputs,
            step_length=step_length,
            parameters=designs[i],
            normalize_quaternions=normalize_quaternions,
            y_measured=y_measured,
        )

    return f


@njit
def objective_function(
    state_space_equation: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    inputs: np.ndarray,
    parameters: np.ndarray,
    normalize_quaternions: bool,
    y_measured: np.ndarray,
    step_length: float = 0.1,
) -> np.ndarray:
    y_predicted = predict(
        state_space_equation=state_space_equation,
        initial_state=initial_state,
        inputs=inputs,
        step_length=step_length,
        parameters=parameters,
        normalize_quaternions=normalize_quaternions,
    )
    if np.isnan(y_predicted).any():
        return np.full(len(initial_state), NAN_FILLER)
    else:
        return mean_absolute_error(y_measured, y_predicted)


@njit
def predict(
    state_space_equation: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    inputs: np.ndarray,
    parameters: np.ndarray,
    normalize_quaternions: bool,
    step_length: float = 0.1,
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
    xl: Union[np.ndarray, None],
    xu: Union[np.ndarray, None],
    n_var: int,
    n_obj: int,
    normalize_quaternions: bool,
    n_constr=0,
    pop_size=100,
    n_max_gen=100,
    verbose=True,
    display=MyDisplay(),
) -> Result:

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
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=int(pop_size / 2),
        crossover=SimulatedBinaryCrossover(eta=15, prob=0.9),
        mutation=PolynomialMutation(prob=None, eta=20),
    )
    termination_default = MultiObjectiveDefaultTermination(n_max_gen=n_max_gen)
    termination_design_space = DesignSpaceToleranceTermination(n_max_gen=n_max_gen)

    res = minimize(
        problem,
        algorithm,
        termination_design_space,
        verbose=verbose,
        display=display,
    )

    return res


def get_knee_point(F: ndarray) -> ndarray:
    dm = get_decision_making("high-tradeoff")
    return dm.do(F)


def samples_for_linear_surge_model():

    M = np.linspace(1, 70, 100, dtype=np.float64)
    D = np.linspace(1, 100, 100, dtype=np.float64)

    tau = []
    tau_break = [[0, 0, 0, 0, 0, 0] for _i in range(100)]
    tau_pos = [[10, 0, 0, 0, 0, 0] for _i in range(100)]
    tau_neg = [[-10, 0, 0, 0, 0, 0] for _i in range(100)]
    for i in range(1):
        tau += tau_pos + tau_break + tau_neg + tau_break + tau_pos + tau_break

    tau = np.array(tau, dtype=np.float64)
    x0 = np.array([0, 0], dtype=np.float64)

    m = 25
    d = 30

    theta = np.array([m, d], dtype=np.float64)

    y = predict(
        state_space_equation=linear_surge,
        initial_state=x0,
        inputs=tau,
        parameters=theta,
        normalize_quaternions=False,
    )
    f = np.empty((len(M), len(D)))
    for i, m in enumerate(M):
        for j, d in enumerate(D):
            theta[0] = m
            theta[1] = d
            f_val = objective_function(
                state_space_equation=linear_surge,
                initial_state=x0,
                inputs=tau,
                parameters=theta,
                normalize_quaternions=False,
                y_measured=y,
            )
            f[j, i] = sum(f_val)

    save_dir = Path("results/objective_function/linear_surge/long_tau")
    Path.mkdir(save_dir, parents=True, exist_ok=True)

    pd.DataFrame(f).to_csv(save_dir / "f.csv", header=False, index=False)
    pd.DataFrame(D).to_csv(save_dir / "D.csv", header=False, index=False)
    pd.DataFrame(M).to_csv(save_dir / "M.csv", header=False, index=False)


def optimize_linear_surge():
    tau = []
    tau_break = [[0, 0, 0, 0, 0, 0] for _i in range(100)]
    tau_pos = [[10, 0, 0, 0, 0, 0] for _i in range(100)]
    tau_neg = [[-10, 0, 0, 0, 0, 0] for _i in range(100)]
    for i in range(1):
        tau += tau_pos + tau_break + tau_neg + tau_break + tau_pos + tau_break

    tau = np.array(tau, dtype=np.float64)
    x0 = np.array([0, 0], dtype=np.float64)

    m = 25
    d = 30

    theta = np.array([m, d], dtype=np.float64)

    y = predict(
        state_space_equation=linear_surge,
        initial_state=x0,
        inputs=tau,
        parameters=theta,
        normalize_quaternions=False,
    )

    xl = np.array([1, 1], dtype=float64)
    xu = np.array([100, 100], dtype=float64)
    res = calculate_pareto_front(
        linear_surge,
        tau,
        y,
        x0,
        xl,
        xu,
        n_var=2,
        n_obj=2,
        normalize_quaternions=False,
        pop_size=1000,
        n_max_gen=1000,
    )

    if res:
        save_dir = Path("results/objective_function/linear_surge/long_tau")
        Path.mkdir(save_dir, parents=True, exist_ok=True)
        pd.DataFrame(res.X).to_csv(save_dir / "resX.csv", header=False, index=False)
        pd.DataFrame(res.F).to_csv(save_dir / "resF.csv", header=False, index=False)
        pd.DataFrame(theta).to_csv(save_dir / "theta.csv", header=False, index=False)
        pd.DataFrame(tau).to_csv(save_dir / "tau.csv", header=False, index=False)


if __name__ == "__main__":
    # samples_for_linear_surge_model()
    optimize_linear_surge()
