import logging
import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, List, Union
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit, prange
from numpy import float64, ndarray
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_decision_making, get_sampling, get_termination
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

from auv_models import (
    diagonal_slow,
    diagonal_slow_without_g,
    linear_surge,
    nonlinear_surge,
)
from helper import (
    ETA_DOFS,
    NU_DOFS,
    PREPROCESSED_DIR,
    SYNTHETIC_DIR,
    TAU_DOFS,
    DFKeys,
    X_better_ramp,
    X_many_sin,
    X_ramp,
    X_sin,
    X_single_ramp,
    X_single_sin,
    X_sinousoidals,
    X_triple_ramp,
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
from plotting import plot_surge_timeseries, plot_objective_function

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


@njit
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
    x_dot_limit = 10  # m/s or rad/s

    for u in inputs[0 : len(inputs) - 1]:

        # integrate state change
        x_dot = state_space_equation(x, u, parameters) * step_length

        # return none if state space equation reached an illegal state
        if (
            np.isnan(x_dot).any()
            or np.isinf(x_dot).any()
            or (np.abs(x_dot) > x_dot_limit).any()
        ):
            states[:] = np.nan
            return states

        x += x_dot

        # normalize quaternions
        x = normalizer(x, normalize_quaternions)

        # save current state
        i += 1
        states[i, :] = x

    return states


def minimal_calculate_pf(eom, tau, y, xl, xu, norm_quats=False, pop_size=None):
    if not pop_size:
        pop_size = 10 ** len(xl)

    return calculate_pareto_front(
        eom,
        tau,
        y,
        x0=y[0],
        xl=np.array(xl, dtype=float64),
        xu=np.array(xu, dtype=float64),
        n_var=len(xl),
        n_obj=len(y[0]),
        normalize_quaternions=norm_quats,
        n_constr=0,
        pop_size=pop_size,
        n_max_gen=1000,
        verbose=True,
        display=MyDisplay(),
    )


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
    n_max_gen=1000,
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
        # n_offsprings=int(pop_size / 2),
        crossover=SimulatedBinaryCrossover(eta=15, prob=0.9),
        mutation=PolynomialMutation(prob=None, eta=15),
    )
    termination_default = MultiObjectiveDefaultTermination(n_max_gen=n_max_gen)
    termination_design_space = DesignSpaceToleranceTermination(n_max_gen=n_max_gen)
    termination_generations = get_termination("n_gen", 100)

    res = minimize(
        problem,
        algorithm,
        termination_default,
        verbose=verbose,
        display=display,
    )

    return res


def save_res(res, eom, tau, y, dir: Path, theta=None):
    if res:
        pd.DataFrame(res.X).to_csv(dir / "resX.csv", header=False, index=False)
        pd.DataFrame(res.F).to_csv(dir / "resF.csv", header=False, index=False)
        pd.DataFrame(tau).to_csv(dir / "tau.csv", header=False, index=False)
        pd.DataFrame(y).to_csv(dir / "y.csv", header=False, index=False)
        if theta:
            pd.DataFrame(theta).to_csv(dir / "theta.csv", header=False, index=False)

        try:
            I = get_knee_point(res.F)
            knee_points = res.X[I]
            pd.DataFrame(knee_points).to_csv(
                dir / "knee_points.csv", header=False, index=False
            )
            for i, kp in enumerate(knee_points):
                y_hat = predict(
                    state_space_equation=eom,
                    initial_state=y[0],
                    inputs=tau,
                    parameters=kp,
                    normalize_quaternions=False,
                )
                pd.DataFrame(y_hat).to_csv(
                    dir / ("y_hat-kp%i.csv" % i), header=False, index=False
                )
        except:
            print("could not compute knee points. Using first from resX instead")
            y_hat = predict(
                state_space_equation=eom,
                initial_state=y[0],
                inputs=tau,
                parameters=res.X[0],
                normalize_quaternions=False,
            )
            pd.DataFrame(y_hat).to_csv(
                dir / "y_hat-resX.csv", header=False, index=False
            )
    else:
        print("did not optimize %s" % dir.name)


def get_knee_point(F: ndarray) -> ndarray:
    dm = get_decision_making("high-tradeoff")
    return dm.do(F)


def samples_for_linear_surge_model(tau, theta, save_dir):

    tau = np.copy(tau)
    theta = np.copy(theta)
    x0 = np.array([0, 0], dtype=np.float64)

    M = np.linspace(10, 120, 100, dtype=np.float64)
    D = np.linspace(10, 30, 50, dtype=np.float64)

    y = predict(
        state_space_equation=linear_surge,
        initial_state=x0,
        inputs=tau,
        parameters=theta,
        normalize_quaternions=False,
    )
    f = np.empty((len(D), len(M)))
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

    pd.DataFrame(f).to_csv(save_dir / "f.csv", header=False, index=False)
    pd.DataFrame(D).to_csv(save_dir / "D.csv", header=False, index=False)
    pd.DataFrame(M).to_csv(save_dir / "M.csv", header=False, index=False)


def optimize_linear_surge(tau, theta, save_dir):

    tau = np.copy(tau)
    theta = np.copy(theta)
    x0 = np.array([0, 0], dtype=np.float64)

    y = predict(
        state_space_equation=linear_surge,
        initial_state=x0,
        inputs=tau,
        parameters=theta,
        normalize_quaternions=False,
    )

    xl = np.array([10, 1], dtype=float64)
    xu = np.array([120, 20], dtype=float64)
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
        pop_size=200,
    )

    if res:
        pd.DataFrame(res.X).to_csv(save_dir / "resX.csv", header=False, index=False)
        pd.DataFrame(res.F).to_csv(save_dir / "resF.csv", header=False, index=False)
        pd.DataFrame(theta).to_csv(save_dir / "theta.csv", header=False, index=False)
        pd.DataFrame(tau).to_csv(save_dir / "tau.csv", header=False, index=False)
        pd.DataFrame(y).to_csv(save_dir / "y.csv", header=False, index=False)

    try:
        I = get_knee_point(res.F)
        knee_points = res.X[I]
        pd.DataFrame(knee_points).to_csv(
            save_dir / "knee_points.csv", header=False, index=False
        )
        if len(knee_points) == 1:
            kp = knee_points[0]
            y_hat = predict(
                state_space_equation=linear_surge,
                initial_state=x0,
                inputs=tau,
                parameters=kp,
                normalize_quaternions=False,
            )
            pd.DataFrame(y_hat).to_csv(
                save_dir / "y_hat.csv", header=False, index=False
            )
    except:
        print("could not compute knee points")


def optimize(
    EOM,
    tau: ndarray,
    theta: ndarray,
    initial_states: list,
    xl: list,
    xu: list,
    dir: Path,
    normalize_quaternions=False,
):

    tau = np.copy(tau)
    theta = np.copy(theta)
    x0 = np.array(initial_states, dtype=float64)

    y = predict(
        state_space_equation=EOM,
        initial_state=x0,
        inputs=tau,
        parameters=theta,
        normalize_quaternions=False,
    )

    res = calculate_pareto_front(
        EOM,
        tau,
        y,
        x0,
        xl=np.array(xl, dtype=float64),
        xu=np.array(xu, dtype=float64),
        n_var=len(xl),
        n_obj=len(x0),
        normalize_quaternions=normalize_quaternions,
        pop_size=10 ** len(xl),
    )

    if res:
        pd.DataFrame(res.X).to_csv(dir / "resX.csv", header=False, index=False)
        pd.DataFrame(res.F).to_csv(dir / "resF.csv", header=False, index=False)
        pd.DataFrame(theta).to_csv(dir / "theta.csv", header=False, index=False)
        pd.DataFrame(tau).to_csv(dir / "tau.csv", header=False, index=False)
        pd.DataFrame(y).to_csv(dir / "y.csv", header=False, index=False)

        try:
            I = get_knee_point(res.F)
            knee_points = res.X[I]
            pd.DataFrame(knee_points).to_csv(
                dir / "knee_points.csv", header=False, index=False
            )
            for i, kp in enumerate(knee_points):
                y_hat = predict(
                    state_space_equation=EOM,
                    initial_state=x0,
                    inputs=tau,
                    parameters=kp,
                    normalize_quaternions=False,
                )
                pd.DataFrame(y_hat).to_csv(
                    dir / ("y_hat-kp%i.csv" % i), header=False, index=False
                )
        except:
            print("could not compute knee points")
    else:
        print("Optimazation of %s failed" % dir.name)


def linear_surge_plot_prep_thetas():

    X = X_better_ramp()
    tau = np.array([[x, 0, 0, 0, 0, 0] for x in X], dtype=np.float64)

    thetas = [(10, 10), (10, 50), (100, 10), (100, 50), (300, 10), (300, 50)]
    # for theta in thetas:
    #     m = theta[0]
    #     d = theta[1]
    #     theta = np.array([m, d], dtype=np.float64)

    #     save_dir = Path("results/objective_function/linear_surge/m%i-d%i" % (m, d))
    #     Path.mkdir(save_dir, parents=True, exist_ok=True)

    #     samples_for_linear_surge_model(tau, theta, save_dir)
    #     optimize_linear_surge(tau, theta, save_dir)

    #     plot_objective_function(save_dir)
    m = 10
    d = 50
    theta = np.array([m, d], dtype=np.float64)

    save_dir = Path("results/objective_function/linear_surge/m%i-d%i" % (m, d))
    Path.mkdir(save_dir, parents=True, exist_ok=True)

    samples_for_linear_surge_model(tau, theta, save_dir)
    optimize_linear_surge(tau, theta, save_dir)

    plot_objective_function(save_dir)


def linear_surge_taus():

    tau_single_ramp = np.array(
        [[x, 0, 0, 0, 0, 0] for x in X_single_ramp], dtype=np.float64
    )
    tau_triple_ramp = np.array(
        [[x, 0, 0, 0, 0, 0] for x in X_triple_ramp], dtype=np.float64
    )
    tau_single_sin = np.array(
        [[x, 0, 0, 0, 0, 0] for x in X_single_sin],
        dtype=np.float64,
    )
    tau_many_sin = np.array(
        [[x, 0, 0, 0, 0, 0] for x in X_many_sin],
        dtype=np.float64,
    )

    taus = [
        tau_single_ramp,
        tau_single_sin,
        tau_many_sin,
        tau_triple_ramp,
    ]
    names = [
        "tau_single_ramp",
        "tau_single_sin",
        "tau_many_sin",
        "tau_triple_ramp",
    ]

    m = 60
    d_l = 20
    theta = np.array([m, d_l], dtype=np.float64)

    for tau, name in zip(taus, names):

        save_dir = Path("results/simulations/linear_surge/%s" % name)
        Path.mkdir(save_dir, parents=True, exist_ok=True)

        samples_for_linear_surge_model(tau, theta, save_dir)

        # optimize(
        #     EOM=linear_surge,
        #     tau=tau,
        #     theta=theta,
        #     initial_states=[0, 0],
        #     xl=[10, 1],
        #     xu=[120, 40],
        #     dir=save_dir,
        # )

        try:
            # plot_surge_timeseries(save_dir)
            plot_objective_function(save_dir)
        except:
            print("No timeseries plotted for %s" % name)


def nonlinear_surge_taus():

    tau_single_ramp = np.array(
        [[x, 0, 0, 0, 0, 0] for x in X_single_ramp], dtype=np.float64
    )
    tau_triple_ramp = np.array(
        [[x, 0, 0, 0, 0, 0] for x in X_triple_ramp], dtype=np.float64
    )
    tau_single_sin = np.array(
        [[x, 0, 0, 0, 0, 0] for x in X_single_sin],
        dtype=np.float64,
    )
    tau_many_sin = np.array(
        [[x, 0, 0, 0, 0, 0] for x in X_many_sin],
        dtype=np.float64,
    )

    taus = [
        tau_single_ramp,
        tau_single_sin,
        tau_many_sin,
        tau_triple_ramp,
    ]
    names = [
        "tau_single_ramp",
        "tau_single_sin",
        "tau_many_sin",
        "tau_triple_ramp",
    ]

    m = 60
    d_l = 20
    d_nl = 2
    theta = np.array([m, d_l, d_nl], dtype=np.float64)

    for tau, name in zip(taus, names):

        save_dir = Path("results/simulations/nonlinear_surge-run-b/%s" % name)
        Path.mkdir(save_dir, parents=True, exist_ok=True)

        optimize(
            EOM=nonlinear_surge,
            tau=tau,
            theta=theta,
            initial_states=[0, 0],
            xl=[10, 1, 0],
            xu=[120, 40, 10],
            dir=save_dir,
        )

        try:
            plot_surge_timeseries(save_dir)
        except:
            print("No timeseries plotted for %s" % name)


def linear_surge_single_plot_prep():
    tau = np.array([[x, 0, 0, 0, 0, 0] for x in X_better_ramp()], dtype=np.float64)

    m = 100
    d = 10
    theta = np.array([m, d], dtype=np.float64)

    save_dir = Path("results/objective_function/linear_surge/close_up")
    Path.mkdir(save_dir, parents=True, exist_ok=True)

    samples_for_linear_surge_model(tau, theta, save_dir)
    optimize_linear_surge(tau, theta, save_dir)

    plot_objective_function(save_dir)


def predict_computation_time():

    name, num = ["single", "ten", "hundred", "thousand"], [1, 10, 100, 1000]

    index = 3
    save_name = "jit-parallel-%s.csv" % name[index]

    savedir = Path("results/computation_time/5min")
    Path.mkdir(savedir, parents=True, exist_ok=True)
    savepath = savedir / save_name

    tau = np.array([[x, 0, 0, 0, 0, 0] for x in X_sin(duration=3000)], dtype=np.float64)
    theta = np.array([30, 30], dtype=np.float64)
    x0 = np.array([0, 0], dtype=np.float64)

    scoreboard = dict()
    # for name, num in zip(["single", "ten", "hundred"], [1, 100, 10000]):

    t1 = time.perf_counter()
    repeated_predicts(num[index], x0, tau, theta)
    t2 = time.perf_counter()

    scoreboard[name[index]] = t2 - t1

    pd.DataFrame(scoreboard, [savepath.stem]).to_csv(
        savepath, index=False, header=False
    )


def surge_test_identification():
    # load data
    run_path = Path("data/preprocessed/surge-1.csv")

    t_start, t_end = 410, 650
    df = pd.read_csv(run_path)
    df = df[df[DFKeys.TIME.value] > t_start]
    df = df[df[DFKeys.TIME.value] < t_end]

    y = df[[DFKeys.POSITION_X.value, DFKeys.SURGE.value]].to_numpy()
    X = df[DFKeys.FORCE_X.value].to_numpy()
    tau = np.array([[x, 0, 0, 0, 0, 0] for x in X], dtype=np.float64)

    y = y * np.array([-1, 1])  # beluga was rotated 180 deg

    # estimate
    res = minimal_calculate_pf(
        nonlinear_surge, tau, y, xl=[10, 0, 0], xu=[1000, 1000, 1000]
    )

    # save results
    save_dir = Path("results/parameter_estimation/surge-1/nonlinear_surge_long_run")
    Path.mkdir(save_dir, parents=True, exist_ok=True)
    save_res(res, nonlinear_surge, tau, y, save_dir)

    # plot results
    plot_surge_timeseries(save_dir)


@njit
def repeated_predicts(num: int, x0: ndarray, tau: ndarray, theta: ndarray):
    for _i in prange(num):
        predict(
            state_space_equation=linear_surge,
            initial_state=x0,
            inputs=tau,
            parameters=theta,
            normalize_quaternions=False,
        )


if __name__ == "__main__":
    surge_test_identification()
    # nonlinear_surge_taus()
    # linear_surge_taus()
