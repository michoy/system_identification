import copy
import cProfile
import io
import pstats
from pathlib import Path
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy import linalg
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_preprocessing import recreate_sampling_times
from auv_models import auv_1DOF_simplified, auv_6DOF_simplified


def predict(state_space_equation: Callable[
    [np.ndarray, np.ndarray, np.ndarray], float], initial_state: np.ndarray,
            inputs: np.ndarray, step_length: float,
            parameters: list) -> np.ndarray:

    x = copy.deepcopy(initial_state)
    x_dot = np.zeros_like(x)
    states = np.zeros((len(inputs), len(initial_state)))
    i = 0

    for u in inputs:

        # save current state
        states[i, :] = x
        i += 1

        # calculate change in state
        x_dot[:] = state_space_equation(x, u, parameters)

        # integrate state change
        x += x_dot * step_length

    return states


def build_objective_function(state_space_equation,
                             states_true: pd.DataFrame,
                             inputs_true: pd.DataFrame,
                             error_metric=mean_squared_error,
                             weighting=None) -> Callable[[list], float]:

    step_length = inputs_true.index[1] - inputs_true.index[0]
    states = states_true.to_numpy()
    inputs = inputs_true.to_numpy()

    def objective_function(parameters: np.ndarray) -> float:

        initial_state = states_true.iloc[0].to_numpy()
        states_predicted = predict(state_space_equation, initial_state, inputs,
                                   step_length, parameters)

        return error_metric(states, states_predicted, multioutput=weighting)

    return objective_function


def optimize_parameters(state_space_equation,
                        states: pd.DataFrame,
                        inputs: pd.DataFrame,
                        initial_guess: np.ndarray,
                        optimization_method='nelder-mead',
                        tolerance=None,
                        options=None,
                        error_metric=mean_squared_error,
                        weighting=None):

    objective_function = build_objective_function(state_space_equation,
                                                  states,
                                                  inputs,
                                                  error_metric=error_metric,
                                                  weighting=weighting)

    res = minimize(objective_function,
                   initial_guess,
                   method=optimization_method,
                   tol=tolerance,
                   options=options)

    return res


def symmetric_mean_absolute_percentage_error(y_true: List[float],
                                             y_pred: List[float]) -> float:
    '''
    sMAPE error metric
    '''
    samples: np.ndarray = np.array(y_true)
    predictions: np.ndarray = np.array(y_pred)
    return np.mean(
        np.abs(samples - predictions) /
        (np.abs(samples) + np.abs(predictions))) * 200


if __name__ == "__main__":

    create_heatmap = False
    do_optimize = False
    find_all_errors = False
    new_version = True
    profile = False

    RUN_NAME = '1D-run-1'
    DATA_PATH = Path('data/processed/%s.pickle' % RUN_NAME)
    PLOT_DIR = Path('plots/1d-param-estimates')
    DEBUG_PLOTS = Path('plots/debug')
    ERROR_METRIC = 'MSE'

    data = pd.read_pickle(DATA_PATH)

    if new_version:

        if profile:
            pr = cProfile.Profile()
            pr.enable()

        options = {
            'xatol': 0.1,
        }

        res = optimize_parameters(auv_1DOF_simplified,
                                  data[['position.x', 'linear.x']],
                                  data[['force.x']],
                                  optimization_method='nelder-mead',
                                  tolerance=None,
                                  options=options,
                                  initial_guess=np.array([60, 22]),
                                  error_metric=mean_squared_error)
        print(res)

        # inputs = data[['force.x']].to_numpy(dtype='float', na_value=np.nan)
        # states = data[['position.x', 'linear.x']].to_numpy(dtype='float', na_value=np.nan)
        # step_length = 0.1
        # mass = 50
        # damp = 30

        # states_predicted = simple_auv.predict(states[0], inputs, step_length, np.array([mass, damp]))
        # df = pd.DataFrame({
        #     'x_true': states[:,0],
        #     'x_pred': states_predicted[:,0],
        #     'u_true': states[:,1],
        #     'u_pred': states_predicted[:,1],
        #     'thrust': inputs[:,0]
        # })

        # error = mean_squared_error(states, states_predicted)

        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        # sns.set_style('white')
        # fig, axes = plt.subplots(2, 1, sharex=True)

        # sns.lineplot(ax=axes[0], data=df[['x_pred', 'x_true']])
        # sns.lineplot(ax=axes[1], data=df[['u_pred', 'u_true']])

        # plt.savefig(DEBUG_PLOTS.joinpath('new_version_prediction.png'))

        # print(error)

    if create_heatmap:
        storage = {'mass': [], 'damp': [], 'error': []}

        for mass in np.arange(45, 52, 0.5):
            for damp in np.arange(19, 21, 0.1):
                inputs = data[['force.x']].to_numpy(dtype='float',
                                                    na_value=np.nan)
                states = data[['position.x',
                               'linear.x']].to_numpy(dtype='float',
                                                     na_value=np.nan)

                step_length = 0.1
                states_predicted = predict(auv_1DOF_simplified,
                                           states[0], inputs, step_length,
                                           np.array([mass, damp]))

                error = mean_absolute_error(states,
                                            states_predicted,
                                            multioutput=[1.0, 0.0])

                storage['mass'].append(mass)
                storage['damp'].append(damp)
                storage['error'].append(error)

        heatmap_df = pd.DataFrame(storage)

        sns.set()
        sns.heatmap(heatmap_df.pivot('mass', 'damp', 'error'),
                    cmap='rocket_r',
                    linewidths=0)
        plt.savefig(DEBUG_PLOTS.joinpath('heatmap-abs-only_pos-noscale.png'))

        # if do_optimize:
        #     mass_guess = 40
        #     damp_guess = 22
        #     x0 = np.array([mass_guess, damp_guess])
        #     res = minimize(objective_function, x0, method='nelder-mead')
        #     mass_optimal = res.x[0]
        #     damp_optimal = res.x[1]
        #     res_error = res.fun

        #     print('Optimal parameters with %s are mass=%f and damp=%f and give an error of %f' % (ERROR_METRIC, mass_optimal, damp_optimal, res_error))
        #     objective_function(res.x, plot=True)

        # if find_all_errors:

        mass_guess = 50
        damp_guess = 20
        x0 = np.array([mass_guess, damp_guess])

        objective_mse = build_objective_function(error_metric='MSE')
        objective_mae = build_objective_function(error_metric='MAE')
        objective_smape = build_objective_function(error_metric='SMAPE')

        objective_functions = [objective_mse, objective_mae, objective_smape]
        error_metrics = ['MSE', 'MAP', 'SMAPE']

        results: List[List[Any]] = [['', 'm', 'd', 'MSE', 'MAE', 'SMAPE']]

        for objective_function in objective_functions:

            res = minimize(objective_function, x0, method='nelder-mead')

            error_mse = objective_mse(res.x)
            error_mae = objective_mae(res.x)
            error_smape = objective_smape(res.x)

            results.append([
                error_metrics.pop(0), res.x[0], res.x[1], error_mse, error_mae,
                error_smape
            ])

        df = pd.DataFrame(results)

        table_dir = Path('tables').joinpath(RUN_NAME)

        with table_dir.joinpath('all-errors.tex').open(mode='w') as f:
            print(df.to_latex(
                label='tab:all-errors',
                caption='Optimal parameters and their evaluations',
                index=False),
                  file=f)
