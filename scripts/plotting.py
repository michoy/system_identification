from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auv_models import auv_1DOF_simplified
from helper import ETA_DOFS, NU_DOFS, DFKeys, PARAM_EST_DIR
from parameter_estimation import calculate_pareto_front, predict


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


def plot_synthetic_data():
    NAME = "surge-1"

    SYNTHETIC_DIR = Path("data/synthetic")
    SAVE_DIR = Path("results/synthetic_data") / NAME
    if not Path.is_dir(SAVE_DIR):
        Path.mkdir(SAVE_DIR, parents=True)

    df_path = SYNTHETIC_DIR / (NAME + ".csv")

    df = pd.read_csv(df_path)
    timesteps = df[DFKeys.TIME.value].to_numpy()
    y_measured = df[ETA_DOFS + NU_DOFS].to_numpy()

    for i, dof in enumerate(ETA_DOFS + NU_DOFS):
        plt.plot(timesteps, y_measured[:, i])
        plt.title(dof)
        plt.savefig(SAVE_DIR / (dof + ".jpg"))
        plt.close()


def plot_objective_function():
    READ_DIR = PARAM_EST_DIR / "linear_surge_model" / "m30_d30"
    resX = pd.read_csv(READ_DIR / "resX.csv").to_numpy()
    resF = pd.read_csv(READ_DIR / "resF.csv").to_numpy()

    plt.scatter(resX[0], resX[1], c=resF[0] + resF[1])
    plt.title("Final population")
    plt.xlabel("Mass (kg)")
    plt.ylabel("Damping")
    plt.show()


if __name__ == "__main__":
    plot_objective_function()
