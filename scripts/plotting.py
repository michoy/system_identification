from pathlib import Path
from typing import Callable
from matplotlib.colors import BoundaryNorm, Normalize, LogNorm
from matplotlib.ticker import LogLocator, FixedLocator, FixedFormatter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auv_models import linear_surge
from helper import (
    ETA_DOFS,
    NU_DOFS,
    DFKeys,
    PARAM_EST_DIR,
    PARAM_EST_SIM_DIR,
    normalize,
)
from parameter_estimation import calculate_pareto_front, predict, optimize_linear_surge


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


def plot_pareto_front():
    READ_DIR = PARAM_EST_DIR / "linear_surge_model" / "m30_d30"
    resX = pd.read_csv(READ_DIR / "resX.csv", header=None).to_numpy()
    resF = pd.read_csv(READ_DIR / "resF.csv", header=None).to_numpy()

    plt.scatter(resX[0], resX[1], c=resF[0] + resF[1])
    plt.title("Final population")
    plt.xlabel("Mass (kg)")
    plt.ylabel("Damping")
    plt.show()


def plot_objective_function():
    read_dir = Path("results/objective_function/linear_surge/long_tau")
    M = pd.read_csv(read_dir / "M.csv", header=None).to_numpy()
    D = pd.read_csv(read_dir / "D.csv", header=None).to_numpy()
    F = pd.read_csv(read_dir / "f.csv", header=None).to_numpy()
    resX = pd.read_csv(read_dir / "resX.csv", header=None).to_numpy()
    resF = pd.read_csv(read_dir / "resF.csv", header=None).to_numpy()

    X, Y = np.meshgrid(M, D)
    levels = [0.01]
    for i in range(25):
        levels.append(round(1.5 * levels[i], 3))
    cf = plt.contourf(
        X,
        Y,
        F,
        locator=FixedLocator(levels),
        cmap=plt.get_cmap("Oranges"),
        norm=LogNorm(vmin=0.01, vmax=100, clip=True),
        extend="both",
    )
    plt.colorbar(cf, format="%.3f")
    plt.xlabel("Mass (kg)")
    plt.ylabel("Damping")

    plt.scatter(resX[:, 0], resX[:, 1])

    # plt.savefig(read_dir / "objective_long_tau.pdf")
    plt.show()


if __name__ == "__main__":
    # optimize_linear_surge()
    plot_objective_function()
