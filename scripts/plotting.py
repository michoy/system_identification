from os import read
from pathlib import Path
from typing import Callable
from matplotlib.colors import BoundaryNorm, Normalize, LogNorm
from matplotlib.ticker import LogLocator, FixedLocator, FixedFormatter
from brokenaxes import brokenaxes
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from auv_models import linear_surge
from helper import (
    ETA_DOFS,
    NU_DOFS,
    DFKeys,
    PARAM_EST_DIR,
    PARAM_EST_SIM_DIR,
    PREPROCESSED_DIR,
    X_better_ramp,
    X_ramp,
    X_sin,
    all_similar,
    normalize,
)


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


def plot_objective_function(read_dir):

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

    plt.scatter(resX[:, 0], resX[:, 1], c="tab:cyan")

    if Path.exists(read_dir / "knee_points.csv"):
        knee_points = pd.read_csv(read_dir / "knee_points.csv", header=None).to_numpy()
        plt.scatter(knee_points[:, 0], knee_points[:, 1], c="tab:blue")

    # plt.savefig(read_dir / ("heatmap-%s.pdf" % read_dir.name))
    plt.show()
    plt.close()


def plot_tau_surge():
    tau_sin = X_sin(30, duration=1000)
    tau_single_ramp = X_ramp(30)
    tau_tripple_ramp = X_better_ramp()

    plt.plot(tau_sin, label="tau_sin", linestyle="solid")
    plt.plot(tau_tripple_ramp, label="tau_triple_ramp", linestyle="dotted")
    plt.plot(tau_single_ramp, label="tau_single_ramp", linestyle="dashed")
    # plt.plot(tau_real, label="tau_real", c="orange")

    time_ds = [i for i in range(0, 1001, 200)]
    time_seconds = [int(i * 0.1) for i in range(0, 1001, 200)]
    plt.xticks(time_ds, time_seconds)
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust (N)")
    plt.legend(loc="upper left")

    savedir = Path("results/tau_plot")
    Path.mkdir(savedir, exist_ok=True, parents=True)
    # plt.show()
    plt.savefig(savedir / "ramp&sin.pdf")


def plot_computation_time():

    readdir = Path("results/computation_time")
    reps, nums = ["single", "ten", "hundred", "thousand"], [1, 10, 100, 1000]
    mins = ["5min", "30min"]

    no_jit = []
    for rep in reps:
        with open(readdir / "5min" / ("no-jit-%s.csv" % rep)) as f:
            no_jit.append(float(f.read()))

    jit = []
    for rep in reps:
        with open(readdir / "5min" / ("jit-%s.csv" % rep)) as f:
            jit.append(float(f.read()))

    jit_parallel = []
    for rep in reps:
        with open(readdir / "5min" / ("jit-parallel-%s.csv" % rep)) as f:
            jit_parallel.append(float(f.read()))

    labels = ["n=1", "n=10", "n=100", "n=1000"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig = plt.figure()
    sns.set_style("whitegrid")
    bax = brokenaxes(ylims=((0, 4.9), (30, 32)), hspace=0.1)
    for i in range(4):
        bax.bar(
            [i, i + 5, i + 10],
            [no_jit[i], jit[i], jit_parallel[i]],
            color=colors[i],
            label=labels[i],
        )

    # bax.grid(alpha=0.3)
    bax.legend()
    bax.set_ylabel("Time (s)")
    bax.set_xticks([2, 6, 10])
    bax.set_xticklabels(labels=["", "", "No jit", "", "Jit", "", "", "Jit parallel"])

    plt.savefig(readdir / "computation_time.pdf")
    # plt.show()


if __name__ == "__main__":
    # linear_surge_plot_prep()
    # savedir = Path("results/objective_function/linear_surge/close_up")
    # plot_objective_function(savedir)
    plot_computation_time()
