import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from helper import DFKeys, ETA_DOFS, NU_DOFS
from parameter_estimation import predict


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
