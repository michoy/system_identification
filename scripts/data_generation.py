from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count

from auv_models import diagonal_slow
from helper import DFKeys, ETA_DOFS, NU_DOFS, TAU_DOFS, normalize


def generate_states(
    state_space_equation: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    inputs: np.ndarray,
    parameters: np.ndarray,
    dt: float,
) -> np.ndarray:

    num_elements = len(inputs)
    num_states = 13
    synthetic_data = np.empty((num_elements, num_states), dtype=np.float64)

    eta_initial = [0, 0, 0, 1, 0, 0, 0]
    nu_initial = [0, 0, 0, 0, 0, 0]
    x = np.array(eta_initial + nu_initial, dtype=np.float64)
    synthetic_data[0] = x

    for i, tau in enumerate(inputs):
        x_dot = state_space_equation(x, tau, parameters)
        x += x_dot * dt
        x[3:7] = normalize(x[3:7])
        synthetic_data[i] = x

    return synthetic_data


def synthesize_dataset():

    input_path = Path("data/preprocessed/random-1.csv")
    save_path = Path("data/synthetic") / input_path.name

    M = [30, 60, 60, 10, 30, 30]
    D = [30, 60, 60, 10, 30, 30]
    W = [25]
    B = [24]
    COG = [0, 0, 0]
    COB = [0, 0, -0.1]
    theta = np.array(M + D + W + B + COG + COB)

    df_input = pd.read_csv(input_path)
    tau = df_input[TAU_DOFS].to_numpy()

    dt = 0.1
    time = np.round(
        np.matrix([t for t, _tmp in zip(count(step=dt), tau)]).T, decimals=1
    )

    states = generate_states(
        state_space_equation=diagonal_slow, inputs=tau, parameters=theta, dt=dt
    )

    data = np.concatenate((time, tau, states), axis=1)
    df = pd.DataFrame(data)
    df.columns = [DFKeys.TIME.value] + TAU_DOFS + ETA_DOFS + NU_DOFS
    df.to_csv(save_path)


if __name__ == "__main__":
    synthesize_dataset()
