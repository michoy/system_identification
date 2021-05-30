from pathlib import Path
from typing import Callable, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count

from auv_models import auv_1DOF_simplified, diagonal_slow
from helper import DFKeys, ETA_DOFS, NU_DOFS, SYNTHETIC_DIR, TAU_DOFS
from parameter_estimation import predict


def synthesize_dataset(
    state_space_equation,
    x0: np.ndarray,
    params: np.ndarray,
    tau: np.ndarray,
    colums: List[str],
    normalize_quaternions: bool,
    save_name: Union[None, str] = None,
) -> pd.DataFrame:

    dt = 0.1
    time = np.round(
        np.matrix([t for t, _tmp in zip(count(step=dt), tau)]).T, decimals=1
    )

    states = predict(state_space_equation, x0, tau, dt, params, normalize_quaternions)

    data = np.concatenate((time, tau, states), axis=1)

    df = pd.DataFrame(data)
    df.columns = colums

    if save_name:
        save_path = SYNTHETIC_DIR / save_name
        df.to_csv(save_path)

    return df


if __name__ == "__main__":

    m = 30
    d = 30
    theta = np.array([m, d], dtype=np.float64)

    tau_1 = [[0, 0, 0, 0, 0, 0] for _i in range(10)]
    tau_2 = [[10, 0, 0, 0, 0, 0] for _i in range(100)]
    tau_3 = [[0, 0, 0, 0, 0, 0] for _i in range(10)]
    tau_4 = [[-10, 0, 0, 0, 0, 0] for _i in range(100)]
    tau_5 = [[0, 0, 0, 0, 0, 0] for _i in range(10)]
    tau = np.array(tau_1 + tau_2 + tau_3 + tau_4 + tau_5, dtype=np.float64)

    x0 = np.array([0, 0], dtype=np.float64)

    columns = (
        [DFKeys.TIME.value] + TAU_DOFS + [DFKeys.POSITION_X.value, DFKeys.SURGE.value]
    )

    save_name = "only_surge.csv"

    synthesize_dataset(auv_1DOF_simplified, x0, theta, tau, columns, False, save_name)
