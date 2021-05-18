import cProfile
import io
from pathlib import Path
import pstats

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ETA_DOFS = [
    "position_x",
    "position_y",
    "position_z",
    "orientation_w",
    "orientation_x",
    "orientation_y",
    "orientation_z",
]
NU_DOFS = [
    "linear_x",
    "linear_y",
    "linear_z",
    "angular_x",
    "angular_y",
    "angular_z",
]
TAU_DOFS = ["force.x", "force.y", "force.z", "torque.x", "torque.y", "torque.z"]


def rotation(quat: np.ndarray) -> Rotation:
    """Takes a quaternion on the form [eta, eps1, eps2, eps3] and
    returns a scipy rotation object. The function is useful since it
    converts the quaternion to the form scipy uses [eps1, eps2, eps3, eta]

    Args:
        quat (np.ndarray): quaternion on the form [eta, eps1, eps2, eps3]

    Returns:
        Rotation: scipy rotation object
    """

    eta = quat[0]
    eps1 = quat[1]
    eps2 = quat[2]
    eps3 = quat[3]

    return Rotation.from_quat([eps1, eps2, eps3, eta])


def Jq(eta: np.ndarray) -> np.ndarray:
    """Rotation matrix using quaternions.

    The function is based on Fossen chapter 2.2

    Caution: scipy rotations expect quaternions as [eps1, eps2, eps3, eta] while
    the book uses [eta, eps1, eps2, eps3]

    Args:
        eta (np.ndarray): position and orientation (quat) in NED

    Returns:
        np.ndarray: rotation matrix from BODY to NED
    """

    eta_quat = eta[3]
    eps1 = eta[4]
    eps2 = eta[5]
    eps3 = eta[6]

    J = np.zeros((7, 6))

    # define rotation T
    J[0:3, 0:3] = rotation(eta[3:7]).as_matrix()

    # define rotation R
    J[3:7, 3:6] = 0.5 * np.array(
        [  # TODO: optimize performance. Array creation is slow
            [-eps1, -eps2, -eps3],
            [eta_quat, -eps3, eps2],
            [eps3, eta_quat, -eps1],
            [-eps2, eps1, eta_quat],
        ]
    )

    return J


def get_nu(df: pd.DataFrame) -> np.ndarray:
    return df[NU_DOFS].to_numpy()


def get_eta(df: pd.DataFrame) -> np.ndarray:
    """Retrieve eta from dataframe

    Args:
        df (pd.DataFrame): dataframe containing eta

    Returns:
        np.ndarray: [x, y, z, q_w, q_x, q_y, q_z]
    """
    return df[ETA_DOFS].to_numpy()


def get_tau(df: pd.DataFrame) -> np.ndarray:
    return df[TAU_DOFS].to_numpy()


def make_df(
    time: np.ndarray,
    eta: np.ndarray = None,
    nu: np.ndarray = None,
    tau: np.ndarray = None,
) -> pd.DataFrame:
    data = {"Time": time}
    if type(eta) is np.ndarray:
        for dof, values in zip(ETA_DOFS, eta.T):
            data[dof] = values
    if type(nu) is np.ndarray:
        for dof, values in zip(NU_DOFS, nu.T):
            data[dof] = values
    if type(tau) is np.ndarray:
        for dof, values in zip(TAU_DOFS, tau.T):
            data[dof] = values
    return pd.DataFrame(data)


def profile(func):
    def wrapper(*args, **kwargs):
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)

        save_file = Path("profiling") / (func.__name__ + ".profile") 
        Path.mkdir(save_file.parent, parents=True, exist_ok=True)
        s = io.StringIO()
        ps = pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        with open(save_file, "w") as perf_file:
            perf_file.write(s.getvalue())

        return retval

    return wrapper
