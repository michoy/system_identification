import cProfile
from enum import Enum
import io
from pathlib import Path
import pstats

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


class DFKeys(Enum):
    TIME = "Time"
    POSITION_X = "position_x"
    POSITION_Y = "position_y"
    POSITION_Z = "position_Z"
    ORIENTATION_W = "orientation_w"
    ORIENTATION_X = "orientation_x"
    ORIENTATION_Y = "orientation_y"
    ORIENTATION_Z = "orientation_z"
    ROLL = "roll"
    PITCH = "pitch"
    YAW = "yaw"
    SURGE = "surge"
    SWAY = "sway"
    HEAVE = "heave"
    ROLL_VEL = "roll_vel"
    PITCH_VEL = "pitch_vel"
    YAW_VEL = "yaw_vel"
    FORCE_X = "force_x"
    FORCE_Y = "force_y"
    FORCE_Z = "force_z"
    TORQUE_X = "torque_x"
    TORQUE_Y = "torque_y"
    TORQUE_Z = "torque_z"


ORIENTATIONS_QUAT = [
    DFKeys.ORIENTATION_W.value,
    DFKeys.ORIENTATION_X.value,
    DFKeys.ORIENTATION_Y.value,
    DFKeys.ORIENTATION_Z.value,
]
ORIENTATIONS_EULER = [
    DFKeys.ROLL.value,
    DFKeys.PITCH.value,
    DFKeys.YAW.value,
]
POSITIONS = [
    DFKeys.POSITION_X.value,
    DFKeys.POSITION_Y.value,
    DFKeys.POSITION_Z.value,
]
LINEAR_VELOCITIES = [
    DFKeys.SURGE.value,
    DFKeys.SWAY.value,
    DFKeys.HEAVE.value,
]
ANGULAR_VELOCITIES = [
    DFKeys.ROLL_VEL.value,
    DFKeys.PITCH_VEL.value,
    DFKeys.YAW_VEL.value,
]
TAU_DOFS = [
    DFKeys.FORCE_X.value,
    DFKeys.FORCE_Y.value,
    DFKeys.FORCE_Z.value,
    DFKeys.TORQUE_X.value,
    DFKeys.TORQUE_Y.value,
    DFKeys.TORQUE_Z.value,
]
ETA_DOFS = POSITIONS + ORIENTATIONS_QUAT
ETA_EULER_DOFS = POSITIONS + ORIENTATIONS_EULER
NU_DOFS = LINEAR_VELOCITIES + ANGULAR_VELOCITIES


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
    data = {DFKeys.TIME.value: time}
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
