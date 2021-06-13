import cProfile
from enum import Enum
import enum
import io
from math import cos, sin, degrees, radians
from pathlib import Path
import pstats
from typing import List, Tuple
from numba import njit

import numpy as np
from numpy import ndarray
import pandas as pd
from pyquaternion import Quaternion
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
    SURGE = "surge_vel"
    SWAY = "sway_vel"
    HEAVE = "heave_vel"
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

PREPROCESSED_DIR = Path("data/preprocessed")
SYNTHETIC_DIR = Path("data/synthetic")
PARAM_EST_DIR = Path("results/parameter_estimation")
PARAM_EST_SIM_DIR = PARAM_EST_DIR / "simulations"


@njit
def R(quat: np.ndarray) -> np.ndarray:
    """Compute rotation matrix from BODY to NED given
    a quaternion on the form [eta, eps1, eps2, eps3]

    Based on 2.72 in fossen 2021 draft.

    Args:
        quat (np.ndarray): quaternion on the form [eta, eps1, eps2, eps3]

    Returns:
        np.ndarray: linear velocity rotation matrix
    """

    eta: float = quat[0]
    eps1: float = quat[1]
    eps2: float = quat[2]
    eps3: float = quat[3]

    return np.array(
        [
            [
                1 - 2 * (eps2 ** 2 + eps3 ** 2),
                2 * (eps1 * eps2 - eps3 * eta),
                2 * (eps1 * eps3 + eps2 * eta),
            ],
            [
                2 * (eps1 * eps2 + eps3 * eta),
                1 - 2 * (eps1 ** 2 + eps3 ** 2),
                2 * (eps2 * eps3 - eps1 * eta),
            ],
            [
                2 * (eps1 * eps3 - eps2 * eta),
                2 * (eps2 * eps3 + eps1 * eta),
                1 - 2 * (eps1 ** 2 + eps2 ** 2),
            ],
        ]
    )


@njit
def T(quat: np.ndarray) -> np.ndarray:
    """Computes angular velocity rotation matrix from BODY to NED.
    Based on 2.78) in fossen 2021 draft

    Args:
        quat (np.ndarray): quaternion on the form [eta, eps1, eps2, eps3]

    Returns:
        np.ndarray: angular velocity rotation matrix
    """

    eta: float = quat[0]
    eps1: float = quat[1]
    eps2: float = quat[2]
    eps3: float = quat[3]

    return 0.5 * np.array(
        [
            [-eps1, -eps2, -eps3],
            [eta, -eps3, eps2],
            [eps3, eta, -eps1],
            [-eps2, eps1, eta],
        ]
    )


@njit
def Jq(eta: np.ndarray) -> np.ndarray:
    """Combined R and T rotation matrix for transform of nu.
    Based on eq 2.83) from fossen 2021 draft

    Args:
        eta (np.ndarray): position and orientation (quat) in NED

    Returns:
        np.ndarray: rotation matrix from BODY to NED
    """

    orientation = eta[3:7]
    J = np.zeros((7, 6))
    J[0:3, 0:3] = R(orientation)
    J[3:7, 3:6] = T(orientation)

    return J.astype(np.float64)


@njit
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return (v / norm).astype(np.float64)


@njit
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean squared error between two 2D arrays

    Args:
        y_true (np.ndarray): measurements
        y_pred (np.ndarray): predicted values

    Returns:
        np.ndarray: array of squared errors
    """
    return (
        np.square(np.subtract(y_true, y_pred)).sum(axis=0) / y_pred.shape[0]
    ).astype(np.float64)


@njit
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean absolute error between two 2D arrays

    Args:
        y_true (np.ndarray): measurements
        y_pred (np.ndarray): predicted values

    Returns:
        np.ndarray: array of squared errors
    """
    return np.absolute(
        (np.subtract(y_true, y_pred)).sum(axis=0) / y_pred.shape[0]
    ).astype(np.float64)


@njit
def mean_absolute_error_with_log(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean absolute error between two 2D arrays

    Args:
        y_true (np.ndarray): measurements
        y_pred (np.ndarray): predicted values

    Returns:
        np.ndarray: array of squared errors
    """
    errors_abs = np.absolute(
        (np.subtract(y_true, y_pred)).sum(axis=0) / y_pred.shape[0]
    ).astype(np.float64)
    for i, error in enumerate(errors_abs):
        if error > 1:
            errors_abs[i] = np.log(error)
    return errors_abs


@njit
def is_poistive_def(A: np.ndarray) -> bool:
    if is_symmetric(A):
        return (np.linalg.eigvals(A) > 0).all()
    else:
        return False


@njit
def is_symmetric(A: np.ndarray) -> bool:
    tol = 1e-8
    return (np.abs(A - A.T) < tol).all()


@njit
def normalizer(x: np.ndarray, normalize_quaternions: bool) -> np.ndarray:
    """Normalize quaternions in eta of a full state [eta, nu]

    Args:
        x (np.ndarray): full state [eta, nu]

    Returns:
        np.ndarray: [eta, nu] with normalized quaternions
    """
    if normalize_quaternions:
        x[3:7] = normalize(x[3:7])
    return x


def quat_to_degrees(q_w: float, q_x: float, q_y: float, q_z: float) -> np.ndarray:
    """Convert a quaternion to degrees using the zyx convetion

    Args:
        q_w (float): [description]
        q_x (float): [description]
        q_y (float): [description]
        q_z (float): [description]

    Returns:
        List[float]: list with roll, pitch, yaw in degrees
    """
    yaw, pitch, roll = Quaternion(
        w=q_w,
        x=q_x,
        y=q_y,
        z=q_z,
    ).yaw_pitch_roll
    return np.array([degrees(x) for x in [roll, pitch, yaw]])


def degrees_to_quat_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r = Rotation.from_euler("zyx", [yaw, pitch, roll], degrees=True)
    q_x, q_y, q_z, q_w = r.as_quat()
    return np.array([q_w, q_x, q_y, q_z])


def radians_to_quat_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r = Rotation.from_euler("zyx", [yaw, pitch, roll])
    q_x, q_y, q_z, q_w = r.as_quat()
    return np.array([q_w, q_x, q_y, q_z])


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


def load_tau(csv_path: Path):
    return pd.read_csv(csv_path)[TAU_DOFS].to_numpy(dtype=np.float64)


def load_data(
    csv_path: Path, head_num=None, dtype=np.float64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if head_num:
        df = pd.read_csv(csv_path).head(head_num)
    else:
        df = pd.read_csv(csv_path)
    return numpy_from_df(df, dtype=dtype)


def numpy_from_df(
    df: pd.DataFrame, dtype=np.float64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau = np.ascontiguousarray(df[TAU_DOFS].to_numpy(), dtype=dtype)
    y_measured = np.ascontiguousarray(df[ETA_DOFS + NU_DOFS].to_numpy(), dtype=dtype)
    x0 = np.ascontiguousarray(df[ETA_DOFS + NU_DOFS].loc[0].to_numpy(), dtype=dtype)
    timesteps = np.ascontiguousarray(df[DFKeys.TIME.value].to_numpy(), dtype=dtype)

    return x0, tau, y_measured, timesteps


def dominates(a: ndarray, b: ndarray) -> bool:
    return (a <= b).all() and (a < b).any()


def naive_pareto(X: ndarray, F: ndarray) -> Tuple[ndarray, ndarray]:
    pareto_x, pareto_y = list(), list()
    for x, f in zip(X, F):
        not_domiated = True
        for fi in F:
            if dominates(fi, f):
                not_domiated = False
                break
        if not_domiated:
            pareto_x.append(x)
            pareto_y.append(f)
    return (np.array(pareto_x), np.array(pareto_y))


def all_similar(A: ndarray) -> bool:
    for i, a in enumerate(A):
        for j, b in enumerate(A):
            if i == j:
                continue
            if not np.allclose(a, b, atol=0.001):
                return False
    return True


def X_ramp(amp=30) -> list:
    pause = [0 for _i in range(100)]
    power = lambda a: [a for _i in range(200)]
    return pause + power(amp) + pause


def X_better_ramp() -> list:
    pause = [0 for _i in range(100)]
    power = lambda a: [a for _i in range(200)]
    return pause + power(10) + pause + power(30) + pause + power(60) + pause


def X_sin(amp=30, duration=600) -> list:
    return [amp * np.sin(x / (6 * np.pi)) for x in range(duration)]
