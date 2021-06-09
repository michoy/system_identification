from typing import Union
import numpy as np
from numba import njit

import helper


@njit
def linear_surge(
    state: np.ndarray, thrust: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    """AUV equation of motion for low velocities in 1DOF (surge)

    Args:
        state (np.ndarray): position and velocity in surge
        thrust (np.ndarray): current thrust in surge
        parameters (np.ndarray): mass m and damping coeffitient d

    Returns:
        list: rate of change in surge position and velocity
    """

    # model parameters
    m: float = parameters[0]
    d: float = parameters[1]

    x = state[0]
    u = state[1]

    # equations of motion
    x_dot = u
    u_dot = -(d / m) * u + (1 / m) * thrust[0]

    return np.array([x_dot, u_dot])


@njit
def diagonal_slow_without_g(
    X: np.ndarray, tau: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Simplified model of AUV in 6DOF with purly diagonal matrices.

    Assumptions:
        1) low velocities
        2) No current

    1) => negelible coriolis and mostly linear damping
    2) => no relative velocities

    Only diagonal elements are used for mass M and damping D.
    Restoring forces are based on 4.1.1 (hydrostatics of submerged vehicles) in fossen.

    Args:
        X (np.ndarray[13x1]): current position, orientation (quat) and velocity
        tau (np.ndarray[6x1]): current thrust given by the thrusters
        theta (np.ndarray[19x1]): AUV parameters for mass [0-5], damping [6-11]
                                       and restoring forces [12-19]

    Returns:
        np.ndarray: rate of change in 6DOF position and velocity (X_dot)
                    or array of nan if M is not invertible
    """

    M = np.diag(theta[0:6])  # Mass matrix (added + body)
    D = np.diag(theta[6:12])  # Linear damping matrix
    W = theta[12]  # Weight in newton
    B = theta[13]  # Bouancy in newton
    rg = theta[14:17]  # distance from CO to COG
    rb = theta[17:20]  # distance from CO to COB

    eta = X[0:7]  # position and orientation in NED frame
    nu = X[7:13]  # velocity in BODY frame
    orientation = eta[3:7]

    Jq = helper.Jq(eta)  # rotation of eta from BODY to NED
    R = helper.R(orientation)

    # check if M is invertible
    if np.linalg.det(M) == 0:
        print("Non-invertible mass matrix M recieved in diagonal_slow")
        return np.full(len(X), np.nan)
    M_inv = np.linalg.inv(M)

    fg_ned = np.array([0, 0, W])
    fb_ned = np.array([0, 0, -B])
    fg_body = R.T @ fg_ned
    fb_body = R.T @ fb_ned
    g = -np.concatenate(
        (  # restoring forces in BODY
            fg_body + fb_body,
            np.cross(rg, fg_body) + np.cross(rb, fb_body),
        )
    )

    # equations of motion
    eta_dot = Jq @ nu
    nu_dot = M_inv @ tau - M_inv @ D @ nu - M_inv @ g

    return np.concatenate((eta_dot, nu_dot))


@njit
def diagonal_slow(X: np.ndarray, tau: np.ndarray, theta: np.ndarray) -> np.ndarray:

    M = np.diag(theta[0:6])  # Mass matrix (added + body)
    D = np.diag(theta[6:12])  # Linear damping matrix

    W = 25.0  # Weight in newton
    B = 24.3  # Bouancy in newton
    rg = np.array([0, 0, 0])  # distance from CO to COG
    rb = np.array([0, 0, -0.1])  # distance from CO to COB

    eta = X[0:7]  # position and orientation in NED frame
    nu = X[7:13]  # velocity in BODY frame
    orientation = eta[3:7]

    Jq = helper.Jq(eta)  # rotation of eta from BODY to NED
    R = helper.R(orientation)

    # check if M is invertible
    if np.linalg.det(M) == 0:
        print("Non-invertible mass matrix M recieved in diagonal_slow")
        return np.full(len(X), np.nan)
    M_inv = np.linalg.inv(M)

    fg_ned = np.array([0, 0, W])
    fb_ned = np.array([0, 0, -B])
    fg_body = R.T @ fg_ned
    fb_body = R.T @ fb_ned
    g = -np.concatenate(
        (  # restoring forces in BODY
            fg_body + fb_body,
            np.cross(rg, fg_body) + np.cross(rb, fb_body),
        )
    )

    # equations of motion
    eta_dot = Jq @ nu
    nu_dot = M_inv @ tau - M_inv @ D @ nu - M_inv @ g

    return np.concatenate((eta_dot, nu_dot))
