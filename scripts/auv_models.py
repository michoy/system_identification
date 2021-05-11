import numpy as np
from scipy import linalg
import helper



def auv_1DOF_simplified(state: np.ndarray, thrust: np.ndarray,
                               parameters: np.ndarray) -> list:
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

    return [x_dot, u_dot]


def auv_6DOF_simplified(state: np.ndarray, thrust: np.ndarray,
                 parameters: np.ndarray) -> list:
    """Simplified model of AUV in 6DOF with purly diagonal matrices. 
    
    Assumptions:
        1) low velocities
        2) No current
    
    1) => negelible coriolis and mostly linear damping
    2) => no relative velocities
    
    Only diagonal elements are used for mass M and damping D. 
    Restoring forces are based on 4.1.1 (hydrostatics of submerged vehicles) in fossen. 
        
    Args:
        state (np.ndarray[13x1]): current position, orientation (quat) and velocity  
        thrust (np.ndarray[6x1]): current input given by the thrusters
        parameters (np.ndarray[19x1]): AUV parameters for mass [0-5], damping [6-11] 
                                       and restoring forces [12-19]

    Returns:
        list: rate of change in 6DOF position and velocity (X_dot)
    """

    M = np.diag(parameters[0:6])    # Mass matrix (added + body)
    D = np.diag(parameters[6:12])   # Linear damping matrix
    W = parameters[12]              # Weight in newton
    B = parameters[13]              # Bouancy in newton
    rg = parameters[14:17]          # distance from CO to COG
    rb = parameters[17:20]          # distance from CO to COB
    
    eta = state[0:7]                # position and orientation in NED frame
    nu = state[7:13]                # velocity in BODY frame
    
    orientation = eta[3:8]          # orientation in quaternions
    Jq = helper.Jq(orientation)     # rotation of eta from BODY to NED

    M_inv = linalg.inv(M)           # inverse mass matrix
    
    fg_ned = np.array([0, 0, W])
    fb_ned = np.array([0, 0, -B])
    R = helper.rotation(orientation)
    fg_body = R.apply(fg_ned)
    fb_body = R.apply(fb_ned) 
    g = - np.concatenate((          # restoring forces in BODY
        fg_body + fb_body,
        np.cross(rb, fg_body) + np.cross(rb, fb_body)
    ))
    

    # equations of motion
    eta_dot = Jq@nu
    nu_dot = M_inv@thrust - M_inv@D@nu - M_inv@g
    
    return [*eta_dot, *nu_dot] 
