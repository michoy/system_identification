import numpy as np
from scipy.spatial.transform import Rotation


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


def Jq(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix using quaternions. 
    
    The function is based on Fossen chapter 2.2
    
    Caution: scipy rotations expect quaternions as [eps1, eps2, eps3, eta] while
    the book uses [eta, eps1, eps2, eps3]

    Args:
        quat (np.ndarray): quaternion that defines the rotation

    Returns:
        np.ndarray: corresponding rotation matrix
    """
    
    eta = quat[0]
    eps1 = quat[1]
    eps2 = quat[2]
    eps3 = quat[3]
    
    J = np.zeros((7,6))
    
    # define rotation T
    J[0:3, 0:3] = rotation(quat).as_matrix()
    
    # define rotation R
    J[3:7, 3:6] = 0.5 * np.array([  # TODO: optimize performance. Array creation is slow
        [-eps1, -eps2, -eps3],
        [eta, -eps3, eps2],
        [eps3, eta, -eps1],
        [-eps2, eps1, eta]
    ])
    
    return J
    