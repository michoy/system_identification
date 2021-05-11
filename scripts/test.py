import unittest

import numpy as np

from auv_models import auv_1DOF_simplified, auv_6DOF_simplified
import helper


class SimplifiedSurgeAUVTest(unittest.TestCase):
    def test_auv_1DOF(self):
        x = 0
        u = 0.1
        tau = 1
        m = 10
        d = 20

        state = np.array([x, u])
        thrust = np.array([tau])
        parameters = np.array([m, d])

        result = auv_1DOF_simplified(state, thrust, parameters)
        self.assertEqual(result, [0.1, -0.1])
        

class FullDofAUVSimplifiedTests(unittest.TestCase):

    def test_dimensions_and_type(self):
        eta = [0, 0, 0, 1, 0, 0, 0]
        nu = [0, 0, 0, 0, 0, 0]
        state = np.array(eta + nu)

        M = [10, 10, 10, 10, 10, 10]
        D = [10, 10, 10, 10, 10, 10]
        W = [0]
        B = [0]
        COG = [0, 0, 0]
        COB = [0, 0, 0]
        parameters = np.array(M + D + W + B + COG + COB)

        thrust = np.array([0, 0, 0, 0, 0, 0])

        x_dot = auv_6DOF_simplified(state, thrust, parameters)

        self.assertEqual(type(x_dot), list)
        self.assertEqual(len(x_dot), 13)


class HelperTests(unittest.TestCase):
    
    def test_dimension(self):
        quat = np.array([1, 0, 0, 0])
        Jq = helper.Jq(quat)
        
        self.assertEqual(Jq.shape, (7,6))

    
    def test_trivial_quat(self):
        quat = np.array([1, 0, 0, 0])
        J = helper.Jq(quat)
        
        res = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0.5],
        ])
        
        comparison = J == res
        self.assertTrue(comparison.all())


if __name__ == '__main__':
    unittest.main()
