import unittest
import numpy as np

class TestTarget(unittest.TestCase):
    def test_multiplication_targets(self):
        x = np.array([[0.0, 0.5], [-1.0, 0.5], [1.0, -1.0]])
        y = x[:, :1] * x[:, 1:2]
        np.testing.assert_allclose(y[:, 0], [0.0, -0.5, -1.0])

if __name__ == "__main__":
    unittest.main()
