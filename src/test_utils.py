import unittest
import numpy as np

from utils import mult_matr


class TestMethods(unittest.TestCase):
    def test_mult_mat_two_matrices(self):
        first = [[1, 2],
                 [2, 1]]
        second = [[2, 1],
                  [1, 2]]
        expected = np.dot(first, second)

        result = mult_matr(first, second)

        self.assertTrue(np.array_equal(result, expected))

    def test_mult_mat_matrix_vector(self):
        first = [1, 2]
        second = [[2, 1],
                  [1, 2]]
        expected = np.dot(first, second)

        result = mult_matr(first, second)

        self.assertTrue(np.array_equal(result, expected))

    def test_mult_mat_vector_matrix(self):
        first = [[2, 1],
                 [1, 2]]
        second = [[1], [2]]
        expected = np.dot(first, second)

        result = mult_matr(first, second)

        self.assertTrue(np.array_equal(result, expected))


if __name__ == '__main__':
    unittest.main()
