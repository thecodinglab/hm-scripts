import unittest
import sympy as sp

from lgs_exakt_qr import qr


class TestQR(unittest.TestCase):
    def test_simple(self):
        a = sp.Matrix([
            [-1, 1, 1],
            [1, -3, -2],
            [5, 1, 4],
        ])

        b = sp.Matrix([0, 5, 3])

        _, _, x = qr(a, b, output=False)

        self.assertEqual(x, sp.Matrix([[-1], [-4], [3]]))

    def test_forced_pivot(self):
        a = sp.Matrix([
            [1, 1, 1],
            [2, 2, 5],
            [4, 6, 8],
        ])

        b = sp.Matrix([1, 0, 0])

        _, _, x = qr(a, b, output=False)

        self.assertEqual(x, sp.Matrix([sp.Rational(7, 3), -sp.Rational(2, 3), -sp.Rational(2, 3)]))

    def test_zero_diagonal(self):
        a = sp.Matrix([
            [0, 1, -3],
            [0, -1, -1],
            [6, 3, 9],
        ])

        b = sp.Matrix([9, -4, 3])

        _, _, x = qr(a, b, output=False)

        self.assertEqual(x, sp.Matrix([-sp.Rational(1, 4), sp.Rational(21, 4), -sp.Rational(5, 4)]))
