import unittest
import sympy as sp

from lgs_exakt_lr import lr


class TestLR(unittest.TestCase):
    def test_simple(self):
        a = sp.Matrix([
            [-1, 1, 1],
            [1, -3, -2],
            [5, 1, 4],
        ])

        b = sp.Matrix([0, 5, 3])

        _, _, x = lr(a, b, pivoting=False, output=False)

        self.assertEqual(x, sp.Matrix([[-1], [-4], [3]]))

    def test_forced_pivot(self):
        a = sp.Matrix([
            [1, 1, 1],
            [2, 2, 5],
            [4, 6, 8],
        ])

        b = sp.Matrix([1, 0, 0])

        _, _, x = lr(a, b, pivoting=False, output=False)

        self.assertEqual(x, sp.Matrix([sp.Rational(7, 3), -sp.Rational(2, 3), -sp.Rational(2, 3)]))

    def test_zero_row(self):
        a = sp.Matrix([
            [0, -2, 0],
            [2, -1, 1],
            [0, -2, 0],
        ])

        b = sp.Matrix([0, 0, 0])

        _, _, x = lr(a, b, pivoting=False, output=False)

        x3 = sp.Symbol('x3')
        self.assertEqual(x, sp.Matrix([-x3 / 2, 0, x3]))
