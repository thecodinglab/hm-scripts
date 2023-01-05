import IPython.display as dp
import sympy as sp

from lgs_exakt_lr import lr
from lgs_exakt_qr import qr_decompose


def is_approx_zero(x: sp.Expr, precision: int = -1) -> bool:
    if x.is_zero:
        return True

    if precision == -1:
        return False

    return x.evalf(precision).round(precision) == 0


def round_entries(a: sp.Matrix, precision: int = -1) -> sp.Matrix:
    return sp.Matrix([[x.round(precision) for x in a.row(row)] for row in range(a.rows)])


def qr_iter(a: sp.Matrix, iterations: int = 100, precision: int = -1, output: bool = False) -> [sp.Matrix, sp.Matrix]:
    n = a.rows
    p = sp.eye(n)

    for i in range(iterations):
        qi, ri = qr_decompose(a, precision)

        a = ri @ qi
        p = p @ qi

        if output:
            dp.display(dp.Math(f'A_{{ {i + 1} }} = {sp.latex(a)}, \\quad P_{{ {i + 1} }} = {sp.latex(p)}'))

    return a, p


def qr_eigen(a: sp.Matrix, iterations: int = 100, precision: int = -1, output: bool = False):
    n = a.rows
    a_iter, _ = qr_iter(a, iterations=iterations, precision=precision, output=output)

    # extract eigenvalues
    eigenvalues = []
    for k in range(n):
        left_zero = k == 0 or is_approx_zero(a_iter[k, k - 1], precision)
        if not left_zero:
            continue

        down_zero = k == n - 1 or is_approx_zero(a_iter[k + 1, k], precision)
        if down_zero:
            eigenvalues.append(a_iter[k, k])
            continue

        # extract 2x2 block to find eigenvalues of quadratic polynomial
        block = a_iter[k:k + 2, k:k + 2]

        l = sp.var('lambda')
        poly = sp.det(block - l * sp.eye(2))

        roots = sp.solve(poly, l)
        eigenvalues.extend(roots)

    return eigenvalues
