import IPython.display as dp
import sympy as sp


def jacobi_iterate(a: sp.Matrix, b: sp.Matrix, x: sp.Matrix, precision: int = -1) -> sp.Matrix:
    res = sp.zeros(x.rows, x.cols)

    for i in range(a.rows):
        s = 0
        s_expr = ''

        for j in range(a.cols):
            if i != j:
                s += a[i, j] * x[j]
                s_expr += f' + ({sp.latex(a[i, j])} \\cdot {sp.latex(x[j])})'

        res[i] = (1 / a[i, i]) * (b[i] - s)
        if precision != -1:
            res[i] = res[i].evalf(precision)

        dp.display(dp.Math(
            f'x_{{ {i} }} = \\frac{{ {sp.latex(b[i])} }}{{ {sp.latex(a[i, i])} }} - \\frac{{ {s_expr[3:]} }}{{ {sp.latex(a[i, i])} }} = {sp.latex(res[i])}'))

    return res


def jacobi(a: sp.Matrix, b: sp.Matrix, x0: sp.Matrix, n: int = 100, precision: int = -1) -> sp.Matrix:
    x = x0
    for i in range(n):
        dp.display(dp.Latex(f'Iteration {i + 1}'))
        x = jacobi_iterate(a, b, x, precision)
    return x
