import IPython.display as dp
import sympy as sp

from util import backwards_substitution, pivot


def gauss(a: sp.Matrix, b: sp.Matrix, pivoting: bool = False, output: bool = False) -> sp.Matrix:
    if output:
        dp.display(dp.Math(f'A = {sp.latex(a)}, \\quad b = {sp.latex(b)}'))
        dp.display(dp.Markdown('## Gauss'))

    n = len(b)
    u = a.copy()
    b = b.copy()

    for k in range(n - 1):
        if u[k, k].is_zero or pivoting:
            pivot(u, {'b': b}, k, output=output)

        for i in range(k + 1, n):
            if not u[i, k].is_zero:
                factor = u[i, k] / u[k, k]
                u[i, k:n] -= factor * u[k, k:n]
                b[i] -= factor * b[k]

                if output:
                    dp.display(dp.Math(
                        f'z_{{{i + 1}}} \\equiv z_{{{i + 1}}} - ({sp.latex(factor)}) \\cdot z_{{{k + 1}}} \\Rightarrow ' + sp.latex(
                            u) + ' \\; ' + sp.latex(b)))

    if output:
        dp.display(dp.Markdown('## Rückwärtseinsetzen'))

    x = backwards_substitution(u, b, symbol='x', output=output)

    if output:
        dp.display(dp.Math('x = ' + sp.latex(x)))

    return x


# TODO fehlerfortpflanzung
