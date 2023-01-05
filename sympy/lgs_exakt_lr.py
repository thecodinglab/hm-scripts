import IPython.display as dp
import sympy as sp

from util import backwards_substitution, forwards_substitution, pivot


def lr_decompose(a: sp.Matrix, pivoting: bool = False, output: bool = False) -> (sp.Matrix, sp.Matrix, sp.Matrix):
    n = a.rows

    l = sp.zeros(n)
    u = a.copy()
    perm = sp.eye(n)

    for k in range(n - 1):
        if u[k, k].is_zero or pivoting:
            pivot(u, {'L': l, 'P': perm}, k, output=output)

        for i in range(k + 1, n):
            if not u[i, k].is_zero:
                factor = u[i, k] / u[k, k]
                u[i, k:n] -= factor * u[k, k:n]
                l[i, k] = factor

                if output:
                    dp.display(dp.Math(
                        f'z_{{{i + 1}}} \\equiv z_{{{i + 1}}} - ({sp.latex(factor)}) \\cdot z_{{{k + 1}}} '
                        f'\\Rightarrow R_{k + i} = {sp.latex(u)}'))

    l = l + sp.eye(n)

    if output:
        dp.display(dp.Math(f'L = {sp.latex(l)}, \\quad R = {sp.latex(u)}, \\quad P = {sp.latex(perm)}'))

    return l, u, perm


def lr(a: sp.Matrix, b: sp.Matrix, pivoting: bool = False, output: bool = False) -> [sp.Matrix, sp.Matrix, sp.Matrix]:
    if output:
        dp.display(dp.Math(f'A = {sp.latex(a)}, \\quad b = {sp.latex(b)}'))
        dp.display(dp.Markdown('## LR-Zerlegung'))

    l, u, perm = lr_decompose(a, pivoting=pivoting, output=output)

    if output:
        dp.display(dp.Markdown('## Vorwärtseinsetzen'))

        if perm != sp.eye(b.rows):
            dp.display(dp.Math(
                f'Ly = Pb: {sp.latex(l)} y = {sp.latex(perm)} {sp.latex(b)} = {sp.latex(perm @ b)}'))
        else:
            dp.display(dp.Math(
                f'Ly = b: {sp.latex(l)} y = {sp.latex(b)}'))

    y = forwards_substitution(l, perm @ b, symbol='y', output=output)

    if output:
        dp.display(dp.Math('y = ' + sp.latex(y)))

        dp.display(dp.Markdown('## Rückwärtseinsetzen'))
        dp.display(dp.Math(f'Rx = y: {sp.latex(u)} x = {sp.latex(y)}'))

    x = backwards_substitution(u, y, symbol='x', output=output)

    if output:
        dp.display(dp.Math('x = ' + sp.latex(x)))

    return l, u, x

# TODO solve with column containing all zeros
