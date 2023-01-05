import sympy as sp
import IPython.display as dp


def backwards_substitution(u: sp.Matrix, b: sp.Matrix, symbol: str = 'x') -> sp.Matrix:
    n = len(b)
    x = sp.zeros(n, 1)

    for i in range(n - 1, -1, -1):
        s = sp.Number(0)
        if i != n - 1:
            s = u[i, i + 1:n].dot(x[i + 1:n])

        x[i] = (b[i] - s) / u[i, i]

        eq = f'{symbol}_{{{i}}} = '
        if not s.is_zero:
            eq += f'\\frac{{{sp.latex(b[i])} - ({sp.latex(s)})}}{{{sp.latex(u[i, i])}}} = '
        eq += f'\\frac{{{sp.latex(b[i] - s)}}}{{{sp.latex(u[i, i])}}} = {sp.latex(x[i])}'

        dp.display(dp.Math(eq))

    return x


def forwards_substitution(l: sp.Matrix, b: sp.Matrix, symbol: str = 'x') -> sp.Matrix:
    n = len(b)
    x = sp.zeros(n, 1)

    for i in range(n):
        s = sp.Number(0)
        if i != 0:
            s = l[i, 0:i].dot(x[0:i])

        x[i] = (b[i] - s) / l[i, i]

        eq = f'{symbol}_{{{i}}} = '
        if not s.is_zero:
            eq += f'\\frac{{{sp.latex(b[i])} - ({sp.latex(s)})}}{{{sp.latex(l[i, i])}}} = '
        eq += f'\\frac{{{sp.latex(b[i] - s)}}}{{{sp.latex(l[i, i])}}} = {sp.latex(x[i])}'

        dp.display(dp.Math(eq))

    return x


def __swap_row(mat: sp.Matrix, i: int, j: int):
    mat[i, :], mat[j, :] = mat[j, :], mat[i, :]


def pivot(mat: sp.Matrix, perm: sp.Matrix, j: int):
    col = abs(mat[j:, j])

    max_idx = 0
    for i in range(1, len(col)):
        if col[max_idx].is_zero or (len(col[i].free_symbols) == 0 and col[i] > col[max_idx]):
            max_idx = i

    if max_idx != 0:
        prev_matrix = sp.latex(mat) + ' \\; ' + sp.latex(perm)

        __swap_row(mat, j, j + max_idx)
        __swap_row(perm, j, j + max_idx)

        dp.display(dp.Math(
            f'\\text{{pivot: swapping rows {j + 1} and {j + max_idx + 1} in }}{prev_matrix} \\text{{ gives }}' +
            sp.latex(mat) + sp.latex(perm)))
