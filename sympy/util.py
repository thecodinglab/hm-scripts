import IPython.display as dp
import sympy as sp


def backwards_substitution(u: sp.Matrix, b: sp.Matrix, symbol: str = 'x', output: bool = False) -> sp.Matrix:
    n = len(b)
    x = sp.zeros(n, 1)

    for i in range(n - 1, -1, -1):
        zero_row = all([x.is_zero for x in u[i, :]])
        if zero_row:
            x[i] = sp.Symbol(f'{symbol}{i + 1}')

            if output:
                dp.display(dp.Math(f'{symbol}_{{{i + 1}}} = {sp.latex(x[i])}'))

            continue

        s = sp.Number(0)
        if i != n - 1:
            s = u[i, i + 1:n].dot(x[i + 1:n])

        x[i] = (b[i] - s) / u[i, i]

        if output:
            eq = f'{symbol}_{{{i + 1}}} = '
            if not s.is_zero:
                eq += f'\\frac{{{sp.latex(b[i])} - ({sp.latex(s)})}}{{{sp.latex(u[i, i])}}} = '
            eq += f'\\frac{{{sp.latex(b[i] - s)}}}{{{sp.latex(u[i, i])}}} = {sp.latex(x[i])}'

            dp.display(dp.Math(eq))

    return x


def forwards_substitution(l: sp.Matrix, b: sp.Matrix, symbol: str = 'x', output: bool = False) -> sp.Matrix:
    n = len(b)
    x = sp.zeros(n, 1)

    for i in range(n):
        zero_row = all([x.is_zero for x in l[i, :]])
        if zero_row:
            x[i] = sp.Symbol(f'{symbol}{i + 1}')

            if output:
                dp.display(dp.Math(f'{symbol}_{{{i + 1}}} = {sp.latex(x[i])}'))

            continue

        s = sp.Number(0)
        if i != 0:
            s = l[i, 0:i].dot(x[0:i])

        x[i] = (b[i] - s) / l[i, i]

        if output:
            eq = f'{symbol}_{{{i + 1}}} = '
            if not s.is_zero:
                eq += f'\\frac{{{sp.latex(b[i])} - ({sp.latex(s)})}}{{{sp.latex(l[i, i])}}} = '
            eq += f'\\frac{{{sp.latex(b[i] - s)}}}{{{sp.latex(l[i, i])}}} = {sp.latex(x[i])}'

            dp.display(dp.Math(eq))

    return x


def swap_row(mat: sp.Matrix, i: int, j: int):
    mat[i, :], mat[j, :] = mat[j, :], mat[i, :]


def find_pivot(col: sp.Matrix) -> int:
    max_idx = 0
    for i in range(1, len(col)):
        if col[max_idx].is_zero or (len(col[i].free_symbols) == 0 and col[i] > col[max_idx]):
            max_idx = i

    return max_idx


def pivot(mat: sp.Matrix, perm: dict[str, sp.Matrix], j: int, output: bool = False) -> int:
    col = abs(mat[j:, j])
    max_idx = find_pivot(col)

    if max_idx != 0:
        perm_str = ' \\; '.join([f'{n} = {sp.latex(p)}' for n, p in perm.items()])
        prev_matrix = sp.latex(mat) + ' \\; ' + perm_str

        swap_row(mat, j, j + max_idx)
        for p in perm.values():
            swap_row(p, j, j + max_idx)

        if output:
            perm_str = ' \\; '.join([f'{n} = {sp.latex(p)}' for n, p in perm.items()])

            dp.display(dp.Math(
                f'\\text{{pivot: swapping rows {j + 1} and {j + max_idx + 1} in }}{prev_matrix} \\text{{ gives }}' +
                sp.latex(mat) + ' \\; ' + perm_str))
