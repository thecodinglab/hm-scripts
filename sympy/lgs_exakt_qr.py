import IPython.display as dp
import sympy as sp

from util import backwards_substitution


def sign(x: sp.Expr) -> sp.Expr:
    if x.is_zero:
        return sp.Number(1)
    return sp.sign(x)


def gen_householder_matrix(a: sp.Matrix, i: int, precision: int = -1, output: bool = False) -> sp.Matrix:
    ai = a[:, 0]
    ei = sp.eye(ai.rows)[:, 0]

    ai_norm = sp.sqrt(sum(ai[i] ** 2 for i in range(len(ai))))

    v = ai + sign(ai[0]) * ai_norm * ei
    u = v / v.norm()

    if precision == -1:
        u = sp.simplify(u)
    else:
        u = u.evalf(precision)

    h = sp.eye(ai.rows) - 2 * (u @ sp.transpose(u))
    if precision == -1:
        h = sp.simplify(h)

    if output:
        v_str = f'v_{i + 1} = {sp.latex(v)}, \\; |v_{i + 1}| = {sp.latex(v.norm())} '
        if precision == -1:
            v_str += '= ' + sp.latex(sp.simplify(v.norm()))
        else:
            v_str += '\\approx ' + sp.latex(v.norm().evalf(precision))

        dp.display(dp.Math(f'A_{i + 1} = {sp.latex(a)} \\; \\rightarrow \\; a_{i + 1} = {sp.latex(ai)}'))
        dp.display(dp.Math(v_str))
        dp.display(dp.Math(f'u_{i + 1} = {sp.latex(u)} \\; \\rightarrow \\; u_{i + 1}^T = {sp.latex(sp.transpose(u))}'))
        dp.display(dp.Math(
            f'H_{i + 1} = I_{len(ai)} - 2 u_{i + 1} u_{i + 1}^T = I_{len(ai)} - 2 \\cdot {sp.latex(u @ sp.transpose(u))} '
            f'= {sp.latex(h)}'))

    return h


def expand_matrix(mat: sp.Matrix, n: int):
    offset = n - mat.rows

    res = sp.eye(n)
    res[offset:, offset:] = mat
    return res


def qr_decompose(a: sp.Matrix, precision: int = -1, output: bool = False) -> (sp.Matrix, sp.Matrix):
    n = a.rows

    r = a.copy()
    q = sp.eye(n)

    for i in range(n - 1):
        if output:
            dp.display(dp.Markdown(f'Iteration {i + 1}'))

        hi = gen_householder_matrix(r[i:, i:], i, precision, output)
        qi = expand_matrix(hi, n)

        r = qi @ r
        q = q @ qi

        if precision == -1:
            r = sp.simplify(r)
            q = sp.simplify(q)
        else:
            # fill lower column with zeros to fix floating point errors
            r = sp.Matrix(r)
            r[(i + 1):, i] = sp.zeros(n - i - 1, 1)

        if output:
            dp.display(dp.Math(
                f'Q_{i + 1} = {sp.latex(qi)} \\rightarrow Q = Q \\cdot Q_{i + 1}^T = {sp.latex(q)}, \\; R = Q_{i + 1} '
                f'\\cdot R = {sp.latex(r)}'))

    return q, r


def qr(a: sp.Matrix, b: sp.Matrix, precision: int = -1, output: bool = False) -> [sp.Matrix, sp.Matrix, sp.Matrix]:
    if output:
        dp.display(dp.Math(f'A = {sp.latex(a)}, \\quad b = {sp.latex(b)}'))
        dp.display(dp.Markdown('## QR-Zerlegung'))

    q, r = qr_decompose(a, precision, output=output)

    if output:
        dp.display(dp.Markdown('Resultat'))
        dp.display(dp.Math(f'Q = {sp.latex(q)}, \\quad R = {sp.latex(r)}'))

    y = sp.transpose(q) @ b

    if output:
        dp.display(dp.Markdown('## Rückwärtseinsetzen'))

    x = backwards_substitution(r, y, output=output)

    if output:
        dp.display(dp.Math('x = ' + sp.latex(x)))

    return q, r, sp.simplify(x)
