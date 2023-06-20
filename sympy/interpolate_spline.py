import sympy as sp
import IPython.display as dp

from typing import Tuple, List


def generate_splines(t: sp.Symbol, x: sp.Matrix, degree: int) -> Tuple[sp.Expr, List[sp.Symbol]]:
    n = x.shape[0]

    dp.display(dp.Markdown(f"### Splines"))

    splines = []
    coeffs = []

    for j in range(n - 1):
        spline = 0

        for i in range(degree + 1):
            symbol = chr(ord('a') + i)
            c = sp.symbols(f"{symbol}_{{{j}}}")
            coeffs.append(c)

            s = c * (t - x[j])**i
            spline += s

        dp.display(dp.Math(f"S_{{{j}}} = {sp.latex(spline)}"))
        splines.append(spline)

    return splines, coeffs


def build_conditions(splines: List[sp.Expr], t: sp.Symbol, x: sp.Matrix, y: sp.Matrix, cond_type='natural') -> List[sp.Eq]:
    n = x.shape[0]

    dp.display(dp.Markdown(f"### Conditions"))
    dp.display(dp.Markdown(f"#### Support points"))

    # S_i(x_i) = y_i
    for i in range(n - 1):
        spline_i = splines[i].subs(t, x[i])
        res = sp.Eq(spline_i, y[i])

        dp.display(dp.Math(
            f"S_{{{i}}}({x[i]}) = {sp.latex(spline_i)} = {y[i]} \\quad \\Rightarrow \\quad {sp.latex(res)}"))
        yield res

    # S_{n - 1}(x_n) = y_n
    spline_n1 = splines[n - 2].subs(t, x[n - 1])
    res = sp.Eq(spline_n1, y[n - 1])

    dp.display(dp.Math(
        f"S_{{{n - 2}}}({x[n - 1]}) = {sp.latex(spline_n1)} = {y[n - 1]} \\quad \\Rightarrow \\quad {sp.latex(res)}"))
    yield res

    dp.display(dp.Markdown(f"#### Continuity"))

    # S_i(x_{i + 1}) = S_{i + 1}(x_{i + 1})
    for i in range(n - 2):
        spline_i = splines[i].subs(t, x[i + 1])
        spline_i1 = splines[i + 1].subs(t, x[i + 1])
        res = sp.Eq(spline_i, spline_i1)

        dp.display(dp.Math(
            f"S_{{{i}}}({x[i + 1]}) = {sp.latex(spline_i)} = {sp.latex(spline_i1)} = S_{{{i + 1}}}({x[i + 1]}) \\quad \\Rightarrow \\quad {sp.latex(res)}"))
        yield res

    dp.display(dp.Markdown(f"#### Continuity of derivatives"))

    # S_i'(x_{i + 1}) = S_{i + 1}'(x_{i + 1})
    for i in range(n - 2):
        diff_i = sp.diff(splines[i], t).subs(t, x[i + 1])
        diff_i1 = sp.diff(splines[i + 1], t).subs(t, x[i + 1])
        res = sp.Eq(diff_i, diff_i1)

        dp.display(dp.Math(
            f"S_{{{i}}}'({x[i + 1]}) = {sp.latex(diff_i)} = {sp.latex(diff_i1)} = S_{{{i + 1}}}'({x[i + 1]}) \\quad \\Rightarrow \\quad {sp.latex(res)}"))
        yield res

    dp.display(dp.Markdown(f"#### Continuity of second derivatives"))

    # S_i''(x_{i + 1}) = S_{i + 1}''(x_{i + 1})
    for i in range(n - 2):
        diff_i = sp.diff(splines[i], t, t).subs(t, x[i + 1])
        diff_i1 = sp.diff(splines[i + 1], t, t).subs(t, x[i + 1])
        res = sp.Eq(diff_i, diff_i1)

        dp.display(dp.Math(
            f"S_{{{i}}}'({x[i + 1]}) = {sp.latex(diff_i)} = {sp.latex(diff_i1)} = S_{{{i + 1}}}'({x[i + 1]}) \\quad \\Rightarrow \\quad {sp.latex(res)}"))
        yield res

    # natural conditions
    if cond_type == 'natural':
        dp.display(dp.Markdown(f"#### Natural conditions"))

        # S_0''(x_0) = 0
        diff_0 = sp.diff(splines[0], t, t).subs(t, x[0])
        res = sp.Eq(diff_0, 0)

        dp.display(dp.Math(
            f"S_{{{0}}}'({x[0]}) = {sp.latex(diff_0)} = 0 \\quad \\Rightarrow \\quad {sp.latex(res)}"))
        yield res

        # S_{n - 1}''(x_n) = 0
        diff_n1 = sp.diff(splines[n - 2], t, t).subs(t, x[n - 1])
        res = sp.Eq(diff_n1, 0)

        dp.display(dp.Math(
            f"S_{{{n - 2}}}'({x[n - 1]}) = {sp.latex(diff_n1)} = 0 \\quad \\Rightarrow \\quad {sp.latex(res)}"))
        yield res

    # periodic conditions
    if cond_type == 'periodic':
        dp.display(dp.Markdown(f"#### Periodic conditions"))

        # S_0'(x_0) = S_{n - 1}'(x_n)
        diff_0 = sp.diff(splines[0], t).subs(t, x[0])
        diff_n1 = sp.diff(splines[n - 2], t).subs(t, x[n - 1])
        res = sp.Eq(diff_0, diff_n1)

        dp.display(dp.Math(
            f"S_{{{0}}}'({x[0]}) = {sp.latex(diff_0)} = {sp.latex(diff_n1)} = S_{{{n - 2}}}'({x[n - 1]}) \\quad \\Rightarrow \\quad {sp.latex(res)}"))
        yield res

        # S_0''(x_0) = S_{n - 1}''(x_n)
        diff_0 = sp.diff(splines[0], t, t).subs(t, x[0])
        diff_n1 = sp.diff(splines[n - 2], t, t).subs(t, x[n - 1])
        res = sp.Eq(diff_0, diff_n1)

        dp.display(dp.Math(
            f"S_{{{0}}}'({x[0]}) = {sp.latex(diff_0)} = {sp.latex(diff_n1)} = S_{{{n - 2}}}'({x[n - 1]}) \\quad \\Rightarrow \\quad {sp.latex(res)}"))
        yield res


    # not-a-knot conditions
    if cond_type == 'not-a-knot':
        dp.display(dp.Markdown(f"#### Not-a-knot conditions"))

        # S_0'''(x_1) = S_1'''(x_1)
        diff_0 = sp.diff(splines[0], t, t, t).subs(t, x[1])
        diff_1 = sp.diff(splines[1], t, t, t).subs(t, x[1])
        res = sp.Eq(diff_0, diff_1)

        dp.display(dp.Math(
            f"S_{{{0}}}'({x[1]}) = {sp.latex(diff_0)} = {sp.latex(diff_1)} = S_{{{1}}}'({x[1]}) \\quad \\Rightarrow \\quad {sp.latex(res)}"))
        yield res


def spline(data: sp.Matrix, t: sp.Symbol, degree: int, cond_type = 'natural') -> sp.Expr:
    x = data[:, 0]
    y = data[:, 1]

    splines, coeffs = generate_splines(t, x, degree)
    conditions = list(build_conditions(splines, t, x, y, cond_type))

    # solve
    sol = sp.solve(conditions, coeffs)

    dp.display(dp.Markdown(f"### Coefficients"))
    for symbol in coeffs:
        dp.display(dp.Math(f"{sp.latex(symbol)} = {sp.latex(sol[symbol])}"))

    # construct spline
    splines = [splines[i].subs(sol) for i in range(len(splines))]
    spline = sp.Piecewise(*[(splines[i], t < x[i + 1])
                          for i in range(len(splines))])

    dp.display(dp.Markdown(f"### Spline"))
    dp.display(dp.Math(f"S(t) = {sp.latex(spline)}"))
    return splines
