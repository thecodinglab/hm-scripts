import sympy as sp

import IPython.display as dp


def romberg_init(
    j: int,
    f: sp.Expr,
    x: sp.Symbol,
    a: float,
    b: float,
    precision: int = 4,
):
    two_j = 2 ** j
    h = (b - a) / two_j

    f_a = f.subs(x, a)
    f_b = f.subs(x, b)
    t1 = (f_a + f_b) / 2

    sum_xi = [a + i * h for i in range(1, two_j)]
    sum_yi = [f.subs(x, xi) for xi in sum_xi]

    total_sum = t1 + sum(sum_yi)
    res = h * total_sum

    sum_xi_expr = " + ".join([f"f({xi:.{precision}f})" for xi in sum_xi])
    sum_yi_expr = " + ".join([f"{yi:.{precision}f}" for yi in sum_yi])

    expr = "$\\begin{aligned}\n"
    expr += f"T_{{{j}, 0}} &= \\frac{{{b} - {a}}}{{2^{{{j}}}}} \\cdot \\left( \\frac{{f({a}) + f({{b}})}}{{2}} + \\sum_{{i=1}}^{{2^{{{j}}} - 1}} f({a} + i \\cdot {h}) \\right) \\\\\n"
    expr += f" &= \\frac{{{b - a}}}{{{two_j}}} \\cdot \\left( \\frac{{{f_a:.{precision}f} + {f_b:.{precision}f}}}{{2}} + \\sum_{{i=1}}^{{{two_j} - 1}} f({a} + i \\cdot {h}) \\right) \\\\\n"
    expr += f" &= {h:.{precision}f} \\cdot \\left( {t1:.{precision}f} + \\sum_{{i=1}}^{{{two_j - 1}}} f({a} + i \\cdot {h}) \\right) \\\\\n"

    if len(sum_xi) > 0:
        expr += f" &= {h:.{precision}f} \\cdot \\left( {t1:.{precision}f} + {sum_xi_expr} \\right) \\\\\n"
        expr += f" &= {h:.{precision}f} \\cdot \\left( {t1:.{precision}f} + {sum_yi_expr} \\right) \\\\\n"

    expr += f" &= {h:.{precision}f} \\cdot {total_sum:.{precision}f} \\\\\n"
    expr += f" &= {res:.{precision}f}\n"

    expr += "\\end{aligned}$"
    dp.display(dp.Math(expr))

    return res


def romberg_step(
    j: int,
    k: int,
    T: list[list[float]],
    precision: int = 4,
):
    four_k = 4 ** k
    four_k_minus_one = four_k - 1

    T_jk_minus_one = T[j][k - 1]
    T_j_minus_one_k_minus_one = T[j - 1][k - 1]

    res = (four_k * T_jk_minus_one - T_j_minus_one_k_minus_one) / four_k_minus_one

    expr = f"T_{{{j}, {k}}} = \\frac{{4^{{{k}}} \\cdot T_{{{j}, {k-1}}} - T_{{{j-1}, {k-1}}}}}{{4^{{{k}}} - 1}}"
    expr += f" = \\frac{{{four_k} \\cdot {T_jk_minus_one:.{precision}f} - {T_j_minus_one_k_minus_one:.{precision}f}}}{{{four_k_minus_one}}}"
    expr += f" = {res:.{precision}f}"
    dp.display(dp.Math(expr))

    return res


def romberg(
    f: sp.Expr,
    x: sp.Symbol,
    a: float,
    b: float,
    n: int,
    precision: int = 4
):
    dp.display(dp.Markdown("## Romberg-Extrapolation"))
    dp.display(dp.Math(
        "T_{j0} = \\frac{b - a}{2^j} \\cdot \\left( \\frac{f(a) + f(b)}{2} + \\sum_{i=1}^{2^j - 1} f(a + i \\cdot h) \\right)"))
    dp.display(dp.Math(
        "T_{j,k} = \\frac{4^k \\cdot T_{j, k-1} - T_{j-1, k-1}}{4^k - 1}"))

    dp.display(dp.Markdown("### Rechenschritte"))

    T = [[0 for _ in range(n)] for _ in range(n)]

    dp.display(dp.Markdown("#### Initialisierung"))
    for j in range(n):
        T[j][0] = romberg_init(j, f, x, a, b, precision)

    dp.display(dp.Markdown("#### Iteration"))
    for j in range(1, n):
        for k in range(1, j + 1):
            T[j][k] = romberg_step(j, k, T, precision)

    dp.display(dp.Markdown("### Ergebnis"))

    expr = "$\\begin{matrix}\n"
    for j in range(n):
        for k in range(j + 1):
            expr += f"T_{{{j}, {k}}} = {T[j][k]:.{precision}f} & "
        expr += "\\\\\n"
    expr += "\\end{matrix}$"

    res = T[n - 1][n - 1]

    dp.display(dp.Math(expr))
    dp.display(
        dp.Math(f"T = T_{{j, k}} = T_{{{n-1}, {n-1}}} = {res:.{precision}f}"))

    dp.display(dp.Markdown("### Verifikation"))

    integral = sp.integrate(f, (x, a, b))
    dp.display(
        dp.Math(f"\\int_{{{a}}}^{{{b}}} f(x) \\, dx = {integral:.{precision}f}"))
    dp.display(dp.Math(
        f"\\text{{Absoluter Fehler}} = |{integral - res:.{precision}f}| = {abs(integral - res):.{precision}f}"))
    dp.display(dp.Math(
        f"\\text{{Relativer Fehler}} = \\frac{{|{integral - res:.{precision}f}|}}{{|{integral:.{precision}f}|}} = {abs(integral - res) / abs(integral):.{precision}f}"))

    return res
