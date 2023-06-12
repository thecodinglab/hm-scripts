import numpy as np
import sympy as sp

import IPython.display as dp

def simpson(
        f: sp.Expr,
        x: sp.Symbol,
        a: float,
        b: float,
        n: int,
        precision: int = 4,
):
    width = (b - a) / n
    linspace = np.linspace(a, b, n + 1)

    dp.display(dp.Markdown("## Simpsonregel"))

    dp.display(dp.Math(f"f({sp.latex(x)}) = {sp.latex(f)}"))
    dp.display(dp.Math(
        "Sf(h) = \\frac{h}{3} \\cdot \\left( \\frac{1}{2} \\cdot f(a) + \\sum_{i=1}^{n-1} f(a + i \\cdot h) + 2 \\cdot \\sum_{i=1}^{n} f \\left( a + i \\cdot h - \\frac{h}{2} \\right) + \\frac{1}{2} \\cdot f(b) \\right)"))
    dp.display(dp.Math(
        f"h = \\frac{{b - a}}{{n}} = \\frac{{{b} - {a}}}{{{n}}} = {width:.{precision}f}"))
    
    dp.display(dp.Markdown("### Rechenschritte"))

    a_val = f.subs(x, a)
    b_val = f.subs(x, b)

    dp.display(dp.Math(f"f(a) = f({a:.{precision}f}) = {a_val:.{precision}f}"))
    dp.display(dp.Math(f"f(b) = f({b:.{precision}f}) = {b_val:.{precision}f}"))

    f_sum = 0
    for i in range(1, n):
        val = f.subs(x, linspace[i])
        if False:
            dp.display(dp.Math(f"f({linspace[i]:.{precision}f}) = {val:.{precision}f}"))
        f_sum += val
    dp.display(dp.Math(f"\\sum_{{i=1}}^{{{n - 1}}} f(a + i \\cdot h) = {f_sum:.{precision}f}"))

    f_center_sum = 0
    for i in range(n):
        f_center_sum += f.subs(x, (linspace[i] + linspace[i + 1]) / 2)
    dp.display(dp.Math(f"\\sum_{{i=1}}^{{{n}}} f \\left( a + i \\cdot h - \\frac{{h}}{{2}} \\right) = {f_center_sum:.{precision}f}"))

    dp.display(dp.Markdown("### Resultat"))

    res = (width / 3) * (0.5 * a_val + f_sum + 2 * f_center_sum + 0.5 * b_val)
    dp.display(dp.Math(f"Sf(h) = \\frac{{{width:.{precision}f}}}{{3}} \\cdot \\left( \\frac{{1}}{{2}} \\cdot {a_val:.{precision}f} + {f_sum:.{precision}f} + 2 \\cdot {f_center_sum:.{precision}f} + \\frac{{1}}{{2}} \\cdot {b_val:.{precision}f} \\right) = {res:.{precision}f}"))
    
    dp.display(dp.Markdown("### Verifikation"))

    integral = sp.integrate(f, (x, a, b))
    dp.display(dp.Math(
        f"\\int_{{{a}}}^{{{b}}} {sp.latex(f)} \; dx = {sp.latex(integral)}"))
    dp.display(dp.Math(
        f"\\text{{Absoluter Fehler}} = |{integral - res:.{precision}f}| = {abs(integral - res):.{precision}f}"))
    dp.display(dp.Math(
        f"\\text{{Relativer Fehler}} = \\frac{{|{integral - res:.{precision}f}|}}{{|{integral:.{precision}f}|}} = {abs(integral - res) / abs(integral):.{precision}f}"))

    return res
