import sympy as sp
import numpy as np

import IPython.display as dp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from typing import Tuple


def rechteck_segment(
    i: int,
    f: sp.Expr,
    x: sp.Symbol,
    a: float,
    b: float,
    width: float = None,
    precision: int = 4
) -> Tuple[sp.Expr, Polygon]:
    if width == None:
        width = b - a

    center = (a + b) / 2.
    height = f.subs(x, center)

    expr = f"a_{{{i+1}}}"
    expr += f" = \\int_{{{a:.{precision}f}}}^{{{b:.{precision}f}}} {sp.latex(f)} \; dx"
    expr += f" \\approx {width:.{precision}f} \\cdot f \\left(\\frac{{{a:.{precision}f} + {b:.{precision}f}}}{{2}} \\right)"
    expr += f" = {width:.{precision}f} \\cdot f({center:.{precision}f})"
    expr += f" = {width:.{precision}f} \\cdot {height:.{precision}f}"
    expr += f" = {width * height:.{precision}f}"

    dp.display(dp.Math(expr))

    poly = Polygon([(a, 0), (a, height), (b, height), (b, 0)],
                   closed=True, fill=True)
    return width * height, poly


def rechteck(
    f: sp.Expr,
    x: sp.Symbol,
    a: float,
    b: float,
    n: int,
    precision: int = 4,
    plot: bool = False
):
    width = (b - a) / n
    linspace = np.linspace(a, b, n + 1)

    dp.display(dp.Markdown("## Rechteckregel"))

    dp.display(dp.Math(f"f({sp.latex(x)}) = {sp.latex(f)}"))
    dp.display(dp.Math(
        "Rf(h) = h \\cdot \\sum_{i=1}^{n} f \\left(a + i \cdot h + \\frac{h}{2} \\right)"))
    dp.display(
        dp.Math(f"h = \\frac{{b - a}}{{n}} = \\frac{{{b} - {a}}}{{{n}}} = {width:.{precision}f}"))

    dp.display(dp.Markdown("### Rechenschritte"))

    patches = []
    s = 0

    for i in range(n):
        area, poly = rechteck_segment(
            i, f, x,
            a=linspace[i],
            b=linspace[i + 1],
            width=width,
            precision=precision,
        )

        patches.append(poly)
        s += area

    dp.display(dp.Markdown("### Resultat"))
    dp.display(dp.Math(
        f"\\int_{{{a}}}^{{{b}}} {sp.latex(f)} \; dx = \\sum a_i \\approx {s:.{precision}f}"))

    dp.display(dp.Markdown("### Verifikation"))

    integral = sp.integrate(f, (x, a, b))
    dp.display(dp.Math(
        f"\\int_{{{a}}}^{{{b}}} {sp.latex(f)} \; dx = {sp.latex(integral)}"))
    dp.display(dp.Math(
        f"\\text{{Absoluter Fehler}} = |{integral - s:.{precision}f}| = {abs(integral - s):.{precision}f}"))
    dp.display(dp.Math(
        f"\\text{{Relativer Fehler}} = \\frac{{|{integral - s:.{precision}f}|}}{{|{integral:.{precision}f}|}} = {abs(integral - s) / abs(integral):.{precision}f}"))

    if plot:
        lambdified = sp.lambdify(x, f, 'numpy')
        plot_linspace = np.linspace(a, b, 1000)
        plt.plot(plot_linspace, lambdified(plot_linspace))

        collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
        collection.set_array(np.linspace(0, 1, len(patches)))
        plt.gca().add_collection(collection)

    return s
