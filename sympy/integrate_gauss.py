import sympy as sp

import IPython.display as dp


def gauss_1(f: sp.Expr, x: sp.Symbol, a: float, b: float, precision: int = 4):
    f_x = f.subs(x, (a + b) / 2)

    expr = "G_1f = (b - a) \\cdot f \\left( \\frac{a + b}{2} \\right)"
    expr += f" = ({b} - {a}) \\cdot f \\left( \\frac{{{a} + {b}}}{{2}} \\right)"
    expr += f" = ({b - a}) \\cdot f \\left( \\frac{{{a + b}}}{{2}} \\right)"
    expr += f" = ({b - a}) \\cdot f \\left( {((a + b) / 2):.{precision}f} \\right)"
    expr += f" = ({b - a}) \\cdot {f_x:.{precision}f}"
    expr += f" = {((b - a) * f_x):.{precision}f}"

    dp.display(dp.Math(expr))
    return (b - a) * f_x


def gauss_2(f: sp.Expr, x: sp.Symbol, a: float, b: float, precision: int = 4):
    t1 = (a + b) / 2
    t2 = (b - a) / 2
    t3 = 1 / sp.sqrt(3) * t2

    f1 = f.subs(x, -t3 + t1)
    f2 = f.subs(x, t3 + t1)
    res = t2 * (f1 + f2)

    expr = "$\\begin{aligned}\n"
    expr += "G_2f &= \\frac{b - a}{2} \\cdot \\left[ f \\left( - \\frac{1}{\\sqrt{3}} \\cdot \\frac{b - a}{2} + \\frac{a + b}{2} \\right) + f \\left( \\frac{1}{\\sqrt{3}} \\cdot \\frac{b - a}{2} + \\frac{a + b}{2} \\right) \\right] \\\\\n"
    expr += f" &= \\frac{{{b} - {a}}}{{2}} \\cdot \\left[ f \\left( - \\frac{{1}}{{\\sqrt{{3}}}} \\cdot \\frac{{{b} - {a}}}{{2}} + \\frac{{{a} + {b}}}{{2}} \\right) + f \\left( \\frac{{1}}{{\\sqrt{{3}}}} \\cdot \\frac{{{b} - {a}}}{{2}} + \\frac{{{a} + {b}}}{{2}} \\right) \\right] \\\\\n"
    expr += f" &= \\frac{{{b - a}}}{{2}} \\cdot \\left[ f \\left( - \\frac{{{b - a}}}{{2 \\cdot \\sqrt{{3}}}} + \\frac{{{a + b}}}{{2}} \\right) + f \\left( \\frac{{{b - a}}}{{2 \\cdot \\sqrt{{3}}}} + \\frac{{{a + b}}}{{2}} \\right) \\right] \\\\\n"
    expr += f" &= {t2:.{precision}f} \\cdot \\left[ f \\left( {-t3 + t1:.{precision}f} \\right) + f \\left( {t3 + t1:.{precision}f} \\right) \\right] \\\\\n"
    expr += f" &= {t2:.{precision}f} \\cdot \\left[ {f1:.{precision}f} + {f2:.{precision}f} \\right] \\\\\n"
    expr += f" &= {res:.{precision}f}\n"
    expr += "\\end{aligned}$"

    dp.display(dp.Latex(expr))

    return res


def gauss_3(f: sp.Expr, x: sp.Symbol, a: float, b: float, precision: int = 4):
    t1 = (a + b) / 2
    t2 = (b - a) / 2
    t3 = sp.sqrt(0.6) * t2

    f1 = f.subs(x, -t3 + t1)
    f2 = f.subs(x, t1)
    f3 = f.subs(x, t3 + t1)

    res = t2 * (5 / 9 * f1 + 8 / 9 * f2 + 5 / 9 * f3)

    expr = "$\\begin{aligned}\n"
    expr += "G_3f &= \\frac{b - a}{2} \\cdot \\left[ \\frac{5}{9} \\cdot f \\left( - \\sqrt{0.6} \\cdot \\frac{b - a}{2} + \\frac{a + b}{2} \\right) + \\frac{8}{9} \\cdot f \\left( \\frac{a + b}{2} \\right) + \\frac{5}{9} \\cdot f \\left( \\sqrt{0.6} \\cdot \\frac{b - a}{2} + \\frac{a + b}{2} \\right) \\right] \\\\\n"
    expr += f" &= \\frac{{{b} - {a}}}{{2}} \\cdot \\left[ \\frac{{5}}{{9}} \\cdot f \\left( - \\sqrt{{0.6}} \\cdot \\frac{{{b} - {a}}}{{2}} + \\frac{{{a} + {b}}}{{2}} \\right) + \\frac{{8}}{{9}} \\cdot f \\left( \\frac{{{a} + {b}}}{{2}} \\right) + \\frac{{5}}{{9}} \\cdot f \\left( \\sqrt{{0.6}} \\cdot \\frac{{{b} - {a}}}{{2}} + \\frac{{{a} + {b}}}{{2}} \\right) \\right] \\\\\n"
    expr += f" &= \\frac{{{b - a}}}{{2}} \\cdot \\left[ \\frac{{5}}{{9}} \\cdot f \\left( - \\sqrt{{0.6}} \\cdot \\frac{{{b - a}}}{{2}} + \\frac{{{a + b}}}{{2}} \\right) + \\frac{{8}}{{9}} \\cdot f \\left( \\frac{{{a + b}}}{{2}} \\right) + \\frac{{5}}{{9}} \\cdot f \\left( \\sqrt{{0.6}} \\cdot \\frac{{{b - a}}}{{2}} + \\frac{{{a + b}}}{{2}} \\right) \\right] \\\\\n"
    expr += f" &= {t2:.{precision}f} \\cdot \\left[ \\frac{{5}}{{9}} \\cdot f \\left( {-t3 + t1:.{precision}f} \\right) + \\frac{{8}}{{9}} \\cdot f \\left( {t1:.{precision}f} \\right) + \\frac{{5}}{{9}} \\cdot f \\left( {t3 + t1:.{precision}f} \\right) \\right] \\\\\n"
    expr += f" &= {t2:.{precision}f} \\cdot \\left[ \\frac{{5}}{{9}} \\cdot {f1:.{precision}f} + \\frac{{8}}{{9}} \\cdot {f2:.{precision}f} + \\frac{{5}}{{9}} \\cdot {f3:.{precision}f} \\right] \\\\\n"
    expr += f" &= {res:.{precision}f}\n"
    expr += "\\end{aligned}$"

    dp.display(dp.Latex(expr))

    return res
