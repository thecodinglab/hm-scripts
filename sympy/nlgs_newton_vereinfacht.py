import IPython.display as dp
import sympy as sp


def print_partial(name: str, func: sp.Expr, symbol: sp.Symbol):
    df = sp.diff(func, symbol)

    dp.display(dp.Math(
        f"\\frac{{\\partial {name}}}{{\\partial {sp.latex(symbol)}}} = \\frac{{\\partial}}{{\\partial {sp.latex(symbol)}}} \\left( {sp.latex(func)} \\right) = {sp.latex(df)}"))


def jacobian(symbols: sp.Matrix, funcs: sp.Matrix):
    dp.display(dp.Markdown("## Partial Derivatives"))

    for i, func in enumerate(funcs):
        for _, symbol in enumerate(symbols):
            print_partial(f"f_{i}", func, symbol)

    dp.display(dp.Markdown("## Jacobi Matrix"))

    res = funcs.jacobian(symbols)
    dp.display(dp.Math(f"Df(x) = {sp.latex(res)}"))

    return res


def newton_step(i: int, symbols: sp.Matrix, funcs: sp.Matrix, df_x0_inv: sp.Matrix, x: sp.Matrix, precision: int):
    dp.display(dp.Markdown(f"Iteration {i + 1}"))

    f_x = funcs.subs(zip(symbols, x))
    delta = -df_x0_inv * f_x

    f_x_eval = f_x.evalf(precision)
    delta_eval = delta.evalf(precision)

    dp.display(dp.Math(
        f"f(x_{{{i}}}) = f \\left( {sp.latex(x)} \\right) = {sp.latex(f_x)} \\approx {sp.latex(f_x_eval)}"))
    dp.display(dp.Math(
        f"\\delta = -Df(x_0)^{{-1}} f(x_{{{i}}}) = - {sp.latex(df_x0_inv)} {sp.latex(f_x)} = {sp.latex(delta)} \\approx {sp.latex(delta_eval)}"))

    x_next = x + delta
    x_next_eval = x_next.evalf(precision)

    dp.display(dp.Math(
        f"x_{{{i + 1}}} = x_{{{i}}} + \\delta = {sp.latex(x)} + {sp.latex(delta)} = {sp.latex(x_next)} \\approx {sp.latex(x_next_eval)}"))

    return x_next_eval


def newton(symbols: sp.Matrix, funcs: sp.Matrix, x0: sp.Matrix, n: int, precision: int = 4):
    dp.display(dp.Math(f"f(x) = {sp.latex(funcs)}"))

    df = jacobian(symbols, funcs)
    df_x0 = df.subs(zip(symbols, x0))
    df_x0_eval = df_x0.evalf(precision)

    df_x0_inv = df_x0.inv()
    df_x0_inv_eval = df_x0_inv.evalf(precision)

    dp.display(dp.Math(
        f"Df(x_0) = Df \\left( {sp.latex(x0)} \\right) = {sp.latex(df_x0)} \\approx {sp.latex(df_x0_eval)}"))
    dp.display(dp.Math(
        f"Df(x_0)^{{-1}} = \\left( Df \\left( {sp.latex(x0)} \\right) \\right)^{{-1}} = {sp.latex(df_x0)}^{{-1}} = {sp.latex(df_x0_inv)} \\approx {sp.latex(df_x0_inv_eval)}"))

    dp.display(dp.Markdown("## Newton Verfahren"))

    x = x0
    for i in range(n):
        x = newton_step(i, symbols, funcs, df_x0_inv, x, precision)
