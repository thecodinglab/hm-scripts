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


def newton_step(i: int, symbols: sp.Matrix, funcs: sp.Matrix, jacobian: sp.Matrix, x: sp.Matrix, precision: int):
    dp.display(dp.Markdown(f"Iteration {i + 1}"))

    f_x = funcs.subs(zip(symbols, x))
    df_x = jacobian.subs(zip(symbols, x))
    df_x_inv = df_x.inv()

    delta = -df_x_inv * f_x

    f_x_eval = f_x.evalf(precision)
    df_x_eval = df_x.evalf(precision)
    delta_eval = delta.evalf(precision)

    dp.display(dp.Math(
        f"f(x_{{{i}}}) = f \\left( {sp.latex(x)} \\right) = {sp.latex(f_x)} \\approx {sp.latex(f_x_eval)}"))
    dp.display(dp.Math(
        f"Df(x_{{{i}}}) = Df \\left( {sp.latex(x)} \\right) = {sp.latex(df_x)} \\approx {sp.latex(df_x_eval)}"))
    dp.display(dp.Math(
        f"\\delta = -Df(x_{{{i}}})^{{-1}} f(x_{{{i}}}) = -{sp.latex(df_x)}^{{-1}} {sp.latex(f_x)} = -{sp.latex(df_x_inv)} {sp.latex(f_x)} = {sp.latex(delta)} \\approx {sp.latex(delta_eval)}"))

    x_next = x + delta
    x_next_eval = x_next.evalf(precision)

    dp.display(dp.Math(
        f"x_{{{i + 1}}} = x_{{{i}}} + \\delta = {sp.latex(x)} + {sp.latex(delta)} = {sp.latex(x_next)} \\approx {sp.latex(x_next_eval)}"))

    return x_next_eval


def newton(symbols: sp.Matrix, funcs: sp.Matrix, x0: sp.Matrix, n: int, precision: int = 4):
    dp.display(dp.Math(f"f(x) = {sp.latex(funcs)}"))
    df = jacobian(symbols, funcs)

    dp.display(dp.Markdown("## Newton Verfahren"))

    x = x0
    for i in range(n):
        x = newton_step(i, symbols, funcs, df, x, precision)
