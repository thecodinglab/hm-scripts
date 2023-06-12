import sympy as sp
import IPython.display as dp


def lagrange(data: sp.Matrix, t: sp.Symbol, precision: int = 4) -> sp.Expr:
    n = data.shape[0]
    x = data[:, 0]
    y = data[:, 1]

    def l(i, t, x):
        t_num = 1
        t_den = 1

        num_str = []
        den_str = []

        for j in range(n):
            if i == j:
                continue

            num = t - x[j]
            den = x[i] - x[j]

            t_num *= num
            t_den *= den

            num_str.append(f"({sp.latex(t)} - {sp.latex(x[j])})")
            den_str.append(f"({sp.latex(x[i])} - {sp.latex(x[j])})")

        product = t_num / t_den
        poly = sp.expand(product)

        full_str_num = ' \\cdot '.join(num_str)
        full_str_den = ' \\cdot '.join(den_str)

        dp.display(dp.Math(f"l_{{{i}}}({sp.latex(t)}) = \\prod_{{j = 0, j \\neq {i}}}^{{{n - 1}}} \\frac{{t - x_j}}{{x_i - x_j}} = \\frac{{{full_str_num}}}{{{full_str_den}}} = \\frac{{{full_str_num}}}{{{t_den.evalf(precision)}}} = {sp.latex(product)} = {sp.latex(poly)}"))

        return poly

    poly_sum = 0
    for i in range(n):
        poly_sum += l(i, t, x) * y[i]

    simp = sp.simplify(poly_sum).evalf(precision)
    dp.display(dp.Math(
        f"p(t) = \\sum_{{i = 0}}^{{{n - 1}}} l_i(t) \\cdot y_i = {sp.latex(simp)}"))

    return simp
