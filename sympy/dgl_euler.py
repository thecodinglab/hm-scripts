import sympy as sp
import IPython.display as dp


def classic(f, a, b, n, y0, print_steps=False):
    h = (b - a) / n

    x_list = [a]
    y_list = [y0]

    for i in range(n):
        x = x_list[-1]
        y = y_list[-1]

        y_next = y + h * f(x, y)
        x_next = x + h

        if print_steps:
            x_next_str = f"x_{{{i + 1}}} = x_{{{i}}} + h = {sp.latex(x)} + {h} = {sp.latex(x_next)}"
            y_next_str = f"y_{{{i + 1}}} = y_{{{i}}} + h \\cdot f(x_{{{i}}}, y_{{{i}}}) = {sp.latex(y)} + {h} \\cdot {sp.latex(f(x, y))} = {sp.latex(y_next)}"
            dp.display(dp.Math(x_next_str + " \\quad " + y_next_str))

        x_list.append(x_next)
        y_list.append(y_next)

    return x_list, y_list


def midpoint(f, a, b, n, y0, print_steps=False):
    h = (b - a) / n

    x_list = [a]
    y_list = [y0]

    for i in range(n):
        x = x_list[-1]
        y = y_list[-1]

        x_mid = x + h / 2
        y_mid = y + h / 2 * f(x, y)

        if print_steps:
            x_mid_str = f"x_{{{i + 0.5}}} = x_{{{i}}} + \\frac{{h}}{{2}} = {sp.latex(x)} + \\frac{{{h}}}{{2}} = {sp.latex(x_mid)}"
            y_mid_str = f"y_{{{i + 0.5}}} = y_{{{i}}} + \\frac{{h}}{{2}} \\cdot f(x_{{{i}}}, y_{{{i}}}) = {sp.latex(y)} + \\frac{{{h}}}{{2}} \\cdot {sp.latex(f(x, y))} = {sp.latex(y_mid)}"
            dp.display(dp.Math(x_mid_str + " \\quad " + y_mid_str))

        y_next = y + h * f(x_mid, y_mid)
        x_next = x + h

        if print_steps:
            x_next_str = f"x_{{{i + 1}}} = x_{{{i}}} + h = {sp.latex(x)} + {h} = {sp.latex(x_next)}"
            y_next_str = f"y_{{{i + 1}}} = y_{{{i}}} + h \\cdot f(x_{{{i + 0.5}}}, y_{{{i + 0.5}}}) = {sp.latex(y)} + {h} \\cdot f \\left({sp.latex(x_mid)}, {sp.latex(y_mid)} \\right) = {sp.latex(y)} + {h} \\cdot {sp.latex(f(x_mid, y_mid))} = {sp.latex(y_next)}"
            dp.display(dp.Math(x_next_str + " \\quad " + y_next_str))

        y_list.append(y_next)
        x_list.append(x_next)

    return x_list, y_list


def modified(f, a, b, n, y0, print_steps=False):
    h = (b - a) / n

    x_list = [a]
    y_list = [y0]

    for i in range(n):
        x = x_list[-1]
        y = y_list[-1]

        k1 = f(x, y)
        k2 = f(x + h, y + h * k1)

        if print_steps:
            k1_str = f"k_1 = f(x_{{{i}}}, y_{{{i}}}) = {sp.latex(f(x, y))}"
            k2_str = f"k_2 = f(x_{{{i}}} + h, y_{{{i}}} + h \\cdot k_1) = f \\left({sp.latex(x)} + {h}, {sp.latex(y)} + {h} \\cdot {sp.latex(k1)} \\right) = {sp.latex(f(x + h, y + h * k1))}"
            dp.display(dp.Math(k1_str + " \\quad " + k2_str))

        y_next = y + h * (k1 + k2) / 2
        x_next = x + h

        if print_steps:
            x_next_str = f"x_{{{i + 1}}} = x_{{{i}}} + h = {sp.latex(x)} + {h} = {sp.latex(x_next)}"
            y_next_str = f"y_{{{i + 1}}} = y_{{{i}}} + \\frac{{h}}{{2}} \\cdot (k_1 + k_2) = {sp.latex(y)} + \\frac{{{h}}}{{2}} \\cdot \\left( {sp.latex(k1)} + {sp.latex(k2)} \\right) = {sp.latex(y_next)}"
            dp.display(dp.Math(x_next_str + " \\quad " + y_next_str))

        y_list.append(y_next)
        x_list.append(x_next)

    return x_list, y_list
