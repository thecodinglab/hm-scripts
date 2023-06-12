import IPython.display as dp


def classic(f, a, b, n, y0, precision=4, print_steps=False):
    h = (b - a) / n

    x_list = [a]
    y_list = [y0]

    for i in range(n):
        x = x_list[-1]
        y = y_list[-1]

        y_next = y + h * f(x, y)
        x_next = x + h

        if print_steps:
            x_next_str = f"x_{{{i + 1}}} = x_{{{i}}} + h = {x:.{precision}f} + {h} = {x_next:.{precision}f}"
            y_next_str = f"y_{{{i + 1}}} = y_{{{i}}} + h \\cdot f(x_{{{i}}}, y_{{{i}}}) = {y:.{precision}f} + {h} \\cdot {f(x, y):.{precision}f} = {y_next:.{precision}f}"
            dp.display(dp.Math(x_next_str + " \\quad " + y_next_str))

        x_list.append(x_next)
        y_list.append(y_next)

    return x_list, y_list


def midpoint(f, a, b, n, y0, precision=4, print_steps=False):
    h = (b - a) / n

    x_list = [a]
    y_list = [y0]

    for i in range(n):
        x = x_list[-1]
        y = y_list[-1]

        x_mid = x + h / 2
        y_mid = y + h / 2 * f(x, y)

        if print_steps:
            x_mid_str = f"x_{{{i + 0.5}}} = x_{{{i}}} + \\frac{{h}}{{2}} = {x:.{precision}f} + \\frac{{{h}}}{{2}} = {x_mid:.{precision}f}"
            y_mid_str = f"y_{{{i + 0.5}}} = y_{{{i}}} + \\frac{{h}}{{2}} \\cdot f(x_{{{i}}}, y_{{{i}}}) = {y:.{precision}f} + \\frac{{{h}}}{{2}} \\cdot {f(x, y):.{precision}f} = {y_mid:.{precision}f}"
            dp.display(dp.Math(x_mid_str + " \\quad " + y_mid_str))

        y_next = y + h * f(x_mid, y_mid)
        x_next = x + h

        if print_steps:
            x_next_str = f"x_{{{i + 1}}} = x_{{{i}}} + h = {x:.{precision}f} + {h} = {x_next:.{precision}f}"
            y_next_str = f"y_{{{i + 1}}} = y_{{{i}}} + h \\cdot f(x_{{{i + 0.5}}}, y_{{{i + 0.5}}}) = {y:.{precision}f} + {h} \\cdot f({x_mid:.{precision}f}, {y_mid:.{precision}f}) = {y:.{precision}f} + {h} \\cdot {f(x_mid, y_mid):.{precision}f} = {y_next:.{precision}f}"
            dp.display(dp.Math(x_next_str + " \\quad " + y_next_str))

        y_list.append(y_next)
        x_list.append(x_next)

    return x_list, y_list


def modified(f, a, b, n, y0, precision=4, print_steps=False):
    h = (b - a) / n

    x_list = [a]
    y_list = [y0]

    for i in range(n):
        x = x_list[-1]
        y = y_list[-1]

        k1 = f(x, y)
        k2 = f(x + h, y + h * k1)

        if print_steps:
            k1_str = f"k_1 = f(x_{{{i}}}, y_{{{i}}}) = {f(x, y):.{precision}f}"
            k2_str = f"k_2 = f(x_{{{i}}} + h, y_{{{i}}} + h \\cdot k_1) = f({x:.{precision}f} + {h}, {y:.{precision}f} + {h} \\cdot {k1:.{precision}f}) = {f(x + h, y + h * k1):.{precision}f}"
            dp.display(dp.Math(k1_str + " \\quad " + k2_str))

        y_next = y + h * (k1 + k2) / 2
        x_next = x + h

        if print_steps:
            x_next_str = f"x_{{{i + 1}}} = x_{{{i}}} + h = {x:.{precision}f} + {h} = {x_next:.{precision}f}"
            y_next_str = f"y_{{{i + 1}}} = y_{{{i}}} + \\frac{{h}}{{2}} \\cdot (k_1 + k_2) = {y:.{precision}f} + \\frac{{{h}}}{{2}} \\cdot ({k1:.{precision}f} + {k2:.{precision}f}) = {y_next:.{precision}f}"
            dp.display(dp.Math(x_next_str + " \\quad " + y_next_str))

        y_list.append(y_next)
        x_list.append(x_next)

    return x_list, y_list
