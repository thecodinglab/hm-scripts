import numpy as np
import IPython.display as dp


def classic(f, a, b, n, y0, precision=4, print_steps=False):
    h = (b - a) / n

    x_list = [a]
    y_list = [y0]

    for i in range(n):
        x = x_list[-1]
        y = y_list[-1]

        k1 = f(x, y)
        k2 = f(x + h / 2, y + h / 2 * k1)
        k3 = f(x + h / 2, y + h / 2 * k2)
        k4 = f(x + h, y + h * k3)

        x_next = x + h
        y_next = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        if print_steps:
            dp.display(dp.Markdown(f"#### Iteration {i + 1}"))

            k1_str = f"k_1 = f(x_{{{i}}}, y_{{{i}}}) = {f(x, y):.{precision}f}"
            k2_str = f"k_2 = f(x_{{{i}}} + \\frac{{h}}{{2}}, y_{{{i}}} + \\frac{{h}}{{2}} \\cdot k_1) = f({x:.{precision}f} + \\frac{{{h}}}{{2}}, {y:.{precision}f} + \\frac{{{h}}}{{2}} \\cdot {k1:.{precision}f}) = {f(x + h/2, y + h/2 * k1):.{precision}f}"
            k3_str = f"k_3 = f(x_{{{i}}} + \\frac{{h}}{{2}}, y_{{{i}}} + \\frac{{h}}{{2}} \\cdot k_2) = f({x:.{precision}f} + \\frac{{{h}}}{{2}}, {y:.{precision}f} + \\frac{{{h}}}{{2}} \\cdot {k2:.{precision}f}) = {f(x + h/2, y + h/2 * k2):.{precision}f}"
            k4_str = f"k_4 = f(x_{{{i}}} + h, y_{{{i}}} + h \\cdot k_3) = f({x:.{precision}f} + {h}, {y:.{precision}f} + {h} \\cdot {k3:.{precision}f}) = {f(x + h, y + h * k3):.{precision}f}"
            dp.display(dp.Math(
                f"\\begin{{aligned}}& {k1_str} \\\\ & {k2_str} \\\\ & {k3_str} \\\\ & {k4_str} \\end{{aligned}}"))

            x_next_str = f"x_{{{i + 1}}} = x_{{{i}}} + h = {x:.{precision}f} + {h} = {x_next:.{precision}f}"
            y_next_str = f"y_{{{i + 1}}} = y_{{{i}}} + \\frac{{h}}{{6}} \\cdot (k_1 + 2 \\cdot k_2 + 2 \\cdot k_3 + k_4) = {y:.{precision}f} + \\frac{{{h}}}{{6}} \\cdot ({k1:.{precision}f} + 2 \\cdot {k2:.{precision}f} + 2 \\cdot {k3:.{precision}f} + {k4:.{precision}f}) = {y_next:.{precision}f}"
            dp.display(dp.Math(
                f"\\begin{{aligned}}& {x_next_str} \\\\ & {y_next_str} \\end{{aligned}}"))

        x_list.append(x_next)
        y_list.append(y_next)

    return x_list, y_list


def dynamic(f, a, b, n, y0,
            intermediate_steps,             # s
            previous_intermediate_weights,  # c_a
            intermediate_weights,           # c_b
            intermediate_step_factors,      # c_c
            precision=4, print_steps=False):
    h = (b - a) / n

    x_list = [a]
    y_list = [y0]

    for i in range(n):
        x = x_list[-1]
        y = y_list[-1]

        k = np.zeros(intermediate_steps)
        k[0] = f(x, y)

        for j in range(1, intermediate_steps):
            k[j] = f(x + intermediate_step_factors[j] * h, y + h *
                     np.sum(previous_intermediate_weights[j, :j] * k[:j]))

        x_next = x + h
        y_next = y + h * np.sum(intermediate_weights * k)

        x_list.append(x_next)
        y_list.append(y_next)

    return x_list, y_list
