import numpy as np
import sympy as sp
import IPython.display as dp


def classic(f, a, b, n, y0, print_steps=False):
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

            k1_str = f"k_1 = f(x_{{{i}}}, y_{{{i}}}) = {sp.latex(f(x, y))}"
            k2_str = f"k_2 = f(x_{{{i}}} + \\frac{{h}}{{2}}, y_{{{i}}} + \\frac{{h}}{{2}} \\cdot k_1) = f({sp.latex(x)} + \\frac{{{h}}}{{2}}, {sp.latex(y)} + \\frac{{{h}}}{{2}} \\cdot {sp.latex(k1)}) = {sp.latex(f(x + h/2, y + h/2 * k1))}"
            k3_str = f"k_3 = f(x_{{{i}}} + \\frac{{h}}{{2}}, y_{{{i}}} + \\frac{{h}}{{2}} \\cdot k_2) = f({sp.latex(x)} + \\frac{{{h}}}{{2}}, {sp.latex(y)} + \\frac{{{h}}}{{2}} \\cdot {sp.latex(k2)}) = {sp.latex(f(x + h/2, y + h/2 * k2))}"
            k4_str = f"k_4 = f(x_{{{i}}} + h, y_{{{i}}} + h \\cdot k_3) = f({sp.latex(x)} + {h}, {sp.latex(y)} + {h} \\cdot {sp.latex(k3)}) = {sp.latex(f(x + h, y + h * k3))}"
            dp.display(dp.Math(
                f"\\begin{{aligned}}& {k1_str} \\\\ & {k2_str} \\\\ & {k3_str} \\\\ & {k4_str} \\end{{aligned}}"))

            x_next_str = f"x_{{{i + 1}}} = x_{{{i}}} + h = {sp.latex(x)} + {h} = {sp.latex(x_next)}"
            y_next_str = f"y_{{{i + 1}}} = y_{{{i}}} + \\frac{{h}}{{6}} \\cdot (k_1 + 2 \\cdot k_2 + 2 \\cdot k_3 + k_4) = {sp.latex(y)} + \\frac{{{h}}}{{6}} \\cdot ({sp.latex(k1)} + 2 \\cdot {sp.latex(k2)} + 2 \\cdot {sp.latex(k3)} + {sp.latex(k4)}) = {sp.latex(y_next)}"
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
        if print_steps:
            dp.display(dp.Markdown(f"#### Iteration {i + 1}"))
            k0_str = f"k_0 = f(x_{{{i}}}, y_{{{i}}}) = f({x:.{precision}f}, {y:.{precision}f}) = {f(x, y):.{precision}f}"
            dp.display(dp.Math(k0_str))

        for j in range(1, intermediate_steps):
            x_j = x + intermediate_step_factors[j] * h
            y_j = y + h * np.sum(previous_intermediate_weights[j, :j] * k[:j])
            k[j] = f(x_j, y_j)
            
            if print_steps:
                sum_str = '\\left['
                for m in range(j):
                    sum_str += f"{previous_intermediate_weights[j, m]:.{precision}f} \\cdot {k[m]:.{precision}f}"
                    if m != j - 1:
                        sum_str += ' + '
                sum_str += '\\right]'

                k_str = f"k_{{{j}}} = f(x_{{{i}}} + c_{{{j}}} \\cdot h, y_{{{i}}} + h \\cdot \\sum_{{m = 0}}^{{{j - 1}}} c_{{m, {j}}} \\cdot k_m)"
                k_str += f" = f({x:.{precision}f} + {intermediate_step_factors[j]} \\cdot {h}, {y:.{precision}f} + {h} \\cdot {sum_str})"
                k_str += f" = f({x_j:.{precision}f}, {y_j:.{precision}f}) = {k[j]:.{precision}f}"
                dp.display(dp.Math(k_str))

        x_next = x + h
        y_next = y + h * np.sum(intermediate_weights * k)

        if print_steps:
            sum_str = '\\left['
            for m in range(intermediate_steps):
                sum_str += f"{intermediate_weights[m]:.{precision}f} \\cdot {k[m]:.{precision}f}"
                if m != intermediate_steps - 1:
                    sum_str += ' + '
            sum_str += '\\right]'

            x_next_str = f"x_{{{i + 1}}} = x_{{{i}}} + h = {x:.{precision}f} + {h} = {x_next:.{precision}f}"
            y_next_str = f"y_{{{i + 1}}} = y_{{{i}}} + h \\cdot \\sum_{{m = 0}}^{{{intermediate_steps - 1}}} b_m \\cdot k_m = {y:.{precision}f} + {h} \\cdot {sum_str} = {y_next:.{precision}f}"
            dp.display(dp.Math(
                f"\\begin{{aligned}}& {x_next_str} \\\\ & {y_next_str} \\end{{aligned}}"))

        x_list.append(x_next)
        y_list.append(y_next)

    return x_list, y_list
