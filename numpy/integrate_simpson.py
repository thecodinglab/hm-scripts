import sympy as sp
import numpy as np
import scipy.integrate as integrate

#######################
# Definitionen        #
#######################

a = 0
b = 1

# region variant function 1: numpy
f = lambda x: x ** 2
diff4_f = lambda x: 0
# endregion

# region viariant function 2: sympy
x = sp.Symbol('x')
sp_f = sp.exp(-x ** 2)
sp_diff4_f = sp.diff(sp_f, x, 4)

f = sp.lambdify(x, sp_f, 'numpy')
diff4_f = sp.lambdify(x, sp_diff4_f, 'numpy')
# endregion

# region variant interval 1: given
n = 10
# endregion

# region variant interval 2: max error
max_error = 1e-6

diff4_max = np.max(np.abs(diff4_f(np.linspace(a, b, 1000))))
optimal_width = (2880 * max_error / ((b - a) * diff4_max)) ** (1 / 4)
print(f"optimal width: {optimal_width}")
n = int(np.ceil((b - a) / optimal_width))
# endregion

print(f"n: {n}")
print()

#######################
# Integration         #
#######################

integral = integrate.quad(f, a, b)

width = (b - a) / n
linspace = np.linspace(a, b, n + 1)

f_sum = np.sum(f(linspace[1:-1]))
f_center_sum = np.sum(f(linspace[:-1] + width / 2))
estimation = width / 3 * (0.5 * f(a) + f_sum + 2 * f_center_sum + 0.5 * f(b))

print(f"estimated: {estimation}")
print(f"real:      {integral[0]}")
print()

#######################
# Fehlerabschätzung   #
#######################

diff4_max = np.max(np.abs(diff4_f(linspace)))
max_error = (width ** 4) / 2880 * (b - a) * diff4_max

print(f"max error:  {max_error}")
print(f"real error: {np.abs(integral[0] - estimation)}")
print(f"range:      [{estimation - max_error}, {estimation + max_error}]")
