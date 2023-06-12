import sympy as sp
import numpy as np
import scipy.integrate as integrate

#######################
# Definitionen        #
#######################

a = 0
b = 1

# region variant function 1: numpy
f = lambda x: np.exp(-x ** 2)
diff2_f = lambda x: (4 * x ** 2 - 2) * np.exp(-x ** 2)
# endregion


# region variant function 2: sympy
x = sp.Symbol('x')
sp_f = sp.exp(-x ** 2)
sp_diff2_f = sp.diff(sp_f, x, 2)

f = sp.lambdify(x, sp_f, 'numpy')
diff2_f = sp.lambdify(x, sp_diff2_f, 'numpy')
# endregion

# region variant interval 1: given
n = 10
# endregion

# region variant interval 2: max error
max_error = 1e-6

diff2_max = np.max(np.abs(diff2_f(np.linspace(a, b, 1000))))
optimal_width = np.sqrt(24 * max_error / ((b - a) * diff2_max))
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
estimation = width * np.sum(f(linspace[:-1] + width / 2))

print(f"estimated: {estimation}")
print(f"real:      {integral[0]}")
print()

#######################
# Fehlerabschätzung   #
#######################

diff2_max = np.max(np.abs(diff2_f(linspace)))
max_error = (width ** 2) / 24 * (b - a) * diff2_max

print(f"max error:  {max_error}")
print(f"real error: {np.abs(integral[0] - estimation)}")
print(f"range:      [{estimation - max_error}, {estimation + max_error}]")
