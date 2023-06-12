import sympy as sp
import numpy as np
import scipy.integrate as integrate

#######################
# Definitionen        #
#######################

a = 0
b = 0.5
n = 3

# region variant function 1: numpy
f = lambda x: np.exp(-x ** 2)
# endregion


# region viariant function 2: sympy
x = sp.Symbol('x')
sp_f = sp.exp(-x ** 2)
f = sp.lambdify(x, sp_f, 'numpy')
# endregion

#######################
# Integration         #
#######################

integral = integrate.quad(f, a, b)


def gauss_1(f, a, b):
    return (b - a) * f((a + b) / 2)


def gauss_2(f, a, b):
    w = (a + b) / 2
    t = 1 / np.sqrt(3) * (b - a) / 2
    return (b - a) / 2 * (f(-t + w) + f(t + w))


def gauss_3(f, a, b):
    w = (a + b) / 2
    t = np.sqrt(0.6) * ((b - a) / 2)

    return ((b - a) / 2) * ((5 / 9) * f(-t + w) + (8 / 9) * f(w) + (5 / 9) * f(t + w))


lookup = {
    1: gauss_1,
    2: gauss_2,
    3: gauss_3
}

estimation = lookup[n](f, a, b)

print(f"estimated: {estimation}")
print(f"real:      {integral[0]}")
print(f"error:     {np.abs(integral[0] - estimation)}")
