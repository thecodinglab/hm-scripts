import numpy as np
import scipy.integrate as integrate

#######################
# Definitionen        #
#######################

a = 0
b = 0.5
n = 3

f = lambda x: np.exp(-x ** 2)

#######################
# Integration         #
#######################

integral = integrate.quad(f, a, b)

T = np.zeros((n + 1, n + 1))

for j in range(n + 1):
    h_j = (b - a) / (2 ** j)

    s = 0
    for i in range(1, 2 ** j):
        s += f(a + i * h_j)

    T[j, 0] = h_j * (0.5 * (f(a) + f(b)) + s)

    for k in range(1, j + 1):
        T[j, k] = T[j, k - 1] + (T[j, k - 1] - T[j - 1, k - 1]) / (4 ** k - 1)

estimation = T[n, n]
print(T)

print(f"estimated: {estimation}")
print(f"real:      {integral[0]}")
print(f"error:     {np.abs(integral[0] - estimation)}")
