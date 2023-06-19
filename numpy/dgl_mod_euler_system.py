import numpy as np

#########################
# Definitionen          #
#########################

n = 5
x0 = 0
y0 = np.array([0, 0])
step_size = 0.01


def f(x, y): return np.array([y[1],
                              -80*y[1] - 1/(10**-4)*y[0]+100])

#########################
# Modifiziertes Euler   #
#########################


x = np.arange(x0, n, step_size)
y = [y0]

for i in range(n):
    k1 = f(x[i], y[i])
    k2 = f(x[i] + step_size, y[i] + step_size * k1)
    y.append(y[i] + step_size / 2 * (k1 + k2))

y = np.array(y)
print(y)
