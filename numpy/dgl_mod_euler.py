import numpy as np

#########################
# Definitionen          #
#########################

a = -1.5
b = 1.5
n = 5

y0 = 0
f = lambda x, y: x**2 + 0.1 * y

#########################
# Modifiziertes Euler   #
#########################

step_size = (b - a) / n

x = np.linspace(a, b, n + 1)
y = np.zeros(n + 1)
y[0] = y0

for i in range(n):
    k1 = f(x[i], y[i])
    k2 = f(x[i] + step_size, y[i] + step_size * k1)
    y[i + 1] = y[i] + step_size / 2 * (k1 + k2)

print(y)
