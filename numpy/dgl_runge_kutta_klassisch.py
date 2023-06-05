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
# Runge-Kutta Verfahren #
#########################

step_size = (b - a) / n

x = np.linspace(a, b, n + 1)
y = np.zeros(n + 1)
y[0] = y0

for i in range(n):
    k1 = f(x[i], y[i])
    k2 = f(x[i] + step_size / 2, y[i] + step_size / 2 * k1)
    k3 = f(x[i] + step_size / 2, y[i] + step_size / 2 * k2)
    k4 = f(x[i] + step_size, y[i] + step_size * k3)
    y[i + 1] = y[i] + step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

print(y)
