import numpy as np

#########################
# Definitionen          #
#########################

a = -1.5
b = 1.5
n = 5

y0 = 0
f = lambda x, y: x**2 + 0.1 * y

p = 4

c_a = np.array([
    [0, 0, 0, 0],
    [0.5, 0, 0, 0],
    [0, 0.5, 0, 0],
    [0, 0, 1, 0]
])

c_b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
c_c = np.array([0, 0.5, 0.5, 1])

#########################
# Runge-Kutta Verfahren #
#########################

step_size = (b - a) / n

x = np.linspace(a, b, n + 1)
y = np.zeros(n + 1)
y[0] = y0

for i in range(n):
    k = np.zeros(p)

    k[0] = f(x[i], y[i])
    for j in range(1, p):
        k[j] = f(x[i] + step_size * c_c[j], y[i] +
                 step_size * np.sum(c_a[j, :j] * k[:j]))

    y[i + 1] = y[i] + step_size * np.sum(c_b * k)

print(y)
