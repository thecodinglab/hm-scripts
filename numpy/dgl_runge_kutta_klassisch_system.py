import numpy as np
import matplotlib.pyplot as plt

#########################
# Definitionen          #
#########################

a = 0
b = 8
n = int((b-a)/0.05)

y0 = np.array([20, 0])


#def f(x, y): return np.array([y[1], -0.1*y[1]*np.abs(y[1])-10])
def f(x, y):
    if y[0] < 0 and y[1] < 0:
        y[1] = -y[1]
    return np.array([y[1], (-0.1*y[1]*np.abs(y[1])-10)])

#########################
# Runge-Kutta Verfahren #
#########################


step_size = (b - a) / n

x = np.linspace(a, b, n + 1)
y = [y0]

for i in range(n):
    k1 = f(x[i], y[i])
    k2 = f(x[i] + step_size / 2, y[i] + step_size / 2 * k1)
    k3 = f(x[i] + step_size / 2, y[i] + step_size / 2 * k2)
    k4 = f(x[i] + step_size, y[i] + step_size * k3)
    y.append(y[i] + step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

y = np.array(y)
print(y)

plt.plot(x, y, label=["x", "x'"])
plt.legend()
plt.show()
