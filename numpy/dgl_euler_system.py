import matplotlib.pyplot as plt
import numpy as np

#######################
# Definitionen        #
#######################

n = 5
x0 = 0
y0 = np.array([0, 0])
step_size = 0.01


def f(x, y): return np.array([y[1], -80*y[1] - 1/(10**-4)*y[0]+100])

#######################
# Euler Verfahren     #
#######################


x = np.arange(x0, n, step_size)
y = [y0]

for i in range(n):
    y.append(y[i] + step_size * f(x[i], y[i]))

y = np.array(y)
print(y)
