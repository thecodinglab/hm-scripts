import numpy as np
import matplotlib.pyplot as plt

############################################
# Definitions                              #
############################################

f = lambda x: 1. / (np.cos(x + np.pi / 4) - 1) + 2
df = lambda x: np.sin(x + np.pi / 4) / (np.cos(x + np.pi / 4) - 1) ** 2

x0 = 1
x1 = 1.1
n = 30

############################################
# Sekanten-Verfahren                       #
############################################

x_prev = x0
x_curr = x1

for i in range(n):
    x = x_curr - f(x_curr) * (x_curr - x_prev) / (f(x_curr) - f(x_prev))
    x_prev = x_curr
    x_curr = x

    print(f'x_{i + 1} = {x}')
