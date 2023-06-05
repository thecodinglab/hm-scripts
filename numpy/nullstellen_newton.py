import numpy as np

############################################
# Definitions                              #
############################################

f = lambda x: 1. / (np.cos(x + np.pi / 4) - 1) + 2
df = lambda x: np.sin(x + np.pi / 4) / (np.cos(x + np.pi / 4) - 1) ** 2

x0 = 1
n = 100

############################################
# Newton-Verfahren                         #
############################################

x = x0
for i in range(n):
    x -= f(x) / df(x)
    print(f'x_{i + 1} = {x}')
