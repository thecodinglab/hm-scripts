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

dx0 = df(x0)
print(dx0)

x = x0
for i in range(n):
    x -= f(x) / dx0
    print(f'x_{i + 1} = {x}')
