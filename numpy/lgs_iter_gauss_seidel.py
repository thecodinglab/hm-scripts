import numpy as np

#######################
# Definitionen        #
#######################

a = np.array([
    [4, -1, 1],
    [-2, 5, 1],
    [1, -2, 5],
], dtype=np.float64)

b = np.array([5, 11, 12], dtype=np.float64)
x0 = np.array([0, 0, 0], dtype=np.float64)

n = 5

#######################
# LGS iterativ lösen  #
#######################

s = a.shape

x = x0
for _ in range(n):
    for i in range(s[0]):
        x[i] = (1 / a[i, i]) * (b[i] - np.sum(a[i, :i] * x[:i]) - np.sum(a[i, i + 1:] * x[i + 1:]))

#######################
# Ausgabe             #
#######################

print(f'x = {x}')
