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

d = np.diag(np.diag(a))
l = np.tril(a, -1)
u = np.triu(a, 1)

inv = np.linalg.inv(d)

x = x0
for _ in range(n):
    x = inv @ (b - (l + u) @ x)

#######################
# Ausgabe             #
#######################

print(f'x = {x}')
