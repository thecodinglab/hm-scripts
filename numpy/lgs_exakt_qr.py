import numpy as np

#######################
# Definitionen        #
#######################

a = np.array([
    [1, 1, 0],
    [3, -1, 2],
    [2, -1, 3],
], dtype=np.float64)

b = np.array([1, 1, 0], dtype=np.float64)

#######################
# LGS lösen           #
#######################

q, r = np.linalg.qr(a)

y = np.transpose(q) @ b
x = np.linalg.solve(r, y)

#######################
# Ausgabe             #
#######################

print(f'x = {x}')
