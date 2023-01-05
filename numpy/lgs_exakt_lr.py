import numpy as np
import scipy as sp

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

p, l, u = sp.linalg.lu(a)

y = sp.linalg.solve_triangular(l, p @ b, lower=True)
x = sp.linalg.solve_triangular(u, y, lower=False)

#######################
# Ausgabe             #
#######################

print(f'x = {x}')
