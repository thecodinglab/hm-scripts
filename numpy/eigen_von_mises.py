import numpy as np

#######################
# Definitionen        #
#######################

a = np.array([
    [1, 1, 0],
    [3, -1, 2],
    [2, -1, 3],
], dtype=np.float64)

x0 = np.array([1, 0, 0], dtype=np.float64)

n = 9

#######################
# Von Mises Iteration #
#######################

v = x0 / np.linalg.norm(x0, ord=2)

for i in range(n):
    v = a @ v
    v = v / np.linalg.norm(v, ord=2)
    print(f'v_{i + 1} = {v}, λ_{i + 1} = {(v.T @ a @ v) / (v.T @ v)}')

#######################
# Ausgabe             #
#######################

print('')
print(f'v = {v}, λ = {(v.T @ a @ v) / (v.T @ v)}')
