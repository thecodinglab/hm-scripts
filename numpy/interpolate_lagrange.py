import numpy as np

##########################
# Definitionen           #
##########################

data = np.array([
    [8., 11.2],
    [10., 13.4],
    [12., 15.3],
    [14., 19.5],
])

t = 11

##########################
# Lagrange Interpolation #
##########################

n = data.shape[0]
x = data[:, 0]
y = data[:, 1]

p = np.zeros(n)

def l(i, t, x):
    return np.prod([(t - x[j]) / (x[i] - x[j]) for j in range(n) if i != j])

for i in range(n):
    p[i] = l(i, t, x) * y[i]

print(np.sum(p))
