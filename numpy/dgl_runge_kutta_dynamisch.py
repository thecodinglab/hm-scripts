import numpy as np

#########################
# Definitionen          #
#########################

a = 2
b = 5
n = 1

y0 = 1
def f(x, y): return x / y


# c_1 | a_11 a_12 ... a_1s
# c_2 | a_21 a_22 ... a_2s
# ... | ...
# c_s | a_s1 a_s2 ... a_ss
# ----|--------------------
#     | b_1  b_2  ... b_s
butcher = np.array([
    [0, 0, 0, 0],
    [1/3, 1/3, 0, 0],
    [2/3, 0, 2/3, 0],
    [0, 1/4, 0, 3/4]
])

s = butcher.shape[0] - 1
c_a = butcher[:-1, 1:]
c_b = butcher[-1, 1:]
c_c = butcher[:-1, 0]

#########################
# Runge-Kutta Verfahren #
#########################

step_size = (b - a) / n

x = [a]
y = [y0]

for i in range(n):
    k = np.zeros(s)
    k[0] = f(x[-1], y[-1])

    for j in range(1, s):
        x_j = x[-1] + c_c[j] * step_size
        y_j = y[-1] + step_size * np.sum(c_a[j, :j] * k[:j])
        k[j] = f(x_j, y_j)

    x.append(x[-1] + step_size)
    y.append(y[-1] + step_size * np.sum(c_b * k))

print(y)
