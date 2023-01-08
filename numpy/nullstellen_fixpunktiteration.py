import numpy as np
import matplotlib.pyplot as plt

############################################
# Definitions                              #
############################################

# in the form x = F(x)
f = lambda x: 1. / (np.cos(x + np.pi / 4) - 1) + 2
df = lambda x: np.sin(x + np.pi / 4) / (np.cos(x + np.pi / 4) - 1) ** 2

range_a = 1
range_b = 2

plot_range_from = 0
plot_range_to = np.pi

tolerance = 1e-6
x0 = 1

############################################
# Visualization                            #
############################################

x = np.linspace(plot_range_from, plot_range_to, 1000)
y = f(x)
dy = df(x)

plt.figure(1)
plt.plot(x, x, label='y = x')
plt.plot(x, y, label='F(x)')
plt.plot(x, dy, ':', label='F\'(x)')
plt.legend()
plt.show()

############################################
# Banach fixed point theorem               #
############################################

range_y = f(np.linspace(range_a, range_b, 1000))
range_y_min = np.min(range_y)
range_y_max = np.max(range_y)

if range_y_min < min(range_a, range_b) or range_y_max > max(range_a, range_b):
    print('Banach fixed point theorem does not hold. Range of F is not contained in [a, b].')

# Lipschitz constant
range_dy = df(np.linspace(range_a, range_b, 1000))
lipschitz = np.max(np.abs(range_dy))
print(f'Lipschitz constant: {lipschitz}')

if lipschitz < tolerance or lipschitz > 1:  # 0 < alpha < 1
    print('Banach fixed point theorem does not hold. Lipschitz constant is too large.')

# Check if fixed point iteration converges
df0 = abs(df(x0))
if df0 >= 1:
    print(f'Fixed point iteration does not converge. f\'(x0) = {df0} >= 1')


############################################
# Fixpoint iteration                       #
############################################

def err_a_priori(x0, x1, alpha, n):
    return (alpha ** n) * abs(x1 - x0) / (1 - alpha)


def err_a_posteriori(x_prev, x_curr, alpha):
    return alpha * abs(x_curr - x_prev) / (1 - alpha)


def estimate_number_of_steps(tol, x0, x1, alpha):
    # tol <= ((alpha^n) / (1 - alpha)) * | x_1 - x_0 |
    # tol * (1 - alpha) <= (alpha^n) * | x_1 - x_0 |
    # (tol * (1 - alpha)) / | x_1 - x_0 | <= alpha^n
    # log((tol * (1 - alpha)) / | x_1 - x_0 |) / log(alpha) <= n
    return np.log((tol * (1 - alpha)) / abs(x1 - x0)) / np.log(alpha)


def fixpoint_iteration(f, x0, tol, alpha, max_iter=100):
    x_prev = x0
    x_curr = f(x_prev)

    n = 1
    while err_a_posteriori(x_prev, x_curr, alpha) > tol and n < max_iter:
        x_prev = x_curr
        x_curr = f(x_prev)
        n += 1

    return x_curr, n


x_fixed_point, n = fixpoint_iteration(f, x0, tolerance, lipschitz)
print(f'Fixed point: {x_fixed_point}, after {n} iterations')
