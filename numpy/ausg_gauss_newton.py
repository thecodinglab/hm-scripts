import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#######################
# Definitionen        #
#######################

x, a, b = sp.symbols('x a b')
symbols = sp.Matrix([a, b])

func = a * sp.exp(b * x)

data = np.array([
    [0., 3.],
    [1., 1.],
    [2., 0.5],
    [3., 0.2],
    [4., 0.05],
])

x0 = np.array([2, 2])
n = 10

p_max = 10 # k>1 für gedämpfte Newton-Iteration

#######################
# Gauss-Newton        #
#######################

data_x = data[:, 0]
data_y = data[:, 1]

funcs = [data_y[i] - func.subs(x, data_x[i]) for i in range(data.shape[0])]

f = sp.Matrix(funcs)
df = f.jacobian(symbols)

f_lambda = sp.lambdify(symbols, f, "numpy")
df_lambda = sp.lambdify(symbols, df, "numpy")

coefs = x0
for i in range(n):
  df_val = df_lambda(*coefs)
  q, r = np.linalg.qr(df_val)
  delta = np.linalg.solve(r, -q.T @ f_lambda(*coefs)).flatten()

  p = 0
  k = 1
  while p<=p_max and np.linalg.norm(f_lambda(*(coefs + delta / k))) > np.linalg.norm(f_lambda(*coefs)):
    p += 1
    k = k * 2

  coefs = coefs + delta / k

func_solved = func.subs([(sym, coef) for sym, coef in zip(symbols, coefs)])
func_lambda = sp.lambdify(x, func_solved, "numpy")

#######################
# Plot                #
#######################

x = np.linspace(0, 4, 100)
y = func_lambda(x)

plt.plot(x, y, label='fit')
plt.plot(data_x, data_y, 'o', label='data')
plt.legend()
plt.show()
