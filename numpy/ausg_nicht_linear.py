import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

####################################
# Definitionen                     #
####################################

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

x0 = np.array([3, -1])

err_max = 1e-5
n = 100

####################################
# Nicht-Lineares Ausgleichsproblem #
####################################

data_x = data[:, 0]
data_y = data[:, 1]

err_func = np.sum([
  (data_y[i] - func.subs(x, data_x[i]))**2 for i in range(data.shape[0])    
])

d_err_funcs = [ sp.diff(err_func, sym) for sym in symbols ]

f = sp.Matrix(d_err_funcs)
df = f.jacobian(symbols)

f_lambda = sp.lambdify(symbols, f, "numpy")
df_lambda = sp.lambdify(symbols, df, "numpy")

####################################
# Newton-Verfahren                 #
####################################

coefs = x0
for i in range(n):
  f_val = f_lambda(*coefs)
  df_val = df_lambda(*coefs)

  delta = np.linalg.solve(df_val, -f_val).flatten()
  coefs = coefs + delta

  err = np.linalg.norm(f_lambda(*coefs))
  if err < err_max:
    break

func_solved = func.subs([(sym, coef) for sym, coef in zip(symbols, coefs)])
func_lambda = sp.lambdify(x, func_solved, "numpy")

####################################
# Plot                             #
####################################

x_vals = np.linspace(0, 4, 100)
y_vals = func_lambda(x_vals)

plt.plot(x_vals, y_vals, label="Fit")
plt.plot(data_x, data_y, "o", label="Daten")
plt.legend()
plt.show()
