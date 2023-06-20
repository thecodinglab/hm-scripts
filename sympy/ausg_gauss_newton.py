import numpy as np
import sympy as sp
import IPython.display as dp

def gauss_newton(data: sp.Matrix, f: sp.Function, symbols: sp.Matrix, x0: sp.Matrix, n: int, print_steps = True):
  x = data[:, 0]
  y = data[:, 1]

  funcs = [y[i] - f(x[i]) for i in range(data.shape[0])]

  f = sp.Matrix(funcs)
  df = f.jacobian(symbols)

  dp.display(dp.Math(f"f(x) = {sp.latex(f)}"))
  dp.display(dp.Math(f"Df(x) = {sp.latex(df)}"))

  f_lambda = sp.lambdify(symbols, f, "numpy")
  df_lambda = sp.lambdify(symbols, df, "numpy")

  x = x0
  for i in range(n):
    df_val = df_lambda(*x)
    q, r = np.linalg.qr(df_val)
    delta = np.linalg.solve(r, -q.T @ f_lambda(*x)).flatten()
    x = x + delta

    if print_steps:
      dp.display(dp.Markdown(f"Iteration {i + 1}"))
      dp.display(dp.Math(f"x_{{{i}}} = {sp.latex(sp.Matrix(x))}"))
      dp.display(dp.Math(f"f(x_{{{i}}}) = {sp.latex(sp.Matrix(f_lambda(*x)))}"))
      dp.display(dp.Math(f"Df(x_{{{i}}}) = {sp.latex(sp.Matrix(df_lambda(*x)))} \\; \\Rightarrow \\; Q_{{{i + 1}}} R_{{{i + 1}}}"))

      dp.display(dp.Math(f"Q_{{{i + 1}}} = {sp.latex(sp.Matrix(q))} \\quad R_{{{i + 1}}} = {sp.latex(sp.Matrix(r))}"))
      dp.display(dp.Math(f"R_{{{i + 1}}} \\delta_{{{i + 1}}} = -Q_{{{i + 1}}}^T f(x_{{{i + 1}}}) \\; \\Rightarrow \\; {sp.latex(sp.Matrix(r))} \\delta_{{{i + 1}}} = -{sp.latex(sp.Matrix(q.T))} {sp.latex(sp.Matrix(f_lambda(*x)))} = {sp.latex(sp.Matrix(-q.T @ f_lambda(*x)))}"))
      dp.display(dp.Math(f"\\delta_{{{i + 1}}} = {sp.latex(sp.Matrix(delta))}"))
      dp.display(dp.Math(f"x_{{{i + 1}}} = x_{{{i}}} + \\delta_{{{i + 1}}} = {sp.latex(sp.Matrix(x))} + {sp.latex(sp.Matrix(delta))} = {sp.latex(sp.Matrix(x + delta))}"))

  return x
