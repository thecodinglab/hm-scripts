import numpy as np
import sympy as sp
import IPython.display as dp
import matplotlib.pyplot as plt

def interpolate_least_squares(exponents: list[int], x, y) -> sp.Expr:
    """
    Interpolates the specified `x` and `y` values by the least squares with the specified `exponents`.

    Attributes
    ----------
    exponents : list[int]
        The exponents to interpolate the coefficients for.
    x : Any
        The x values of the dataset.
    y : Any
        The y values of the dataset.
    """
    def power(pow: int):
        return lambda x: x ** pow

    def least_squares(f, x, y):
        sum = 0
        for i, _ in enumerate(x):
            sum += (y[i] - f(x[i])) ** 2
        return sum
    
    def normal(A, y):
        Q, R = np.linalg.qr(A)
        return np.linalg.solve(R, Q.T @ y)
    
    def spline(funcs, x):
        A = np.zeros((len(x), len(funcs)))
        for i, x_i in enumerate(x):
            for j, f_j in enumerate(funcs):
                A[i, j] = f_j(x_i)
        return A
    
    funcs = [power(exponent) for exponent in exponents]
    A = spline(funcs, x)
    dp.display(dp.Math(f"A = {sp.latex(sp.Matrix(A))}"))

    lam = normal(A, y)
    dp.display(dp.Math(f"\\lambda = {sp.latex(sp.Matrix(lam))}"))

    lam = [(lam[exponents.index(i)] if i in exponents else 0) for i in range(0, max(exponents) + 1)]
    lam = np.array(list(reversed(lam)))
    coefficients = np.polyval(lam, x)
    plt.plot(x, coefficients, label="Estimate")
    plt.scatter(x, y, label="Data")
    plt.legend()
    plt.grid()

    expression = sp.Poly(lam, sp.Symbol("x")).as_expr()
    dp.display(expression)

    err = least_squares(lambda x: np.polyval(lam, x), x, y)
    dp.display(dp.Math(f"\\text{{Error}} = {sp.latex(err)}"))
    return expression
    