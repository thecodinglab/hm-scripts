import IPython.display as dp
import sympy as sp


def normalize(x: sp.Matrix) -> sp.Matrix:
    return x.normalized()


def von_mises_iter(a: sp.Matrix, v: sp.Matrix, precision: int = -1) -> [sp.Matrix, sp.Number]:
    next_eigen_vec = (a @ v).normalized()
    next_eigen_val = (v.T.dot(a @ v)) / (v.T.dot(v))

    if precision != -1:
        next_eigen_vec = next_eigen_vec.evalf(precision)
        next_eigen_val = next_eigen_val.evalf(precision)

    return next_eigen_vec, next_eigen_val


def von_mises(a: sp.Matrix, v0: sp.Matrix, iterations: int = 100, precision: int = -1) -> [sp.Matrix, sp.Number]:
    eigen_vec = v0
    eigen_val = sp.Number(0)

    for i in range(iterations):
        eigen_vec, eigen_val = von_mises_iter(a, eigen_vec, precision)
        dp.display(
            dp.Math(f'v_{{ {i + 1} }} = {sp.latex(eigen_vec)}, \\; \\lambda_{{ {i + 1} }} = {sp.latex(eigen_val)}'))

    dp.display(dp.Math(f'\\text{{Eigenvektor }} v \\approx {sp.latex(eigen_vec)}'))
    dp.display(dp.Math(f'\\text{{Eigenwert }} \\lambda \\approx {sp.latex(eigen_val)}'))
    return eigen_vec, eigen_val
