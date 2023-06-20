import numpy as np
import matplotlib.pyplot as plt

##############################
# Definitionen               #
##############################

plot = True
plot_range = np.linspace(0, 4, 100)

##############################
# f(x) = ax + b              #
#   -> f1(x) = x             #
#   -> f2(x) = 1             #
##############################

data = np.array([
    [1., 6.],
    [2., 6.8],
    [3., 10.],
    [4., 10.5],
])

funcs = [
    lambda x: x,
    lambda x: np.ones(len(x)),
]

lin_x = lambda x: x
lin_y = lambda y: y

def evaluate(l, x):
    return l[0] * x + l[1]

##################################################################
# f(x) = ae^{bx}                                                 #
# log(f(x)) = log(ae^{bx}) = log(a) + log(e^{bx}) = log(a) + bx  #
#   -> f1(x) = 1                                                 #
#   -> f2(x) = x                                                 #
##################################################################

# data = np.array([
#     [0., 3.],
#     [1., 1.],
#     [2., 0.5],
#     [3., 0.2],
#     [4., 0.05],
# ])

# funcs = [
#     lambda x: [1 for _ in x],
#     lambda x: x,
# ]

# lin_x = lambda x: x
# lin_y = lambda y: np.log(y)

# def evaluate(l, x):
#     return np.exp(l[0]) * np.exp(l[1] * x)

####################################
# f(x) = ae^x + b                  #
# f(x) = a * e^ln(x) + b = ax + b  #
#   -> f1(x) = x                   #
#   -> f2(x) = 1                   #
####################################
# TODO: something is wrong here

# data = np.array([
#     # [0., 6.],
#     [1., 12.],
#     [2., 30.],
#     [3., 80.],
#     [4., 140.],
# ], dtype=np.float64)

# funcs = [
#     lambda x: x,
#     lambda x: [1 for _ in x],
# ]

# lin_x = lambda x: np.log(x)
# lin_y = lambda y: y

# def evaluate(l, x):
#     return l[0] * np.exp(x) + l[1]

##############################
# Lineare Ausgleichsrechnung #
##############################

def solve_gauss(A, b):
    return np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

def solve_qr(A, b):
    q, r = np.linalg.qr(A)
    rl = np.dot(q.T, b)
    return np.linalg.solve(r, rl)

solvers = {
    "gauss": solve_gauss,
    "qr": solve_qr,
}

x = data[:, 0]
y = data[:, 1]

A = np.column_stack([ f(lin_x(x)) for f in funcs ])

eq = A.T @ A
res = A.T @ lin_y(y)

for name, solver in solvers.items():
    print(f"solver: {name}")

    l = solver(eq, res)
    print(f"  lambda = {l}")

    err = np.linalg.norm(np.dot(A, l) - lin_y(y))
    print(f"  error = {err}")
    print()

    if plot:
        plt.plot(plot_range, evaluate(l, plot_range), label=name)

##############################
# Plot                       #
##############################

if plot:
    plt.plot(data[:, 0], data[:, 1], "o", label="data")
    plt.legend()
    plt.show()
