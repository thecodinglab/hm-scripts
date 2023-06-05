import numpy as np

# NOTE: set suppress=False to print scientific notation for small numbers
np.set_printoptions(precision=6, suppress=True, floatmode='fixed')

#######################
# Definitionen        #
#######################

funcs = [
    lambda x: 2 * x[0] + 4 * x[1],
    lambda x: 4 * x[0] + 8 * x[1] ** 3
]

jacobi = [
    lambda _: [2, 4],
    lambda x: [4, 24 * x[1] ** 2]
]

x0 = np.array([4, 2])
k_max = 10

term = 'rel_diff'  # 'max_iter', 'rel_diff', 'abs_diff', 'value'
term_max = 1e-6

# TODO: Das Newton-Verfahren konvergiert quadratisch, wenn die Jacobi-Matrix
# invertierbar ist und die Funktionen dreimal stetig differenzierbar sind.

#######################
# Newton Verfahren    #
#######################


def step(i, x, funcs, jacobi, k_max):
    f_x = np.array([
        f(x) for f in funcs
    ])

    df_x = np.array([
        df(x) for df in jacobi
    ])

    # TODO: ability to choose solver?
    gamma_x = np.linalg.solve(df_x, -f_x)

    # TODO: test this
    k_div = 1

    x_k = x
    f_x_k = f_x

    for k in range(k_max):
        x_k = x + gamma_x / k_div
        f_x_k = np.array([
            f(x_k) for f in funcs
        ])

        if np.linalg.norm(f_x_k) < np.linalg.norm(f_x):
            break

        k_div *= 2

    print(f"iteration {i + 1}: x = {x_k}, f(x) = {f_x_k}")
    return x_k


termination_criteria = {
    'max_iter': lambda args: args['i'] >= args['max'],
    'rel_diff': lambda args: np.linalg.norm(args['x'] - args['x_prev']) / np.linalg.norm(args['x']) < args['max'],
    'abs_diff': lambda args: np.linalg.norm(args['x'] - args['x_prev']) < args['max'],
    'value': lambda args: np.linalg.norm(args['f_x']) < args['max'],
}

criterion = termination_criteria[term]

x_prev = x0
i = 0

x = step(i, x0, funcs, jacobi, k_max)

while not criterion({'i': i, 'max': term_max, 'x': x, 'x_prev': x_prev, 'f_x': np.array([f(x) for f in funcs])}):
    x_prev = x
    x = step(i, x, funcs, jacobi, k_max)
    i += 1
