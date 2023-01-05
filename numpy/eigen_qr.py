import numpy as np

######################
# Definitionen       #
######################

a = np.array([
    [1, -2, 0],
    [2, 0, 1],
    [0, -2, 1],
], dtype=np.float64)

n = 100

precision = 1e-10

######################
# QR Verfahren       #
######################

a_iter = a.copy()
p_iter = np.eye(a.shape[0])

for i in range(n):
    q, r = np.linalg.qr(a_iter)
    a_iter = r @ q
    p_iter = p_iter @ q

if (a.T == a).all():
    # symmetric matrix
    eigenvalues = np.diag(a_iter)
    eigenvectors = p_iter

else:
    # non-symmetric matrix
    eigenvalues = []
    for k in range(a.shape[0]):
        if not (k == 0 or abs(a_iter[k, k - 1]) <= precision):
            # entry to the left is not zero
            continue

        if k == a.shape[0] - 1 or abs(a_iter[k + 1, k]) <= precision:
            # entry below is zero
            eigenvalues.append(a_iter[k, k])
            continue

        # extract 2x2 block to find eigenvalues of quadratic polynomial
        block = a_iter[k:k + 2, k:k + 2]
        poly = [1, -block[0, 0] - block[1, 1], block[0, 0] * block[1, 1] - block[0, 1] * block[1, 0]]
        roots = np.roots(poly)
        eigenvalues.extend(roots)

######################
# Ausgabe            #
######################

print(eigenvalues)
