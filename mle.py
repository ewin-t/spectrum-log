import numpy as np

def vander(x):
    # Outpute the Vandermonde matrix
    n = len(x)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i,j] = x[j] ** (n - 1 - i)
    return M

def schur_unequal(l, x):
    # Computes the schur polynomial s_l(x)
    n = len(l)
    # n is the number of rows in lambda
    # d is the weight of lambda
    Mvander = vander(x)
    # The columns are of the form x_i^k through x_i^0
    Mnum = Mvander.copy()
    # Mnum[i, j] = x[j]^(l[i] + n - 1 - i)
    for i in range(n):
        for j in range(n):
            Mnum[i,j] *= x[j] ** l[i]
    return np.linalg.det(Mnum) / np.linalg.det(Mvander)

def schur(l, x):
    # schur_unequal fails when the vandermonde determinant is zero.
    # perturb so that this is not the case
    tol = 0.0001
    val = x[0] + 1
    for i in range(len(x)):
        if(x[i] == val):
            x[i] = x[i-1] + tol
        else:
            val = x[i]
    print(x)
    return schur_unequal(l, x)

# Copied from https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
def partitions(n, l, I=1):
    # 
    yield (n,)
    if(l > 1):
        for i in range(I, n//2 + 1):
            for p in partitions(n-i, l-1, i):
                yield p + (i,)

def optimize(l, tol):
    # try all s_l(x) for x which sums to 1,
    # up to a certain tolerance tol and then output the largest one.
    n = len(l)
    val = 0
    for y in partitions(tol, n):
        x = np.zeros(n)
        for i in range(len(y)):
            x[i] = y[i]/tol
        newval = schur(l, x)
        if(newval > val):
            val = newval
            best = x.copy()
    return val, best

# Schur polynomial (2, 1)
# 1 1   1 2
# 2     2
#print(schur((2, 1), [2/3, 1/3]))
#print(schur((5, 2, 1), [1, 1, 1]))
#for x in accel_asc(10):
#    print(x)
print(optimize((5,2,1,1), 100))
