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
    tol = 0.000001
    val = x[0] + 1
    for i in range(len(x)):
        if(x[i] == val):
            x[i] = x[i-1] + tol
        else:
            val = x[i]
    # print(x)
    return schur_unequal(l, x)

# Some tests
# Schur polynomial (2, 1)
# 1 1   1 2
# 2     2
#print(schur((2, 1), [2/3, 1/3]))
#print(schur((5, 2, 1), [1, 1, 1]))

# Copied from https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
def partitions(n, l, I=1):
    yield (n,)
    if(l > 1):
        for i in range(I, n//2 + 1):
            for p in partitions(n-i, l-1, i):
                yield p + (i,)

def optimize(l, tol):
    # try all s_l(x) for x which sums to 1,
    # up to a certain tolerance tol and then output the largest one.
    d = len(l)
    val = 0
    for y in partitions(tol, d):
        x = np.zeros(d)
        for i in range(len(y)):
            x[i] = y[i] / tol
        # print(x)
        newval = schur(l, x)
        if(newval > val):
            val = newval
            best = x.copy()
    return val, best

r'''
The stuff below is helper stuff for computing the EYD estimator
'''

def rsk_insert(val, ssyt, row=0):
    # row is the current row being inserted
    isSmaller = list(map(lambda x: x < val, ssyt[row]))
    if(True in isSmaller):
        insertIndex = isSmaller.index(True)
        newval = ssyt[row][insertIndex]
        ssyt[row][insertIndex] = val
        if(len(ssyt) == row+1):
            ssyt.append([newval])
        else:
            rsk_insert(newval, ssyt, row+1)
    else:
        ssyt[row].append(val)

def rsk(x, d):
    # RSK correspondence
    # x is a list of elements in [1, d]; outputs l padded with zeroes
    ssyt = [[]]
    for val in x:
        rsk_insert(val, ssyt)
        # print(ssyt)
    lamb = []
    for row in ssyt:
        lamb.append(len(row))
    lamb += (d - len(lamb)) * [0]
    print("The output of the RSK algorithm is", ssyt)
    print("This corresponds to a tableau of", lamb)
    return lamb

# Test from OW survey: should output [4,4,3,3,2],[3,3,1],[2],[1]
# a = [1,3,3,2,1,3,4,4,3,2]
# rsk(a)

def eyd(l):
    # The "empirical young diagram" estimator
    # Given a tableau l with |l| = n, output l/n.
    d = len(l)
    estimate = np.zeros(d)
    for i in range(len(l)):
        estimate[i] = l[i]
    return estimate / np.sum(estimate)

r'''
Now the helper functions for comparing the two.
'''

def generate_tableau(alpha, n):
    d = len(alpha)
    sample = np.random.choice(range(1,d+1), n, p=alpha)
    print("The random sample according to", alpha, "is", sample)
    return rsk(sample, d)

def tvdist(alpha, beta):
    # computes the TV distance between alpha and beta
    return np.sum(np.abs(np.array(alpha) - np.array(beta))) / 2

#print(optimize((10,1,1,1,1), 50))

# d is the support size of the distribution
d = 6
# n is the number of samples
n = 50
# tol is the tolerance
tol = n
alpha = np.ones(d) / d

l = generate_tableau(alpha, n)
est_eyd = eyd(l)
est_mle = optimize(l, tol)[1]
print("EYD:", est_eyd, "with an error of", np.round(tvdist(alpha, est_eyd), decimals=4))
print("MLE:", est_mle, "with an error of", np.round(tvdist(alpha, est_mle), decimals=4))
