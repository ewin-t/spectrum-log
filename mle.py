import numpy as np
import tnn
import scipy.optimize as sopt


def vander(x, shift, same):
    '''
    Output the Vandermonde matrix
    M[i,j] = same[j]th derivative of (x[j] ** (shift[i] + n - 1 - i))
    except when same[i] != 0
    '''
    d = len(x)
    M = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            power = shift[i] + d - 1 - i
            M[i,j] = np.math.factorial(power)/np.math.factorial(power - same[j]) * (x[j] ** (power - same[j])) if power >= same[j] else 0
    return M.T


def schur_det(l, x):
    d = len(x)
    same = []
    for i in range(d):
        if(i > 0 and np.abs(x[i] - x[i-1]) < 1e-5):
            same.append(same[-1] + 1)
        else:
            same.append(0)
    Mvander = vander(x, np.zeros(d), same)
    Mnum = vander(x, l, same)
    return np.linalg.det(Mnum) / np.linalg.det(Mvander)
        
def schur_tnn(l, x):
    '''
    We rely on the observation of [CDEKK18], which states that
        s_lambda(x) = det(V),
    where V denotes the submatrix of the (upper triangular) matrix
        U_{i,j} = h_{i-j}(x),
    where h_i(x) is the complete symmetrix polynomial, indexed by columns
        n+lambda_1, n-1+lambda_2, ...

    Since U is totally non-negative, V is as well. Since U is TNN, it can be decomposed in a really nice way that allows for efficient computation. In particular, we maintain the bi-diagonal decomposition of U, which is essentially compactifying a product of matrices
        D * upper(x1, x2, ..., xn) * upper(x1, ..., x n-1, 0) * ...
    into the matrix form described in (Eq. (14) of [CDEKK18]).
    '''
    d = len(l)

    # handle the case when x contains 0
    x_nnz = np.count_nonzero(x)
    if np.count_nonzero(l) > x_nnz:
        return 0
    if x_nnz < d:
        return schur_tnn(l[:x_nnz], x[:x_nnz])
    
    # Construct the initial decomposition of U
    # bdecomp = np.zeros((d, d+l[0]))
    # for i in range(d):
    #     for j in range(d+l[0]):
    #         if(i == j):
    #             bdecomp[i,j] = 1
    #         if(i < j):
    #             bdecomp[i,j] = x[i]
    end_range = d + l[0]
    bdecomp = np.concatenate((np.identity(d), np.zeros((d, l[0]))), axis=1)
    for i in range(d):
        bdecomp[i, i + 1:] = x[i] * np.ones(end_range - i - 1) # rewrite codes commented out above)

    # Construct the list of columns that need to be removed from U
    to_remove = list(range(end_range))[::-1]
    for i in range(d):
        to_remove.remove(d - i - 1 + l[i]) # the indexing starts at zero
    #print(bdecomp)
    for i in to_remove:
        #print("removing row", i)
        bdecomp = tnn.remove_row(bdecomp.T, i).T
        #print(bdecomp)
    # The determinant is the product of the diagonal entries in the BD matrix
    output = 1
    for i in range(d):
        output *= bdecomp[i][i]
    return output

def schur(l, x, method='tnn'):
    if method == 'det':
        return schur_det(l, x)
    elif method == 'tnn':
        return schur_tnn(l, x)


# Copied from https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
def partitions(n, l, I=1):
    '''
    Partitions of at most l numbers which sum to n
    Builds from the bottom up; I is the smallest any number can be
    '''
    yield (n,)
    if(l > 1):
        for i in range(I, n//2 + 1):
            for p in partitions(n-i, l-1, i):
                yield p + (i,)


def partitions_ball(x0, dist, l1, I=0):
    '''
    Gives a list of integer x (sorted) such that ||x||_1 = l1
        and ||x - x0|| <= dist.
    '''
    n = sum(x0)
    l = len(x0)
    if(l == 1):
        yield (l1,)
    if(l > 1):
        # x[-1] needs to satisfy three criteria
        # I <= x[-1] <= l1/l
        # -dist <= x[-1] - x0[-1] <= dist
        # |(l1 - x[d]) - (||x0||1 - x0[d])| + |x[d] - x0[d]| <= dist
        bounds_itr = filter(lambda val: (np.abs(val - x0[-1]) <= dist)
                           and (np.abs((l1 - val) - (n - x0[-1])) + np.abs(val - x0[-1]) <= dist)
                           and (I <= val)
                           and (val <= l1/l),
                        range(0, l1+1))
        for i in bounds_itr:
            for p in partitions_ball(x0[:-1], dist - np.abs(i - x0[-1]), l1 - i, I=i):
                yield p + (i,)


def optimize_brute(l, tol, smart=False, alpha=False, dist=1):
    # try all s_l(x) for x which sums to 1,
    # up to a certain tolerance tol and then output the largest one.
    d = len(l)
    val = -np.inf
    # bounds = partitions_ball(alpha * tol, dist * tol, tol) if smart else partitions(tol, d)
    bounds = partitions_ball(l, dist * tol, tol) if smart else partitions(tol, d)
    for y in bounds:
        x = np.array([y[i] / tol for i in range(len(y))])
        newval = schur(l, x)
        if(newval > val):
            val = newval
            best = x.copy()
    return val, best

def schur_opt(l):
    def opt_function(x):
        return -schur(l, x)
    return opt_function

def optimize(l, alpha):
    # arg max_x s_l(x),
    # subject to x >= 0 and \sum x = 1.
    # alpha works as the start point
    d = len(l)
    bounds = d * [(0, 1.0),]
    cons = sopt.LinearConstraint(d * [1], [1], [1])
    soln = sopt.minimize(schur_opt(l),
                         alpha,
                         method='trust-constr',
                         bounds=bounds,
                         constraints=cons,
                         options={'disp': True})#, 'initial_tr_radius': 0.5})
    return soln.fun, sorted(soln.x, reverse=True)



r'''
Now the helper functions for comparing the two.
'''
def concavity_test(l):
    # test if s_l is concave in a silly way.
    # s_l is not concave (try (1,1,1))
    d = len(l)
    for t in range(1000):
        x = np.random.random(d)
        x.sort()
        x = x[::-1]
        x /= x.sum()
        y = np.random.random(d)
        y.sort()
        y /= y.sum()
        y = y[::-1]
        z = (x + y) / 2
        sx = schur(l, x)
        sy = schur(l, y)
        sz = schur(l, z)
        if(sx + sy > 2 * sz):
            print(t, x, y, sx, sy, sz)
            return False
    return True
    
#concavity_test((1,1, 1))

#print(optimize_brute((10, 10, 1), 50))
#print(optimize((5, 5, 1), (0.5, 0.5, 0)))
