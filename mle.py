import numpy as np
import tnn


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


def optimize_brute(l, tol, alpha, smart=True, dist=0.5):
    # try all s_l(x) for x which sums to 1,
    # up to a certain tolerance tol and then output the largest one.
    d = len(l)
    val = -np.inf
    bounds = partitions_ball(alpha * tol, dist * tol, tol) if smart else partitions(tol, d)
    # bounds = partitions_ball(l, dist * tol, tol) if smart else partitions(tol, d)
    for y in bounds:
        x = np.array([y[i] / tol for i in range(len(y))])
        newval = schur_tnn(l, x)
        if(newval > val):
            val = newval
            best = x.copy()
    return val, best


def optimize_gradient(l, learning_rate=0.2, iterations=10000):
    # implement gradient/greedy acsent and hope to find the global maximum
    d = len(l)
    n = sum(l)
    x = np.array(l)/n
    y = schur_tnn(l, x)

    for i in range(iterations):
        y_new = -np.inf
        for idx1 in range(d - 1):
            for idx2 in range(idx1 + 1, d):
                for sgn in [1,-1]:
                    dir = np.zeros(d)
                    dir[idx1] = -1
                    dir[idx2] = 1
                    x_update = x + sgn * learning_rate / n * dir
                    y_update = schur_tnn(l, x_update)
                    if y_update > y_new:
                        y_new = y_update
                        x_new = x_update.copy()
        if y_new > y:
            y = y_new
            x = x_new.copy()
        else:
            return y, np.sort(x)[::-1]

    return y, np.sort(x)[::-1]


if __name__=="__main__":
    l = [27, 24, 14, 9, 4, 2]
    y, x = optimize_gradient(l)
    print(x)
    print(schur_tnn(l, x))
