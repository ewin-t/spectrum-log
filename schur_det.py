import numpy as np
import scipy.optimize as sopt
from mle import schur_tnn


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


def schur(l, x, method='tnn'):
    if method == 'det':
        return schur_det(l, x)
    elif method == 'tnn':
        return schur_tnn(l, x)


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