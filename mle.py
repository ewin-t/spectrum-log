import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt

def vander(x, shift, same):
    # Output the Vandermonde matrix
    # M[i,j] = same[j]th derivative of (x[j] ** (shift[i] + n - 1 - i))
    # except when same[i] != 0
    d = len(x)
    M = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            power = shift[i] + d - 1 - i
            M[i,j] = np.math.factorial(power)/np.math.factorial(power - same[j]) * (x[j] ** (power - same[j])) if power >= same[j] else 0
    return M.T

def schur(l, x):
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
        

# Some tests
# Schur polynomial (2, 1)
# 1 1   1 2
# 2     2
#print(schur((2, 1), [2/3, 1/3]))
#print(schur_new((2, 1), [2/3, 1/3]))
#print(schur((5, 2, 1), [1, 1, 1]))
#print(schur_new((5, 2, 1), [1, 1, 1]))
#print(schur((5, 2, 1, 1), [1, 1, 0.5, 0.5]))
#print(schur_new((5, 2, 1, 1), [1, 1, 0.5, 0.5]))

# Copied from https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
def partitions(n, l, I=1):
    yield (n,)
    if(l > 1):
        for i in range(I, n//2 + 1):
            for p in partitions(n-i, l-1, i):
                yield p + (i,)

def optimize_brute(l, tol):
    # try all s_l(x) for x which sums to 1,
    # up to a certain tolerance tol and then output the largest one.
    d = len(l)
    val = -np.inf
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
    # alpha and beta should be sorted when 
    return np.sum(np.abs(np.array(sorted(alpha)) - np.array(sorted(beta)))) / 2

print(optimize_brute((5, 5, 1), 100))
print(optimize((5, 5, 1), (0.5, 0.5, 0)))

r'''
# Below is the check to make sure things are working
# We see a linear relationship between TV error and d, which is what we want
# Since we expect err ~ d/sqrt(n).

# n is the number of samples
n = 1000
# tries is the number of times it will average over
tries = 5

# d is the support size of the distribution
ds = range(2, 10)
# this will store the error of each estimator as a function of d
eyds = []
for d in ds:
    alpha = np.ones(d) / d
    eyd_err = 0
    for tmp in range(tries):
        l = generate_tableau(alpha, n)
        est_eyd_try = eyd(l)
        eyd_err_try = tvdist(alpha, est_eyd_try)
        eyd_err += eyd_err_try
        print("EYD:", est_eyd_try, "with an error of", np.round(eyd_err_try, decimals=4))
    eyd_err /= tries
    eyds.append(eyd_err ** 2)

plt.plot(ds, eyds, label="EYD")
plt.legend()
plt.ylabel('TV error')
plt.xlabel('d')
plt.title('n = 500, avgd over 5 tries')
#plt.axis((0, 6, 0, 20))
plt.show()
'''

# n is the number of samples
n = 50
#tol = 40
# tries is the number of times it will average over
tries = 100

# d is the support size of the distribution
ds = [3, 6]
# this will store the error of each estimator as a function of d
eyds = []
mles = []
for d in ds:
    #alpha = np.ones(d) / d
    alpha = np.array(sorted(range(1, d+1), reverse=True))
    alpha = alpha / np.sum(alpha)
    eyd_err = 0
    mle_err = 0
    for tmp in range(tries):
        #alpha = np.random.random(d)
        #alpha /= np.sum(alpha)
        l = generate_tableau(alpha, n)
        est_eyd_try = eyd(l)
        eyd_err_try = tvdist(alpha, est_eyd_try)
        eyd_err += eyd_err_try
        #brute_fun, est_mle_brute_try = optimize_brute(l, tol)
        smart_fun, est_mle_try = optimize(l, alpha)
        print(smart_fun)
        #print(brute_fun, "vs", -smart_fun)
        mle_err_try = tvdist(alpha, est_mle_try)
        mle_err += mle_err_try
        print("EYD:", est_eyd_try, "with an error of", np.round(eyd_err_try, decimals=4))
        print("MLE:", est_mle_try, "with an error of", np.round(mle_err_try, decimals=4))
    eyd_err /= tries
    mle_err /= tries
    eyds.append(eyd_err)
    mles.append(mle_err)

print("EYDs", eyds)#, eyds[1]/eyds[0])
print("MLEs", mles)#, mles[1]/mles[0])

plt.plot(ds, eyds, label="EYD")
plt.plot(ds, mles, label="MLE")
plt.legend()
plt.ylabel('TV error')
plt.xlabel('d')
#plt.axis((0, 6, 0, 20))
plt.show()
