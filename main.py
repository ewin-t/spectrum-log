import numpy as np
import rsk, mle
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os, time

threading = True
print("CPU counts: ", os.cpu_count())
start_time = time.time()


def tvdist(alpha, beta):
    # computes the TV distance between alpha and beta
    # alpha and beta should be sorted when 
    return np.sum(np.abs(np.array(sorted(alpha)) - np.array(sorted(beta)))) / 2


def one_test(n, alpha, tol=None):
    if tol is None:
        tol = n
    l = rsk.generate_tableau(alpha, n)
    print("(Running CPU {}): The tableau is {}".format(os.getpid(), l))
    est_eyd_try = rsk.eyd(l)
    eyd_err_try = tvdist(alpha, est_eyd_try)

    brute_fun, est_mle_try = mle.optimize_brute(l, tol, smart=True, alpha=alpha, dist=dist_try)
    
    mle_err_try = tvdist(alpha, est_mle_try)

    print("EYD:", est_eyd_try, "with an error of", np.round(eyd_err_try, decimals=4))
    print("MLE:", est_mle_try, "with an error of", np.round(mle_err_try, decimals=4))
    print("MLE outputs:", est_mle_try*n)

    return eyd_err_try, mle_err_try


# Below is the check to make sure things are working
# We see a linear relationship between TV error and d, which is what we want
# Since we expect err ~ d/sqrt(n).


# n is the number of samples
n = 50
tol = 150
# tries is the number of times it will average over
tries = 30
# dist_try denotes the ball in which optimize checks. i.e. MLE will only be correct if the arg max s_lambda(x) is within dist_try of alpha.
dist_try = 0.5

# d is the support size of the distribution
ds = [5,10]
# this will store the error of each estimator as a function of d
eyds = []
mles = []
for d in ds:
    # alpha = np.ones(d) / d
    # alpha = np.array(sorted([np.exp(i/2) for i in range(1, d+1)], reverse=True))
    alpha = np.array(sorted(range(1, d+1), reverse=True))
    alpha = alpha / np.sum(alpha)
    print("alpha: ", alpha)
    eyd_err = 0
    mle_err = 0

    if threading:
        with Pool(os.cpu_count()) as pool:
            zipret = pool.starmap(one_test, zip([n for i in range(tries)], [alpha for i in range(tries)]))
            [eyd_err_all, mle_err_all] = list(zip(*zipret))
            eyd_err = sum(eyd_err_all) / tries
            mle_err = sum(mle_err_all) / tries
    else:
        for tmp in range(tries):
            eyd_err_try, mle_err_try = one_test(n, alpha)
            eyd_err += eyd_err_try
            mle_err += mle_err_try
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

