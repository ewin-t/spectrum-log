import sys
import numpy as np
import rsk, mle
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os, time



def tvdist(alpha, beta):
    # computes the TV distance between alpha and beta
    # alpha and beta should be sorted when 
    return np.sum(np.abs(np.array(sorted(alpha)) - np.array(sorted(beta)))) / 2


def one_test(n, alpha, dist_try, tol=None):
    # dist_try denotes the ball in which optimize checks. i.e. MLE will only be correct if the arg max s_lambda(x) is within dist_try of alpha.

    if tol is None:
        tol = n
    l = rsk.generate_tableau(alpha, n)
    
    est_eyd_try = rsk.eyd(l)
    eyd_err_try = tvdist(alpha, est_eyd_try)

    brute_fun, est_mle_try = mle.optimize_brute(l, tol, smart=True, alpha=alpha, dist=dist_try)
    
    mle_err_try = tvdist(alpha, est_mle_try)

    print("(Running CPU {})\nEYD tableau: {} with error {}\nMLE tableau: {} with error {}".format(os.getpid(), 
                                                                                                  l, np.round(eyd_err_try, decimals=4), 
                                                                                                  est_mle_try * n, np.round(mle_err_try, decimals=4)), flush=True)

    return eyd_err_try, mle_err_try


if __name__ == '__main__':
    # import cProfile
    # # if check avoids hackery when not profiling
    # # Optional; hackery *seems* to work fine even when not profiling, it's just wasteful
    # if sys.modules['__main__'].__file__ == cProfile.__file__:
    #     import main  # Imports you again (does *not* use cache or execute as __main__)
    #     globals().update(vars(main))  # Replaces current contents with newly imported stuff
    #     sys.modules['__main__'] = main  # Ensures pickle lookups on __main__ find matching version
    

    threading = True
    print("CPU counts: ", os.cpu_count())
    start_time = time.time()

    # We see a linear relationship between TV error and d, which is what we want
    # Since we expect err ~ d/sqrt(n).
    # n is the number of samples
    n = 80
    tol = 150
    # tries is the number of times it will average over
    tries = 24*8
    dist_try = 1

    # d is the support size of the distribution
    ds = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # this will store the error of each estimator as a function of d
    eyds = []
    mles = []
    for d in ds:
        # alpha = np.ones(d) / d
        alpha = np.array(sorted(range(1, d+1), reverse=True))
        alpha = alpha / np.sum(alpha)
        print("n=tol={}, tries={}, dist_try={}, d={}, alpha={}".format(n, tries, dist_try, d, alpha))
        eyd_err = 0
        mle_err = 0

        if threading:
            with Pool(os.cpu_count()) as pool:
                zipret = pool.starmap(one_test, zip([n for i in range(tries)], [alpha for i in range(tries)], [dist_try for i in range(tries)]))
                [eyd_err_all, mle_err_all] = list(zip(*zipret))
                eyd_err = sum(eyd_err_all) / tries
                mle_err = sum(mle_err_all) / tries
        else:
            for tmp in range(tries):
                eyd_err_try, mle_err_try = one_test(n, alpha, dist_try)
                eyd_err += eyd_err_try
                mle_err += mle_err_try
                eyd_err /= tries
                mle_err /= tries

        print("summary: d={} has EYD error {} and MLE error {} with runtime {}".format(d, eyd_err, mle_err, time.time() - start_time))
        start_time = time.time()

        eyds.append(eyd_err)
        mles.append(mle_err)

    print("EYDs", eyds) #, eyds[1]/eyds[0])
    print("MLEs", mles) #, mles[1]/mles[0])

    # print("Total runtime: ", time.time() - start_time)

    plt.plot(ds, eyds, label="EYD")
    plt.plot(ds, mles, label="MLE")
    plt.legend()
    plt.ylabel('TV error')
    plt.xlabel('d')
    #plt.axis((0, 6, 0, 20))
    plt.savefig("fig.png")
    plt.show()
