import numpy as np
import rsk, mle
from multiprocessing import Pool
import os, time
from itertools import repeat


def generate_distributions(d):
    # # d is required to be a multiple of 6
    # x1 = np.zeros(d) # (1/2 uniform)
    # for i in range(int(d/2)):
    #     x1[i] = 2/d
    
    # x2 = np.zeros(d) # mu * (1/3 uniform) + (1 - mu) * (uniform) where mu = 1/sqrt(2)
    # for i in range(int(d/3)):
    #     x2[i] = 3/d/np.sqrt(2)
    # for i in range(d):
    #     x2[i] += 1/d * (1-1/np.sqrt(2))
    
    # d is required to be a multiple of 3
    x1 = np.zeros(d) # (2/3 uniform)
    for i in range(2*int(d/3)):
        x1[i] = 3/2/d

    x2 = np.zeros(d) # mu * (1/3 uniform) + (1 - mu) * (uniform) where mu = 1/2
    for i in range(int(d/3)):
        x2[i]=3/2/d
    for i in range(d):
        x2[i] += 1/2/d

    return x1, x2


def one_test_error(n, x1, x2):
    l1 = rsk.generate_tableau(x1, n)
    l2 = rsk.generate_tableau(x2, n)
    return (int( mle.schur_tnn(l1, x1) < mle.schur_tnn(l1, x2) ) + int( mle.schur_tnn(l2, x1) > mle.schur_tnn(l2, x2) )) / 2
    # return int( mle.schur_tnn(l1, x1) < mle.schur_tnn(l1, x2) )


if __name__=="__main__":
    threshold = 0.3
    iterations = 10000
    threading = True
    print("CPU counts: ", os.cpu_count())
    print("Threshold = {} and iterations = {}".format(threshold, iterations))
    start_time = time.time()

    default_start, default_end = 4, 6

    ds = [3 * i for i in range(1, 20)]
    best_ns = []


    for d in ds:
        print("Running d = ", d)
        start = default_start * d
        end = default_end * d
        # end = ns[int(d/6)-1]+4
        # start = end-8
        x1, x2 = generate_distributions(d)

        while True: # check that the start has error_rate > threshold
            with Pool(os.cpu_count()) as pool:
                counter = pool.starmap(one_test_error, zip([start for _ in range(iterations)], repeat(x1), repeat(x2)))
                error_rate = sum(counter) / iterations
                print("Error rate for n = {} is {}".format(start, error_rate))
                if error_rate > threshold:
                    break
                elif error_rate < threshold:
                    end = start
                    start -= d
                else:
                    end = start
                    start = end - 1
                    break

        while True:
            with Pool(os.cpu_count()) as pool:
                counter = pool.starmap(one_test_error, zip([end for _ in range(iterations)], repeat(x1), repeat(x2)))
                error_rate = sum(counter) / iterations
                print("Error rate for n = {} is {}".format(end, error_rate))
                if error_rate < threshold:
                    break
                elif error_rate > threshold:
                    start = end
                    end += d
                else:
                    start = end - 1
                    break

        print("Search window for n is [{}, {}]".format(start, end))
        while end - start > 1:
            mid = int( (end + start) / 2 )
            with Pool(os.cpu_count()) as pool:
                counter = pool.starmap(one_test_error, zip([mid for _ in range(iterations)], repeat(x1), repeat(x2)))
                error_rate = sum(counter) / iterations
                print("Error rate for n = {} is {}".format(mid, error_rate))
                if error_rate > threshold:
                    start = mid
                else:
                    end = mid

        print("Found n = {}".format(end))
        best_ns.append(end)

    print(best_ns)


