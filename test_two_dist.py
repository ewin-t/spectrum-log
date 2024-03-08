import numpy as np
import rsk, mle
from multiprocessing import Pool
import os, time
from itertools import repeat


def generate_distributions(d):
    # d is required to be a multiple of 6
    x1 = np.zeros(d)
    for i in range(int(d/2)):
        x1[i] = 2/d
    
    x2 = np.zeros(d)
    for i in range(int(d/3)):
        x2[i] = 3/d/np.sqrt(2)
    for i in range(d):
        x2[i] += 1/d * (1-1/np.sqrt(2))

    return x1, x2


# def type_one_error(x1, x2, n, iterations=10000):
#     counter = 0
#     for i in range(iterations):
#         l1 = rsk.generate_tableau(x1, n)
#         if mle.schur_tnn(l1, x1) < mle.schur_tnn(l1, x2):
#             counter += 1

#     return counter / iterations


def one_test_error(n, x1, x2):
    l1 = rsk.generate_tableau(x1, n)
    l2 = rsk.generate_tableau(x2, n)
    return (int( mle.schur_tnn(l1, x1) < mle.schur_tnn(l1, x2) ) + int( mle.schur_tnn(l2, x1) > mle.schur_tnn(l2, x2) )) / 2


if __name__=="__main__":
    threshold = 0.1
    iterations = 1000
    threading = True
    print("CPU counts: ", os.cpu_count())
    start_time = time.time()

    default_start, default_end = 4, 7

    ds = [6 * i for i in range(1, 13)]
    # (d,n) = (6,27); (12,59); (18,101); (24,)

    for d in ds:
        print("Running d = ", d)
        start = default_start * d
        end = default_end * d
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



