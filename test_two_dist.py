"""
This script generates two distributions that match on the first moments (up to 3) and computes the error rate for a given number of boxes in the tableaux generated from these distributions. 
It then searches for the optimal number of boxes such that the error rate is below a specified threshold.

Functions:
    generate_distributions(d, k):
        Generates two distributions of length d which match on the first k moments. 

    one_test_error(n, x1, x2):

Main Execution:
    - Sets a threshold and number of iterations.
    - Specify the number of matching moments. 
    - Iterates over a range of distribution sizes.
    - For each size, predicts an initial number of boxes and adjusts the search window to find the optimal number of boxes.
    - Uses multiprocessing to parallelize the error rate computation.
    - Implements a binary search to refine the search for the optimal number of boxes.
    - Prints and stores the optimal number of boxes for each distribution size.
"""

import numpy as np
import rsk, schur
from multiprocessing import Pool
import os
from itertools import repeat


def generate_distributions(d, k):    
    """
    Generate two distributions which match on the first several moments.

    Parameters:
    d (int): The size of the distributions to generate. Must be a multiple of (k + 1).
    k (int): The number of moments that the returned two distributions should match on. 

    Returns:
    tuple: A tuple containing two numpy arrays x1 and x2, each of length d, and they match on the first k moments. 
    """
    assert(d % (k+1) == 0)

    x1, x2 = np.zeros(d), np.zeros(d)
    if k == 1:
        x1 = np.ones(d)/d
        x2[:int(d/2)] = 2/d
    elif k == 2:
        x1[:2*int(d/3)] = 3/2/d
        x2[:int(d/3)] = 3/2/d
        x2 += 1/2/d
    elif k == 3:
        x1[:int(d/2)] = (1+1/np.sqrt(2))/d
        x1[int(d/2):] = (1-1/np.sqrt(2))/d
        x1 = x1 / np.sum(x1)
        x2[:int(d/4)] = 2/d
        x2[int(d/4):3*int(d/4)] = 1/d

    return x1, x2


def one_test_error(n, x1, x2):
    """
    Computes the error rate for a given n and two distributions x1 and x2.

    Parameters:
    n (int): The number of boxes in the tableaux to be generated.
    x1 (array-like): The first distribution.
    x2 (array-like): The second distribution.

    Returns:
    float: The error rate for the given n and distributions.
    """
    l1 = rsk.generate_tableau(x1, n)
    l2 = rsk.generate_tableau(x2, n)
    error_1 = int(schur.schur_tnn(l1, x1) < schur.schur_tnn(l1, x2))
    error_2 = int(schur.schur_tnn(l2, x1) >= schur.schur_tnn(l2, x2))
    return (error_1 + error_2) / 2


if __name__=="__main__":
    threshold = 0.3
    iterations = 100000
    matching_moment = 3
    assert(matching_moment <= 3)

    print("CPU counts: ", os.cpu_count())
    print("Threshold = {} and iterations = {}".format(threshold, iterations))
    print("Two distributions match on their first {} moments".format(matching_moment))

    if matching_moment == 1:
        predicted_scalar = 1.46
        predicted_power = 1
        predicted_offset = 0.17
        ds = [2 * i for i in range(3, 25)]
    elif matching_moment == 2:
        predicted_scalar = 2.16
        predicted_power = 4/3
        predicted_offset = -4.29
        ds = [3 * i for i in range(1, 17)]
    elif matching_moment == 3:
        predicted_scalar = 2.37
        predicted_power = 3/2
        predicted_offset = -3
        ds = [4 * i for i in range(9, 16)]
    
    best_ns = []

    for d in ds:
        print("Running d = ", d)
        predicted_n = int(predicted_scalar * d ** predicted_power - predicted_offset)
        start = max(0, predicted_n - 4)
        end = predicted_n + 4

        x1, x2 = generate_distributions(d, matching_moment)
        print(x1, x2)

        while True: # find a search window for n such that when n=start, error_rate > threshold
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

        while True: # find a search window for n such that when n=end, error_rate < threshold
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
        # implement binary search
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


