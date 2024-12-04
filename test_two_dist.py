"""
This script generates two distributions that match on the first two moments and computes the error rate for a given number of boxes in the tableaux generated from these distributions. 
It then searches for the optimal number of boxes such that the error rate is below a specified threshold.

Functions:
    generate_distributions(d):
        Generates two distributions based on the input parameter d.

    one_test_error(n, x1, x2):

Main Execution:
    - Sets a threshold and number of iterations.
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


def generate_distributions(d):    
    """
    Generate two distributions which match on the first two moments based on the input parameter d.

    Parameters:
    d (int): The size of the distributions to generate. Must be a multiple of 3.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - x1: A numpy array where the first 2/3 of the elements are set to 3/2/d and the rest are zeros.
        - x2: A numpy array where the first 1/3 of the elements are set to 2/d and the rest are set to 1/2/d.
    """
    x1 = np.zeros(d)
    x1[:2*int(d/3)] = 3/2/d

    x2 = np.zeros(d)
    x2[:int(d/3)] = 3/2/d
    x2 += 1/2/d

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
    print("CPU counts: ", os.cpu_count())
    print("Threshold = {} and iterations = {}".format(threshold, iterations))
    ds = [3 * i for i in range(1, 21)]
    best_ns = []

    predicted_scalar = 2.15

    for d in ds:
        print("Running d = ", d)
        predicted_n = int(predicted_scalar * d**(4/3))
        start = max(0, predicted_n - 6)
        end = predicted_n + 2

        x1, x2 = generate_distributions(d)

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


