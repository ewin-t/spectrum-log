from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt


def power_fit(x, a, b, p):
    return a * x ** p + b

def linear_fit(x, a, b):
    return a * x + b

def fourthirds_fit(x, a, b):
    return a * x ** (4/3) + b

def threeseconds_fit(x, a, b):
    return a * x ** (3/2) + b

def fit_plot(x, y, fit_func):
    param, param_cov = curve_fit(fit_func, x, y)
    ans = fit_func(x, *param)
    print(*param)
    plt.plot(x, y, 'o', color ='red', label ="data")
    plt.plot(x, ans, '--', color ='blue', label ="fit")
    plt.legend()
    plt.grid()
    plt.show()


if __name__=="__main__":
    # testing task between uniform vs half uniform (they match on the first moment)
    # the scaling is proved to be linear
    d = np.array([2*i for i in range(3, 25)])
    n = np.array([9, 12, 15, 18, 20, 24, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 64, 68, 71])
    fit_plot(d, n, linear_fit)
    fit_plot(d, n, power_fit)

    # testing task between two distributions which match on the first two moments
    # we conjecture the scaling is n=d**(4/3)
    d = np.array([3*i for i in range(2, 17)])
    n = np.array([21, 37, 56, 76, 97, 120, 145, 171, 196, 223, 253, 281, 312, 342, 376])
    fit_plot(d, n, linear_fit)
    fit_plot(d, n, fourthirds_fit)
    fit_plot(d, n, power_fit)

    # testing task between two distributions which match on the first three moments
    # we conjecture the scaling is n=d**(3/2)
    d = np.array([4*i for i in range(1, 11)])
    n = np.array([19, 51, 95, 147, 209, 274, 347, 426, 511, 598])
    fit_plot(d, n, fourthirds_fit)
    fit_plot(d, n, threeseconds_fit)
    fit_plot(d, n, power_fit)
