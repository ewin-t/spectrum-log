from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

def power_fit(x, a, b, p):
    return a * x ** p + b

def linear_fit(x, a, b):
    return a * x + b

def fourthirds_fit(x, a, b):
    return a * x ** (4/3) + b

def threeseconds_fit(x, a, b):
    return a * x ** (3/2) + b

def fit_plot(x, y, fit_func, do_plot = False):
    param, param_cov = curve_fit(fit_func, x, y)
    ans = fit_func(x, *param)
    print(*param)
    print(ans)
    if(do_plot):
        plt.plot(x, y, 'o', color ='red', label ="data")
        plt.plot(x, ans, '--', color ='blue', label ="fit")
        plt.legend()
        plt.grid()
        plt.show()
    return param

def subplot_data(x, y, labels):
    # Make three subplots and then graph them
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4),
                            gridspec_kw={'wspace': 0.08, 'hspace': 0.08})
    axs[0].set_ylabel('Number of samples $n$')
    axs[0].set_xlabel('Dimension $d$')
    for ax, label in zip(axs.flat, labels):
        ax.text(0.05, 0.95, label, fontsize=14, transform=ax.transAxes, va='top')
        ax.scatter(x, y, color='black', s=20)
    
        # Remove axis lines.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return fig, axs

if __name__=="__main__":
    plt.rcParams['font.family'] = 'inconsolata'
    plt.rcParams['font.size'] = '14'
    plt.rcParams["mathtext.default"] = 'regular'
    xfit = np.linspace(0, 750, 1501)

    # testing task between uniform vs half uniform (they match on the first moment)
    # the scaling is proved to be linear
    d = np.array([2*i for i in range(3, 25)])
    n = np.array([9, 12, 15, 18, 20, 24, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 64, 68, 71])

    fig, axs = subplot_data(d, n, ['', 'Linear fit\n  $\mathregular{n = 1.47d - 0.06}$', 'Power law fit\n  $n = 1.26d^{1.04} + 1.08$'])
    fig.suptitle('Sample complexity of distinguishing uniform vs half-uniform', x=0.1, y=0.95, ha='left', fontsize='16')
    axs[0].set(xlim=(0, 50), ylim=(0, 72))
    axs[0].set(xticks=(0, 10, 20, 30, 40, 50), yticks=(0, 10, 20, 30, 40, 50, 60, 70))

    param = fit_plot(d, n, linear_fit)
    axs[1].plot(xfit, linear_fit(xfit, *param), color='tab:blue', lw=1)
    param = fit_plot(d, n, power_fit)
    axs[2].plot(xfit, power_fit(xfit, *param), color='tab:green', lw=1)
    plt.savefig("fig1.svg", bbox_inches='tight')

    # testing task between two distributions which match on the first two moments
    # we conjecture the scaling is n=d**(4/3)
    d = np.array([3*i for i in range(2, 17)])
    n = np.array([21, 37, 56, 76, 97, 120, 145, 171, 196, 223, 253, 281, 312, 342, 376])

    fig, axs = subplot_data(d, n, ['Linear fit\n  $n = 8.50d - 49.10$', '4/3 fit\n  $n = 2.16d^{4/3} - 4.29$', 'Power law fit\n  $n = 1.85d^{1.37} - 0.34$'])
    fig.suptitle('Sample complexity of distinguishing distributions with matching second moments', x=0.1, y=0.95, ha='left', fontsize='16')
    axs[0].set(xlim=(0, 50), ylim=(0, 400))
    axs[0].set(xticks=(0, 10, 20, 30, 40, 50), yticks=(0, 100, 200, 300, 400))

    param = fit_plot(d, n, linear_fit)
    axs[0].plot(xfit, linear_fit(xfit, *param), color='tab:blue', lw=1)
    param = fit_plot(d, n, fourthirds_fit)
    axs[1].plot(xfit, fourthirds_fit(xfit, *param), color='tab:orange', lw=1)
    param = fit_plot(d, n, power_fit)
    axs[2].plot(xfit, power_fit(xfit, *param), color='tab:green', lw=1)
    plt.savefig("fig2.svg", bbox_inches='tight')

    # testing task between two distributions which match on the first three moments
    # we conjecture the scaling is n=d**(3/2)
    d = np.array([4*i for i in range(1, 11)])
    n = np.array([19, 51, 95, 147, 209, 274, 347, 426, 511, 598])

    fig, axs = subplot_data(d, n, ['4/3 fit\n  $n = 4.47d^{4/3} - 25.60$', '3/2 fit\n  $n = 2.37d^{3/2} - 3.04$', 'Power law fit\n  $n = 2.12d^{1.53} - 0.58$'])
    fig.suptitle('Sample complexity of distinguishing distributions with matching third moments', x=0.1, y=0.95, ha='left', fontsize='16')
    axs[0].set(xlim=(0, 45), ylim=(0, 650))
    axs[0].set(xticks=(0, 10, 20, 30, 40), yticks=(0, 100, 200, 300, 400, 500, 600))

    param = fit_plot(d, n, fourthirds_fit)
    axs[0].plot(xfit, fourthirds_fit(xfit, *param), color='tab:orange', lw=1)
    param = fit_plot(d, n, threeseconds_fit)
    axs[1].plot(xfit, threeseconds_fit(xfit, *param), color='tab:red', lw=1)
    param = fit_plot(d, n, power_fit)
    axs[2].plot(xfit, power_fit(xfit, *param), color='tab:green', lw=1)
    plt.savefig("fig3.svg", bbox_inches='tight')
