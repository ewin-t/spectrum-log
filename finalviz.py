import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Global options.
plt.rcParams['font.family'] = 'fira code'

x = np.linspace(6, 48, 22)
y = [9, 12, 15, 18, 20, 24, 27, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 64, 68, 70]

datasets = {
    '': (x, y),
    'Linear fit\n y = 1.46x + 0.17': (x, y),
    'Power law fit\n y = 1.39x^1.01 + 0.56': (x, y),
}

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4),
                        gridspec_kw={'wspace': 0.08, 'hspace': 0.08})
fig.suptitle('MLE performance on uniform vs half-uniform', x=0.1, y=0.95, ha='left', weight='semibold')
axs[0].set(xlim=(0, 50), ylim=(0, 72))
axs[0].set(xticks=(0, 10, 20, 30, 40, 50), yticks=(0, 10, 20, 30, 40, 50, 60, 70))
axs[0].set_ylabel(r'Number of samples $n$')
axs[0].set_xlabel(r'Dimension $d$')

for ax, (label, (x, y)) in zip(axs.flat, datasets.items()):
    print(ax)
    ax.text(0.05, 0.95, label, fontsize=10, transform=ax.transAxes, va='top')
    # ax.tick_params(direction='in', top=True, right=True)
    ax.scatter(x, y, color='black', s=20)
    # ax.plot(x, y, 'o')

    # Remove axis lines.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set spine extent.
    ax.spines['bottom'].set_bounds(0, 50)
    ax.spines['left'].set_bounds(0, 70)

# linear regression
p1, p0 = np.polyfit(x, y, deg=1)  # slope, intercept
print(p1, p0)
axs[1].axline(xy1=(0, p0), slope=p1, color='tab:blue', lw=1)

# polynomial
xfit = np.linspace(0, 50, 1000)
yfit = 1.39 * (xfit ** 1.01) + 0.56
axs[2].plot(xfit, yfit, color='tab:green', lw=1)

plt.savefig("fig1.svg")

# add text box for the statistics
# stats = (f'$\\mu$ = {np.mean(y):.2f}\n'
#          f'$\\sigma$ = {np.std(y):.2f}\n'
#          f'$r$ = {np.corrcoef(x, y)[0][1]:.2f}')
# bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
# ax.text(0.95, 0.07, stats, fontsize=9, bbox=bbox,
#        transform=ax.transAxes, horizontalalignment='right')


