import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Global options.
plt.rcParams['font.family'] = 'fira code'

x = np.linspace(3, 42, 14)
y = [9, 21, 37, 56, 76, 97, 120, 144, 170, 196, 223, 253, 281, 310]

datasets = {
    'Linear fit\n y = 7.86x - 34.49': (x, y),
    '4/3 fit\n y = 2.14x^(4/3) - 2.77': (x, y),
    'Power law fit\n y = 1.85x^1.37 - 0.08': (x, y),
}

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4),
                        gridspec_kw={'wspace': 0.08, 'hspace': 0.08})
fig.suptitle('MLE performance on distributions with matching second moments', x=0.1, y=0.95, ha='left', weight='semibold')
axs[0].set(xlim=(0, 45), ylim=(0, 325))
axs[0].set(xticks=(0, 10, 20, 30, 40), yticks=(0, 100, 200, 300))
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
    ax.spines['bottom'].set_bounds(0, 45)
    ax.spines['left'].set_bounds(0, 320)

# linear regression
p1, p0 = np.polyfit(x, y, deg=1)  # slope, intercept
print(p1, p0)
axs[0].axline(xy1=(0, p0), slope=p1, lw=1)

# 4/3
xfit = np.linspace(0, 50, 1000)
yfit1 = 2.14 * (xfit ** (4/3)) - 2.77
axs[1].plot(xfit, yfit1, color='tab:orange', lw=1)

# polynomial
xfit = np.linspace(0, 50, 1000)
yfit2 = 1.85 * (xfit ** 1.37) - 0.08
axs[2].plot(xfit, yfit2, color='tab:green', lw=1)

plt.savefig("fig2.svg", bbox_inches='tight')

# add text box for the statistics
# stats = (f'$\\mu$ = {np.mean(y):.2f}\n'
#          f'$\\sigma$ = {np.std(y):.2f}\n'
#          f'$r$ = {np.corrcoef(x, y)[0][1]:.2f}')
# bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
# ax.text(0.95, 0.07, stats, fontsize=9, bbox=bbox,
#        transform=ax.transAxes, horizontalalignment='right')


