import matplotlib.pyplot as plt
import numpy as np
import mle

l = (21, 11, 3) #tableau to visualize
n = sum(l)
d = 3
tol = 100

print(l)
print(mle.optimize_brute(l, tol, smart=False))

image = np.zeros((tol+1, tol+1))
for y in mle.partitions(tol, d):
    x = np.zeros(d)
    for i in range(len(y)):
        x[i] = y[i]
    image[int(x[0]),int(x[1])] = mle.schur(l, x/tol)

fig, ax = plt.subplots()
im = ax.imshow(image, origin='lower')
#im = ax.imshow(image)
fig.colorbar(im, ax=ax, label='Interactive colorbar')
plt.show()

