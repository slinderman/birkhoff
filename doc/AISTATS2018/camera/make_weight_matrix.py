import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
sns.set_context("paper")
import numpy as np


N = 10
A = np.random.rand(N, N) < 0.5
W = np.random.randn(N, N)
W[~A] = np.nan

cmap = get_cmap("RdBu")
cmap.set_bad(0.5 * np.ones(3))

fig = plt.figure(figsize=(1.75, 1.25))
ax = fig.add_subplot(111)
divider = make_axes_locatable(ax)

im = plt.imshow(np.kron(W, np.ones((100, 100))), interpolation="none", cmap=cmap, vmin=-2.2, vmax=2.2)
plt.xticks([])
plt.yticks([])

ax_cb = divider.new_horizontal(size="7%", pad=0.05)
fig.add_axes(ax_cb)
cb = plt.colorbar(im, cax=ax_cb)
cb.set_ticks([-2, 0, 2])
cb.set_ticklabels([])

plt.tight_layout(pad=0.1)
plt.savefig("weights.pdf")

plt.show()
