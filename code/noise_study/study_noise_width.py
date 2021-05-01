import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches

# This file focuses on studying the width of the pixel distributions
# We have a file that holds the width of the pixel distribution.
# What is the distribution of the width.

# fetch the width for trim 0

trims = list(map(lambda value: format(value, 'x').upper(), np.arange(16)))
mycolors = cm.get_cmap('viridis', len(trims) + 1)
handles = []
for i in np.arange(len(trims)):
    trim = trims[i]
    files = glob.glob(f"../../data/Module3_VP0-0_Trim{trim}_Noise_Width.csv")
    if len(files) < 1:
        print(f"Skipping {trim}, no noise width file found.")
        continue

    width_matrix = np.loadtxt(files[0], delimiter=',')

    width_bin = np.arange(0, 100, 1)

    hist_data = np.histogram(width_matrix, bins=width_bin)

    color = mycolors(i)

    plt.semilogy(hist_data[1][:-1], hist_data[0], linestyle='-', linewidth=2, color=color, label=f"trim")

    handles.append(patches.Patch(color=color, label=f"Trim:{trim}, mean:{width_matrix.mean()}"))

plt.legend(bbox_to_anchor=(1.042, 0.8), loc=2, borderaxespad=0, handles=handles, frameon=False, handlelength=1.25)
plt.show()
