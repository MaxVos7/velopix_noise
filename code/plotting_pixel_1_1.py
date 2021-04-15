import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


# Plot a specific pixel.
def plot_pixel(filename):
    data = np.array(np.loadtxt('../data/' + filename, delimiter=','))
    plt.plot(data[:, 0], data[:, 1], marker='.')
    plt.xlabel('DAC treshold')
    plt.ylabel('Number of hits')
    plt.show()


# Enter only filename (not entire path) in command line parameter
filename = sys.argv[1]

plot_pixel(filename)
