import glob
import sys
import numpy as np
import matplotlib.pyplot as plt


def make_plot_for_file(axis, filename, minDac=None, maxDac=None):
    data = np.array(np.loadtxt('../data/' + filename + '.csv', delimiter=',', dtype=int))

    dacList = data[:, 0]

    maxDacIndex = np.where(dacList <= maxDac)[0][0] if maxDac else 0
    minDacIndex = np.where(dacList >= minDac)[0][-1] if (minDac) else len(dacList) - 1

    axis.plot(data[maxDacIndex:minDacIndex, 0], data[maxDacIndex:minDacIndex, 1], marker='.')


# Plot a specific pixel.
def plot_pixel(filename, minDac=None, maxDac=None):
    make_plot_for_file(plt.subplot(111), filename, minDac, maxDac)
    plt.show()
    plt.close()


def make_combined_plot_of_all_pixels(fileNames, minDac=None, maxDac=None):
    for fileName in fileNames:
        make_plot_for_file(plt.subplot(), fileName, minDac, maxDac)

    plt.legend(list(map(lambda fileName: fileName.split('_')[4], fileNames)))
    plt.show()


def get_all_pixel_file_names():
    files = glob.glob('../data/Module1_VP0-0_ECS_Scan_Trim*_1of1_Pixel_1_1.csv')

    return list(map(lambda file: file.split('\\')[-1].split('.')[0], files))


fileNames = get_all_pixel_file_names()

# for fileName in fileNames:
#     plot_pixel(fileName)

make_combined_plot_of_all_pixels(fileNames, 1250, 1600)
